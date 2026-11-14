// sumfold_core.v : SumFold engine = LANES x sumfold_lane + adder tree + controller (spec/sumfold_kernel.md sections 3 and 4).
//
// Job (n = cfg_logn, R = n - log2(LANES) device rounds, R + 1 passes):
//   pass 0            : evaluate round 0                                   -> msgs[0]
//   pass i, 1..R-1    : fold with r_{i-1} (in place) and evaluate round i  -> msgs[i]
//   pass R            : final fold with r_{R-1}                            -> lane finals
// AUTO mode runs all passes back to back with challenges preloaded in the challenge RAM. STEP mode stops after every message (st_wait = 1); the host writes r_i and pulses cmd_start.
//
// Result RAM: index 3*i + k holds s_k of round i; index 48 + 2*l holds lane l final f, 48 + 2*l + 1 lane l final g. Words are {c1, c0}.
//
// Per-pass cycle count (cyc_ram[i]) = cycles spent in PINIT..PSTORE/PFIN for pass i:
//   eval passes : pairs(i) + EVAL_WAIT + 4 ; final pass : 2 + FOLD_WAIT + 2*LANES
// with pairs(i) = 2^(n - log2 L - max(i,1)).
`timescale 1ns / 1ps
`include "sumfold_defs.vh"

module sumfold_core #(
    parameter LANES     = 1,
    parameter LOG_NMAX  = 16,
    parameter DRAIN_ADJ = 0            // verification hook only; must be 0 in hardware builds
) (
    input  wire                 clk,
    input  wire                 rst,             // synchronous, active high
    // commands (single-cycle pulses)
    input  wire                 cmd_start,
    input  wire                 cmd_gen,
    input  wire                 cmd_soft_reset,
    // configuration
    input  wire                 cfg_step,        // 0 = AUTO, 1 = STEP (latched at job start)
    input  wire [7:0]           cfg_logn,
    input  wire [63:0]          cfg_seed_f,
    input  wire [63:0]          cfg_seed_g,
    // challenge RAM write port
    input  wire                 ch_we,
    input  wire [3:0]           ch_idx,
    input  wire [127:0]         ch_data,
    // result RAM read port (registered, 1-cycle latency)
    input  wire [6:0]           res_idx,
    output reg  [127:0]         res_data,
    // cycle counters
    input  wire [4:0]           cyc_idx,
    output reg  [31:0]          cyc_round,
    output reg  [63:0]          cyc_total,
    output reg  [63:0]          cyc_gen,
    output reg  [63:0]          cyc_wall,
    // direct table access (accepted only when idle); mem_addr = global element index
    input  wire                 mem_we,
    input  wire                 mem_re,
    input  wire                 mem_sel,         // 0 = f, 1 = g
    input  wire [LOG_NMAX-1:0]  mem_addr,
    input  wire [127:0]         mem_wdata,
    output reg  [127:0]         mem_rdata,
    output reg                  mem_rvalid,
    // status
    output wire                 st_busy,
    output reg                  st_done,
    output reg                  st_error,
    output wire                 st_wait,
    output reg                  st_gen_done,
    output reg  [7:0]           st_round
);
    localparam LG_L  = (LANES >= 8) ? 3 : (LANES >= 4) ? 2 : (LANES >= 2) ? 1 : 0;
    localparam AW    = LOG_NMAX - 1 - LG_L;     // bank address width
    localparam KW    = LOG_NMAX - LG_L;         // lane-local element index width
    localparam ACC_W = 64 + AW;                 // lazy accumulator width per lane
    localparam TW    = ACC_W + LG_L;            // after the lane adder tree
    localparam EVAL_WAIT = `SF_EVAL_LAT + LG_L + `GL_RED_LAT + DRAIN_ADJ;
    localparam FOLD_WAIT = `SF_FOLD_LAT + DRAIN_ADJ;
    localparam GFLUSH    = 4;

    localparam [1:0] M_EVAL0 = 2'd0, M_FUSED = 2'd1, M_FOLD = 2'd2, M_IDLE = 2'd3;
    localparam [3:0] S_IDLE   = 4'd0, S_GEN    = 4'd1, S_GFLUSH = 4'd2, S_PINIT = 4'd3,
                     S_PRUN   = 4'd4, S_PDRAIN = 4'd5, S_PSTORE = 4'd6, S_PFIN  = 4'd7,
                     S_WAIT   = 4'd8;

    // ------------------------------------------------------------ helpers
    function [63:0] xs64;                       // Marsaglia xorshift64 (13, 7, 17)
        input [63:0] x;
        reg   [63:0] t;
        begin
            t    = x ^ (x << 13);
            t    = t ^ (t >> 7);
            xs64 = t ^ (t << 17);
        end
    endfunction

    function [63:0] canon;                      // x mod p for x < 2^64 (x >= p <=> hi = 2^32-1, lo >= 1)
        input [63:0] x;
        begin
            canon = (x[63:32] == 32'hFFFF_FFFF && x[31:0] != 32'd0) ? {32'd0, x[31:0] - 32'd1} : x;
        end
    endfunction

    // ------------------------------------------------------------ state
    reg  [3:0]          state;
    reg  [4:0]          n_r;
    reg  [4:0]          lsh;                    // n - log2 L = R = log2(local elements)
    reg  [4:0]          lsh_mem;                // same, from cfg_logn, for MEM access
    reg  [4:0]          pass;
    reg                 step_r, pend_start;
    reg  [AW-1:0]       j, jlast;
    reg  [7:0]          dcnt;
    reg  [4:0]          scnt;
    reg  [31:0]         pcyc;
    reg  [LOG_NMAX:0]   gcnt;
    reg  [LOG_NMAX:0]   glast;                  // N - 1 of the GEN job (registered: keeps the
                                                // variable shift out of the GEN loop compare)
    reg  [63:0]         xf, xg;
    reg                 gv0;
    reg  [LOG_NMAX-1:0] gi0;

    // lane broadcast registers
    reg                 iss_v;
    reg  [AW-1:0]       iss_j;
    reg                 l_clr;
    reg  [1:0]          l_mode;
    reg  [127:0]        l_r;

    // external access stage (GEN / MEM)
    reg                 x_we_f, x_we_g, x_re, x_rsel;
    reg  [LOG_NMAX-1:0] x_lane;
    reg  [KW-1:0]       x_k;
    reg  [127:0]        x_wf, x_wg;

    // small RAMs (distributed)
    reg  [127:0]        ch_ram  [0:15];
    reg  [127:0]        res_ram [0:63];
    reg  [31:0]         cyc_ram [0:31];
    reg                 res_we;
    reg  [5:0]          res_wa;
    reg  [127:0]        res_wd;
    reg                 cr_we;
    reg  [4:0]          cr_wa;
    reg  [31:0]         cr_wd;

    wire ln_ok = (cfg_logn >= LG_L + 1) && (cfg_logn <= LOG_NMAX);

    // ------------------------------------------------------------ lanes
    wire [6*LANES*ACC_W-1:0] cbus;               // component c of lane l at ((c*LANES)+l)*ACC_W
    wire [LANES*128-1:0]     bus_fe, bus_fo, bus_ge, bus_go, bus_ff, bus_fg;
    wire                     l_rd_en   = iss_v | x_re;
    wire [AW-1:0]            l_rd_addr = iss_v ? iss_j : x_k[KW-1:1];

    genvar l, c;
    generate
        for (l = 0; l < LANES; l = l + 1) begin : g_lane
            wire [6*ACC_W-1:0] acc_l;
            sumfold_lane #(.AW(AW), .ACC_W(ACC_W)) u_lane (
                .clk      (clk),
                .clr      (l_clr),
                .mode     (l_mode),
                .r        (l_r),
                .rd_en    (l_rd_en),
                .rd_pass  (iss_v),
                .rd_addr  (l_rd_addr),
                .wr_f_en  (x_we_f && (x_lane == l)),
                .wr_g_en  (x_we_g && (x_lane == l)),
                .wr_bank  (x_k[0]),
                .wr_addr  (x_k[KW-1:1]),
                .wr_f_data(x_wf),
                .wr_g_data(x_wg),
                .rd_fe    (bus_fe[l*128 +: 128]),
                .rd_fo    (bus_fo[l*128 +: 128]),
                .rd_ge    (bus_ge[l*128 +: 128]),
                .rd_go    (bus_go[l*128 +: 128]),
                .acc      (acc_l),
                .fin_f    (bus_ff[l*128 +: 128]),
                .fin_g    (bus_fg[l*128 +: 128])
            );
            for (c = 0; c < 6; c = c + 1) begin : g_comp
                assign cbus[((c*LANES)+l)*ACC_W +: ACC_W] = acc_l[c*ACC_W +: ACC_W];
            end
        end
    endgenerate

    // ------------------------------------------------------------ adder tree + reduction
    wire [6*64-1:0] red_w;
    generate
        for (c = 0; c < 6; c = c + 1) begin : g_red
            wire [TW-1:0] ts;
            adder_tree #(.N(LANES), .LG(LG_L), .W(ACC_W)) u_tree (
                .clk(clk), .din(cbus[c*LANES*ACC_W +: LANES*ACC_W]), .dout(ts));
            gl_reduce128 u_red (.clk(clk), .x({{(128-TW){1'b0}}, ts}), .y(red_w[c*64 +: 64]));
        end
    endgenerate
    wire [127:0] msg_s0 = {red_w[1*64 +: 64], red_w[0*64 +: 64]};
    wire [127:0] msg_s1 = {red_w[3*64 +: 64], red_w[2*64 +: 64]};
    wire [127:0] msg_s2 = {red_w[5*64 +: 64], red_w[4*64 +: 64]};

    // ------------------------------------------------------------ external access stage
    wire [LOG_NMAX-1:0] esrc  = gv0 ? gi0 : mem_addr;
    wire [4:0]          esh   = gv0 ? lsh : lsh_mem;
    wire [LOG_NMAX-1:0] elane = esrc >> esh;
    wire [LOG_NMAX-1:0] ek    = esrc & ~({LOG_NMAX{1'b1}} << esh);
    wire                mem_ok = (state == S_IDLE);

    always @(posedge clk) begin
        lsh_mem <= cfg_logn[4:0] - LG_L;
        x_lane  <= elane;
        x_k     <= ek[KW-1:0];
        if (rst) begin
            x_we_f <= 1'b0;
            x_we_g <= 1'b0;
            x_re   <= 1'b0;
        end else begin
            x_we_f <= 1'b0;
            x_we_g <= 1'b0;
            x_re   <= 1'b0;
            if (gv0) begin
                x_we_f <= 1'b1;
                x_we_g <= 1'b1;
                x_wf   <= {64'd0, canon(xf)};
                x_wg   <= {64'd0, canon(xg)};
            end else if (mem_ok && mem_we) begin
                x_we_f <= ~mem_sel;
                x_we_g <=  mem_sel;
                x_wf   <= mem_wdata;
                x_wg   <= mem_wdata;
            end else if (mem_ok && mem_re) begin
                x_re   <= 1'b1;
                x_rsel <= mem_sel;
            end
        end
    end

    // MEM readback: lane read data is valid SF_S0_LAT cycles after x_re
    wire                rb_v, rb_sel, rb_bank;
    wire [LOG_NMAX-1:0] rb_lane;
    pipe_delay #(.W(3 + LOG_NMAX), .D(`SF_S0_LAT)) u_rb (
        .clk(clk), .d({x_re, x_rsel, x_k[0], x_lane}), .q({rb_v, rb_sel, rb_bank, rb_lane}));
    always @(posedge clk) begin
        if (rst) begin
            mem_rvalid <= 1'b0;
        end else begin
            mem_rvalid <= rb_v;
            if (rb_v) begin
                if (rb_lane >= LANES)
                    mem_rdata <= 128'd0;
                else case ({rb_sel, rb_bank})
                    2'b00:   mem_rdata <= bus_fe[rb_lane*128 +: 128];
                    2'b01:   mem_rdata <= bus_fo[rb_lane*128 +: 128];
                    2'b10:   mem_rdata <= bus_ge[rb_lane*128 +: 128];
                    default: mem_rdata <= bus_go[rb_lane*128 +: 128];
                endcase
            end
        end
    end

    // ------------------------------------------------------------ small RAM ports
    always @(posedge clk) begin
        if (ch_we)
            ch_ram[ch_idx] <= ch_data;
        if (res_we)
            res_ram[res_wa] <= res_wd;
        if (cr_we)
            cyc_ram[cr_wa] <= cr_wd;
        res_data  <= res_idx[6] ? 128'd0 : res_ram[res_idx[5:0]];
        cyc_round <= cyc_ram[cyc_idx];
    end

    // ------------------------------------------------------------ controller
    assign st_busy = (state != S_IDLE);
    assign st_wait = (state == S_WAIT);

    wire [4:0] pass_div = (pass == 5'd0) ? 5'd1 : pass;   // pairs(i) = 2^(lsh - max(i,1))

    always @(posedge clk) begin
        if (rst || cmd_soft_reset) begin
            state       <= S_IDLE;
            st_done     <= 1'b0;
            st_error    <= 1'b0;
            st_gen_done <= 1'b0;
            st_round    <= 8'd0;
            iss_v       <= 1'b0;
            l_clr       <= 1'b0;
            l_mode      <= M_IDLE;
            gv0         <= 1'b0;
            pend_start  <= 1'b0;
            res_we      <= 1'b0;
            cr_we       <= 1'b0;
            if (rst) begin
                cyc_total <= 64'd0;
                cyc_wall  <= 64'd0;
                cyc_gen   <= 64'd0;
            end
        end else begin
            iss_v  <= 1'b0;
            l_clr  <= 1'b0;
            gv0    <= 1'b0;
            res_we <= 1'b0;
            cr_we  <= 1'b0;
            case (state)
                S_IDLE: begin
                    l_mode <= M_IDLE;
                    if (cmd_gen) begin
                        if (!ln_ok || cfg_seed_f == 64'd0 || cfg_seed_g == 64'd0) begin
                            st_error <= 1'b1;
                        end else begin
                            st_error    <= 1'b0;
                            st_done     <= 1'b0;
                            st_gen_done <= 1'b0;
                            n_r         <= cfg_logn[4:0];
                            lsh         <= cfg_logn[4:0] - LG_L;
                            xf          <= cfg_seed_f;
                            xg          <= cfg_seed_g;
                            gcnt        <= 0;
                            glast       <= (1 << cfg_logn[4:0]) - 1;
                            cyc_gen     <= 64'd0;
                            pend_start  <= cmd_start;
                            step_r      <= cfg_step;
                            state       <= S_GEN;
                        end
                    end else if (cmd_start) begin
                        if (!ln_ok) begin
                            st_error <= 1'b1;
                        end else begin
                            st_error  <= 1'b0;
                            st_done   <= 1'b0;
                            n_r       <= cfg_logn[4:0];
                            lsh       <= cfg_logn[4:0] - LG_L;
                            step_r    <= cfg_step;
                            pass      <= 5'd0;
                            cyc_total <= 64'd0;
                            cyc_wall  <= 64'd0;
                            state     <= S_PINIT;
                        end
                    end
                end

                S_GEN: begin
                    // after this edge xf = x_{gcnt+1}, the value of element gcnt
                    xf      <= xs64(xf);
                    xg      <= xs64(xg);
                    gv0     <= 1'b1;
                    gi0     <= gcnt[LOG_NMAX-1:0];
                    gcnt    <= gcnt + 1'b1;
                    cyc_gen <= cyc_gen + 1'b1;
                    if (gcnt == glast) begin
                        dcnt  <= 8'd0;
                        state <= S_GFLUSH;
                    end
                end

                S_GFLUSH: begin
                    cyc_gen <= cyc_gen + 1'b1;
                    dcnt    <= dcnt + 1'b1;
                    if (dcnt == GFLUSH - 1) begin
                        st_gen_done <= 1'b1;
                        pend_start  <= 1'b0;
                        if (pend_start) begin
                            pass      <= 5'd0;
                            cyc_total <= 64'd0;
                            cyc_wall  <= 64'd0;
                            state     <= S_PINIT;
                        end else begin
                            state <= S_IDLE;
                        end
                    end
                end

                S_PINIT: begin
                    l_clr     <= 1'b1;
                    l_mode    <= (pass == 5'd0) ? M_EVAL0 : ((pass == lsh) ? M_FOLD : M_FUSED);
                    l_r       <= ch_ram[pass[3:0] - 4'd1];
                    jlast     <= (1 << (lsh - pass_div)) - 1;
                    j         <= {AW{1'b0}};
                    pcyc      <= 32'd1;
                    cyc_total <= cyc_total + 1'b1;
                    cyc_wall  <= cyc_wall + 1'b1;
                    state     <= S_PRUN;
                end

                S_PRUN: begin
                    iss_v     <= 1'b1;
                    iss_j     <= j;
                    j         <= j + 1'b1;
                    pcyc      <= pcyc + 1'b1;
                    cyc_total <= cyc_total + 1'b1;
                    cyc_wall  <= cyc_wall + 1'b1;
                    if (j == jlast) begin
                        dcnt  <= 8'd0;
                        state <= S_PDRAIN;
                    end
                end

                S_PDRAIN: begin
                    dcnt      <= dcnt + 1'b1;
                    pcyc      <= pcyc + 1'b1;
                    cyc_total <= cyc_total + 1'b1;
                    cyc_wall  <= cyc_wall + 1'b1;
                    scnt      <= 5'd0;
                    if (pass == lsh) begin
                        if (dcnt == FOLD_WAIT - 1)
                            state <= S_PFIN;
                    end else begin
                        if (dcnt == EVAL_WAIT - 1)
                            state <= S_PSTORE;
                    end
                end

                S_PSTORE: begin
                    res_we    <= 1'b1;
                    // s2 (the last accumulator to settle) is stored first, so EVAL_WAIT is exact
                    res_wa    <= 6'd3 * pass + ((scnt == 5'd0) ? 6'd2 : (scnt == 5'd1) ? 6'd0 : 6'd1);
                    res_wd    <= (scnt == 5'd0) ? msg_s2 : (scnt == 5'd1) ? msg_s0 : msg_s1;
                    scnt      <= scnt + 1'b1;
                    pcyc      <= pcyc + 1'b1;
                    cyc_total <= cyc_total + 1'b1;
                    cyc_wall  <= cyc_wall + 1'b1;
                    if (scnt == 5'd2) begin
                        cr_we    <= 1'b1;
                        cr_wa    <= pass;
                        cr_wd    <= pcyc + 1'b1;
                        st_round <= {3'd0, pass};
                        if (step_r) begin
                            state <= S_WAIT;
                        end else begin
                            pass  <= pass + 1'b1;
                            state <= S_PINIT;
                        end
                    end
                end

                S_PFIN: begin
                    res_we    <= 1'b1;
                    res_wa    <= 6'd48 + scnt;
                    res_wd    <= scnt[0] ? bus_fg[(scnt >> 1)*128 +: 128] : bus_ff[(scnt >> 1)*128 +: 128];
                    scnt      <= scnt + 1'b1;
                    pcyc      <= pcyc + 1'b1;
                    cyc_total <= cyc_total + 1'b1;
                    cyc_wall  <= cyc_wall + 1'b1;
                    if (scnt == 2 * LANES - 1) begin
                        cr_we    <= 1'b1;
                        cr_wa    <= pass;
                        cr_wd    <= pcyc + 1'b1;
                        st_round <= {3'd0, pass};
                        st_done  <= 1'b1;
                        l_mode   <= M_IDLE;
                        state    <= S_IDLE;
                    end
                end

                S_WAIT: begin
                    cyc_wall <= cyc_wall + 1'b1;
                    if (cmd_start) begin
                        pass  <= pass + 1'b1;
                        state <= S_PINIT;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
