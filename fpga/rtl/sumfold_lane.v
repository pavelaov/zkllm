// sumfold_lane.v : one SumFold lane (spec/sumfold_kernel.md, sections 3.1 and 3.2).
//
// Memory: per vector (f, g) an EVEN bank (v[2j] at address j) and an ODD bank (v[2j+1] at address j), each 2^AW x 128 bit. Resident data = 4 * 2^AW * 128 bits per lane.
//
// One pair (v[2j], v[2j+1]) of each vector is read per cycle (II = 1). Pass modes:
//   M_EVAL0 (pass 0)       : evaluate round 0 on the stored pairs
// s0 += f0*g0 (unit C), s1 += f1*g1 (unit A), s2 += (2f1-f0)(2g1-g0) (unit D) M_FUSED (pass 1..R-1)  : fold with r_{i-1}: v'[j] = v[2j] + r*(v[2j+1]-v[2j]) (units A, B), write v'[j] in place to bank (j & 1), address j >> 1, and feed v'[j] into the round-i accumulators: unit C accumulates f'[j]*g'[j] into s0 (j even) or s1 (j odd), unit D the s2 product of the pair (j-1, j).
//   M_FOLD  (pass R)       : final fold only; the single output pair is the lane final (fin_f, fin_g).
// The write address j >> 1 always trails the read address, so folding in place is safe.
//
// Accumulation is lazy: each base-field component is summed as a plain (64 + AW)-bit integer and reduced mod p once per pass in sumfold_core (exactly equal to the mod-p sum).
//
// Timing (cycles after the core issue register holds address j): S0 = SF_S0_LAT (read data), S1 = SF_S1_LAT (folded element), accumulators final at SF_EVAL_LAT.
`timescale 1ns / 1ps
`include "sumfold_defs.vh"

module sumfold_lane #(
    parameter AW    = 12,
    parameter ACC_W = 64 + 12
) (
    input  wire                 clk,
    // pass control (from sumfold_core; registered here)
    input  wire                 clr,
    input  wire [1:0]           mode,
    input  wire [127:0]         r,
    input  wire                 rd_en,
    input  wire                 rd_pass,
    input  wire [AW-1:0]        rd_addr,
    // external writes (GEN / MEM)
    input  wire                 wr_f_en,
    input  wire                 wr_g_en,
    input  wire                 wr_bank,
    input  wire [AW-1:0]        wr_addr,
    input  wire [127:0]         wr_f_data,
    input  wire [127:0]         wr_g_data,
    // raw read data at S0 (MEM readback)
    output wire [127:0]         rd_fe,
    output wire [127:0]         rd_fo,
    output wire [127:0]         rd_ge,
    output wire [127:0]         rd_go,
    // accumulators {s2.c1, s2.c0, s1.c1, s1.c0, s0.c1, s0.c0}
    output wire [6*ACC_W-1:0]   acc,
    // lane finals (valid after the M_FOLD pass)
    output reg  [127:0]         fin_f,
    output reg  [127:0]         fin_g
);
    localparam [1:0] M_EVAL0 = 2'd0, M_FUSED = 2'd1, M_FOLD = 2'd2, M_IDLE = 2'd3;
    localparam LE = `EXT_MUL_LAT;

    // ------------------------------------------------------------ input registers
    reg            clr_q;
    reg  [1:0]     mode_q;
    reg  [127:0]   r_q;
    reg            rv_q;
    (* max_fanout = 64 *) reg [AW-1:0] ra_q;
    reg            wr_f_q, wr_g_q, wr_bank_q;
    reg  [AW-1:0]  wr_addr_q;
    reg  [127:0]   wr_fd_q, wr_gd_q;
    always @(posedge clk) begin
        clr_q     <= clr;
        mode_q    <= mode;
        r_q       <= r;
        rv_q      <= rd_en & rd_pass;
        ra_q      <= rd_addr;
        wr_f_q    <= wr_f_en;
        wr_g_q    <= wr_g_en;
        wr_bank_q <= wr_bank;
        wr_addr_q <= wr_addr;
        wr_fd_q   <= wr_f_data;
        wr_gd_q   <= wr_g_data;
    end

    wire is_eval0 = (mode_q == M_EVAL0);
    wire is_fused = (mode_q == M_FUSED);
    wire is_final = (mode_q == M_FOLD);
    wire is_fold  = is_fused | is_final;

    // ------------------------------------------------------------ memories
    wire [127:0] fnew, gnew;
    wire         v1;
    wire [AW-1:0] j1;

    reg            m_we_fe, m_we_fo, m_we_ge, m_we_go;
    (* max_fanout = 64 *) reg [AW-1:0] m_waddr;
    reg  [127:0]   m_wd_f, m_wd_g;
    always @(posedge clk) begin
        if (v1 & is_fold) begin
            // in-place write-back of v'[j] to bank (j & 1), address j >> 1
            m_we_fe <= ~j1[0];
            m_we_fo <=  j1[0];
            m_we_ge <= ~j1[0];
            m_we_go <=  j1[0];
            m_waddr <= j1 >> 1;
            m_wd_f  <= fnew;
            m_wd_g  <= gnew;
        end else begin
            m_we_fe <= wr_f_q & ~wr_bank_q;
            m_we_fo <= wr_f_q &  wr_bank_q;
            m_we_ge <= wr_g_q & ~wr_bank_q;
            m_we_go <= wr_g_q &  wr_bank_q;
            m_waddr <= wr_addr_q;
            m_wd_f  <= wr_fd_q;
            m_wd_g  <= wr_gd_q;
        end
    end

    wire [127:0] fe_m, fo_m, ge_m, go_m;
    sdp_bram #(.DW(128), .AW(AW)) u_fe (.clk(clk), .we(m_we_fe), .waddr(m_waddr), .wdata(m_wd_f), .raddr(ra_q), .rdata(fe_m));
    sdp_bram #(.DW(128), .AW(AW)) u_fo (.clk(clk), .we(m_we_fo), .waddr(m_waddr), .wdata(m_wd_f), .raddr(ra_q), .rdata(fo_m));
    sdp_bram #(.DW(128), .AW(AW)) u_ge (.clk(clk), .we(m_we_ge), .waddr(m_waddr), .wdata(m_wd_g), .raddr(ra_q), .rdata(ge_m));
    sdp_bram #(.DW(128), .AW(AW)) u_go (.clk(clk), .we(m_we_go), .waddr(m_waddr), .wdata(m_wd_g), .raddr(ra_q), .rdata(go_m));

    // fabric register -> S0
    reg [127:0] fe, fo, ge, go;
    always @(posedge clk) begin
        fe <= fe_m;
        fo <= fo_m;
        ge <= ge_m;
        go <= go_m;
    end
    assign rd_fe = fe;
    assign rd_fo = fo;
    assign rd_ge = ge;
    assign rd_go = go;

    wire          v0;
    wire [AW-1:0] j0;
    pipe_delay #(.W(1 + AW), .D(`RAM_RD_LAT)) u_meta0 (.clk(clk), .d({rv_q, ra_q}), .q({v0, j0}));

    // ------------------------------------------------------------ fold path (units A, B)
    wire [127:0] df, dg;
    ext_sub u_df (.clk(clk), .a(fo), .b(fe), .y(df));            // S0 + 2
    ext_sub u_dg (.clk(clk), .a(go), .b(ge), .y(dg));

    // unit A doubles as the s1 multiplier in pass 0
    reg [127:0] ua_x, ua_y, ub_x, ub_y;
    reg         ua_ev;
    always @(posedge clk) begin
        ua_x  <= is_eval0 ? fo : r_q;
        ua_y  <= is_eval0 ? go : df;
        ua_ev <= is_eval0 & v0;
        ub_x  <= r_q;
        ub_y  <= dg;
    end
    wire [127:0] a_out, b_out;
    wire         a_ev;
    ext_mul u_A (.clk(clk), .a(ua_x), .b(ua_y), .y(a_out));
    ext_mul u_B (.clk(clk), .a(ub_x), .b(ub_y), .y(b_out));
    pipe_delay #(.W(1), .D(LE)) u_aev (.clk(clk), .d(ua_ev), .q(a_ev));

    wire [127:0] fe_d, ge_d;
    pipe_delay #(.W(256), .D(`EXT_SUB_LAT + 1 + LE)) u_dfe (.clk(clk), .d({fe, ge}), .q({fe_d, ge_d}));
    ext_add u_nf (.clk(clk), .a(fe_d), .b(a_out), .y(fnew));      // S1
    ext_add u_ng (.clk(clk), .a(ge_d), .b(b_out), .y(gnew));
    pipe_delay #(.W(1 + AW), .D(`SF_S1_LAT - `SF_S0_LAT)) u_meta1 (.clk(clk), .d({v0, j0}), .q({v1, j1}));

    always @(posedge clk) begin
        if (v1 & is_final) begin
            fin_f <= fnew;
            fin_g <= gnew;
        end
    end

    // ------------------------------------------------------------ evaluation: unit C (products)
    reg [127:0] uc_x, uc_y;
    reg         uc_v, uc_t;       // tag: 0 -> s0, 1 -> s1
    always @(posedge clk) begin
        uc_x <= is_eval0 ? fe : fnew;
        uc_y <= is_eval0 ? ge : gnew;
        uc_v <= is_eval0 ? v0 : (v1 & is_fused);
        uc_t <= is_eval0 ? 1'b0 : j1[0];
    end
    wire [127:0] c_out;
    wire         c_v, c_t;
    ext_mul u_C (.clk(clk), .a(uc_x), .b(uc_y), .y(c_out));
    pipe_delay #(.W(2), .D(LE)) u_cvt (.clk(clk), .d({uc_v, uc_t}), .q({c_v, c_t}));

    // ------------------------------------------------------------ evaluation: unit D (s2)
    reg [127:0] hf, hg;           // even element of the folded pair
    always @(posedge clk) begin
        if (v1 & ~j1[0]) begin
            hf <= fnew;
            hg <= gnew;
        end
    end
    reg [127:0] p_x0f, p_x1f, p_x0g, p_x1g;
    reg         p_v;
    always @(posedge clk) begin
        p_x0f <= is_eval0 ? fe : hf;
        p_x1f <= is_eval0 ? fo : fnew;
        p_x0g <= is_eval0 ? ge : hg;
        p_x1g <= is_eval0 ? go : gnew;
        p_v   <= is_eval0 ? v0 : (v1 & j1[0] & is_fused);
    end
    wire [127:0] dblf, dblg, x0f_d, x0g_d, tf, tg;
    ext_add u_d2f (.clk(clk), .a(p_x1f), .b(p_x1f), .y(dblf));
    ext_add u_d2g (.clk(clk), .a(p_x1g), .b(p_x1g), .y(dblg));
    pipe_delay #(.W(256), .D(`EXT_ADD_LAT)) u_dx0 (.clk(clk), .d({p_x0f, p_x0g}), .q({x0f_d, x0g_d}));
    ext_sub u_tf (.clk(clk), .a(dblf), .b(x0f_d), .y(tf));        // 2 f1 - f0
    ext_sub u_tg (.clk(clk), .a(dblg), .b(x0g_d), .y(tg));        // 2 g1 - g0
    wire d_in_v;
    pipe_delay #(.W(1), .D(`EXT_ADD_LAT + `EXT_SUB_LAT)) u_dv0 (.clk(clk), .d(p_v), .q(d_in_v));
    wire [127:0] d_out;
    wire         d_v;
    ext_mul u_D (.clk(clk), .a(tf), .b(tg), .y(d_out));
    pipe_delay #(.W(1), .D(LE)) u_dv1 (.clk(clk), .d(d_in_v), .q(d_v));

    // ------------------------------------------------------------ lazy accumulators
    reg [127:0] s0_d, s1_d, s2_d;
    reg         s0_v, s1_v, s2_v;
    always @(posedge clk) begin
        s0_v <= c_v & ~c_t;
        s0_d <= c_out;
        s1_v <= a_ev | (c_v & c_t);
        s1_d <= a_ev ? a_out : c_out;
        s2_v <= d_v;
        s2_d <= d_out;
    end

    reg [ACC_W-1:0] acc0l, acc0h, acc1l, acc1h, acc2l, acc2h;
    always @(posedge clk) begin
        if (clr_q) begin
            acc0l <= {ACC_W{1'b0}};
            acc0h <= {ACC_W{1'b0}};
            acc1l <= {ACC_W{1'b0}};
            acc1h <= {ACC_W{1'b0}};
            acc2l <= {ACC_W{1'b0}};
            acc2h <= {ACC_W{1'b0}};
        end else begin
            if (s0_v) begin
                acc0l <= acc0l + s0_d[63:0];
                acc0h <= acc0h + s0_d[127:64];
            end
            if (s1_v) begin
                acc1l <= acc1l + s1_d[63:0];
                acc1h <= acc1h + s1_d[127:64];
            end
            if (s2_v) begin
                acc2l <= acc2l + s2_d[63:0];
                acc2h <= acc2h + s2_d[127:64];
            end
        end
    end
    assign acc = {acc2h, acc2l, acc1h, acc1l, acc0h, acc0l};
endmodule
