// sumfold_axil.v : AXI4-Lite slave (32-bit) implementing the register map of spec/sumfold_kernel.md section 5 around sumfold_core. One outstanding transaction per channel; writes are full-word (WSTRB ignored, the JTAG-to-AXI master always writes 32 bits).
//
// Offset  Name            Access  Meaning
// 0x000   ID              R       0xEDCE2026
// 0x004   CFG             R       [7:0] LANES, [15:8] LOG_NMAX, [31:16] KCLK_MHZ (spec: [23:16]; a
//                                 16-bit field is used so that 300 MHz is representable)
// 0x008   CTRL            RW      W: bit0 START, bit1 GEN, bit2 SOFT_RESET (pulses), bit4 MODE (0 AUTO,1 STEP)
// 0x00C   STATUS          R       bit0 BUSY, bit1 DONE, bit2 ERROR, bit3 WAIT (STEP: message ready),
//                                 bit4 GEN_DONE
// 0x010   LOGN            RW      n for the current job
// 0x014   ROUND           R       index of the last completed pass
// 0x018   SEED_F_LO/HI    RW      (0x018, 0x01C)
// 0x020   SEED_G_LO/HI    RW      (0x020, 0x024)
// 0x040   CH_IDX          RW      challenge RAM index (0..15)
// 0x044   CH_W0..W3       RW      W0 = c0[31:0], W1 = c0[63:32], W2 = c1[31:0], W3 = c1[63:32]
// 0x054   CH_WE           W       bit0: write {W3,W2,W1,W0} to CH[CH_IDX]
// 0x060   RES_IDX         RW      3*round + k selects s_k; 48 + 2*l (+1) selects lane l final f (g)
// 0x064   RES_W0..W3      R       selected result word
// 0x080   CYC_TOTAL_LO/HI R       device-busy cycles of the last job (pass states only)
// 0x088   CYC_GEN_LO/HI   R       cycles of the last GEN
// 0x090   CYC_ROUND_IDX   RW      pass index
// 0x094   CYC_ROUND       R       cycles of the selected pass
// 0x098   CYC_WALL_LO/HI  R       (extension) START to DONE including STEP host waits
// 0x0A0   MEM_ADDR        RW      global element index
// 0x0A4   MEM_SEL         RW      0 = f, 1 = g
// 0x0A8   MEM_W0..W3      RW      element word
// 0x0B8   MEM_WE          W       bit0: write element, bit1: read element into MEM_R
// 0x0C0   MEM_R0..R3      R       (extension) read-back word, 0x0D0 MEM_RVALID count
// 0x0F0   KCLK_MHZ        R       (extension) build kernel clock in MHz
`timescale 1ns / 1ps

module sumfold_axil #(
    parameter LANES    = 1,
    parameter LOG_NMAX = 16,
    parameter KCLK_MHZ = 250
) (
    input  wire        aclk,
    input  wire        aresetn,
    // write address
    input  wire [31:0] s_axi_awaddr,
    input  wire [2:0]  s_axi_awprot,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    // write data
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    // write response
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    // read address
    input  wire [31:0] s_axi_araddr,
    input  wire [2:0]  s_axi_arprot,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    // read data
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,
    // status for LEDs
    output wire        o_busy,
    output wire        o_done,
    output wire        o_error
);
    localparam [31:0] ID_VALUE = 32'hEDCE_2026;
    localparam [15:0] KCLK16   = KCLK_MHZ;
    localparam [31:0] KCLK32   = KCLK_MHZ;
    localparam [7:0]  LANES8   = LANES;
    localparam [7:0]  LOGN8    = LOG_NMAX;
    wire rst = ~aresetn;

    // ------------------------------------------------------------ register file
    reg        r_mode;
    reg [7:0]  r_logn;
    reg [63:0] r_seed_f, r_seed_g;
    reg [3:0]  r_ch_idx;
    reg [31:0] r_ch_w [0:3];
    reg [6:0]  r_res_idx;
    reg [4:0]  r_cyc_idx;
    reg [31:0] r_mem_addr;
    reg        r_mem_sel;
    reg [31:0] r_mem_w [0:3];
    reg [31:0] r_mem_rcnt;

    reg        p_start, p_gen, p_soft, p_ch_we, p_mem_we, p_mem_re;

    // ------------------------------------------------------------ core
    wire [127:0] res_data, mem_rdata;
    wire [31:0]  cyc_round;
    wire [63:0]  cyc_total, cyc_gen, cyc_wall;
    wire         mem_rvalid;
    wire         st_busy, st_done, st_error, st_wait, st_gen_done;
    wire [7:0]   st_round;

    sumfold_core #(.LANES(LANES), .LOG_NMAX(LOG_NMAX)) u_core (
        .clk           (aclk),
        .rst           (rst),
        .cmd_start     (p_start),
        .cmd_gen       (p_gen),
        .cmd_soft_reset(p_soft),
        .cfg_step      (r_mode),
        .cfg_logn      (r_logn),
        .cfg_seed_f    (r_seed_f),
        .cfg_seed_g    (r_seed_g),
        .ch_we         (p_ch_we),
        .ch_idx        (r_ch_idx),
        .ch_data       ({r_ch_w[3], r_ch_w[2], r_ch_w[1], r_ch_w[0]}),
        .res_idx       (r_res_idx),
        .res_data      (res_data),
        .cyc_idx       (r_cyc_idx),
        .cyc_round     (cyc_round),
        .cyc_total     (cyc_total),
        .cyc_gen       (cyc_gen),
        .cyc_wall      (cyc_wall),
        .mem_we        (p_mem_we),
        .mem_re        (p_mem_re),
        .mem_sel       (r_mem_sel),
        .mem_addr      (r_mem_addr[LOG_NMAX-1:0]),
        .mem_wdata     ({r_mem_w[3], r_mem_w[2], r_mem_w[1], r_mem_w[0]}),
        .mem_rdata     (mem_rdata),
        .mem_rvalid    (mem_rvalid),
        .st_busy       (st_busy),
        .st_done       (st_done),
        .st_error      (st_error),
        .st_wait       (st_wait),
        .st_gen_done   (st_gen_done),
        .st_round      (st_round)
    );
    assign o_busy  = st_busy;
    assign o_done  = st_done;
    assign o_error = st_error;

    reg [127:0] mem_rbuf;
    always @(posedge aclk) begin
        if (rst) begin
            mem_rbuf   <= 128'd0;
            r_mem_rcnt <= 32'd0;
        end else if (mem_rvalid) begin
            mem_rbuf   <= mem_rdata;
            r_mem_rcnt <= r_mem_rcnt + 1'b1;
        end
    end

    // ------------------------------------------------------------ write channel
    wire wr_fire = s_axi_awvalid & s_axi_awready & s_axi_wvalid & s_axi_wready;
    wire [7:0] wa = s_axi_awaddr[7:0];

    always @(posedge aclk) begin
        if (rst) begin
            s_axi_awready <= 1'b0;
            s_axi_wready  <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_bresp   <= 2'b00;
            p_start  <= 1'b0; p_gen    <= 1'b0; p_soft   <= 1'b0;
            p_ch_we  <= 1'b0; p_mem_we <= 1'b0; p_mem_re <= 1'b0;
            r_mode   <= 1'b0;
            r_logn   <= 8'd0;
            r_seed_f <= 64'd1;
            r_seed_g <= 64'd2;
            r_ch_idx <= 4'd0;
            r_res_idx <= 7'd0;
            r_cyc_idx <= 5'd0;
            r_mem_addr <= 32'd0;
            r_mem_sel  <= 1'b0;
        end else begin
            p_start  <= 1'b0; p_gen    <= 1'b0; p_soft   <= 1'b0;
            p_ch_we  <= 1'b0; p_mem_we <= 1'b0; p_mem_re <= 1'b0;

            if (!s_axi_awready && s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid) begin
                s_axi_awready <= 1'b1;
                s_axi_wready  <= 1'b1;
            end else begin
                s_axi_awready <= 1'b0;
                s_axi_wready  <= 1'b0;
            end

            if (wr_fire) begin
                s_axi_bvalid <= 1'b1;
                s_axi_bresp  <= 2'b00;
                case (wa)
                    8'h08: begin
                        r_mode  <= s_axi_wdata[4];
                        p_start <= s_axi_wdata[0];
                        p_gen   <= s_axi_wdata[1];
                        p_soft  <= s_axi_wdata[2];
                    end
                    8'h10: r_logn          <= s_axi_wdata[7:0];
                    8'h18: r_seed_f[31:0]  <= s_axi_wdata;
                    8'h1C: r_seed_f[63:32] <= s_axi_wdata;
                    8'h20: r_seed_g[31:0]  <= s_axi_wdata;
                    8'h24: r_seed_g[63:32] <= s_axi_wdata;
                    8'h40: r_ch_idx        <= s_axi_wdata[3:0];
                    8'h54: p_ch_we         <= s_axi_wdata[0];
                    8'h60: r_res_idx       <= s_axi_wdata[6:0];
                    8'h90: r_cyc_idx       <= s_axi_wdata[4:0];
                    8'hA0: r_mem_addr      <= s_axi_wdata;
                    8'hA4: r_mem_sel       <= s_axi_wdata[0];
                    8'hB8: begin
                        p_mem_we <= s_axi_wdata[0];
                        p_mem_re <= s_axi_wdata[1] & ~s_axi_wdata[0];
                    end
                    default: ;
                endcase
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end
        end
    end

    // array registers (no reset needed)
    always @(posedge aclk) begin
        if (wr_fire) begin
            case (wa)
                8'h44: r_ch_w[0]  <= s_axi_wdata;
                8'h48: r_ch_w[1]  <= s_axi_wdata;
                8'h4C: r_ch_w[2]  <= s_axi_wdata;
                8'h50: r_ch_w[3]  <= s_axi_wdata;
                8'hA8: r_mem_w[0] <= s_axi_wdata;
                8'hAC: r_mem_w[1] <= s_axi_wdata;
                8'hB0: r_mem_w[2] <= s_axi_wdata;
                8'hB4: r_mem_w[3] <= s_axi_wdata;
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------ read channel
    wire [7:0] ra = s_axi_araddr[7:0];
    reg  [31:0] rmux;
    always @(*) begin
        case (ra)
            8'h00: rmux = ID_VALUE;
            8'h04: rmux = {KCLK16, LOGN8, LANES8};
            8'h08: rmux = {27'd0, r_mode, 4'd0};
            8'h0C: rmux = {27'd0, st_gen_done, st_wait, st_error, st_done, st_busy};
            8'h10: rmux = {24'd0, r_logn};
            8'h14: rmux = {24'd0, st_round};
            8'h18: rmux = r_seed_f[31:0];
            8'h1C: rmux = r_seed_f[63:32];
            8'h20: rmux = r_seed_g[31:0];
            8'h24: rmux = r_seed_g[63:32];
            8'h40: rmux = {28'd0, r_ch_idx};
            8'h44: rmux = r_ch_w[0];
            8'h48: rmux = r_ch_w[1];
            8'h4C: rmux = r_ch_w[2];
            8'h50: rmux = r_ch_w[3];
            8'h60: rmux = {25'd0, r_res_idx};
            8'h64: rmux = res_data[31:0];
            8'h68: rmux = res_data[63:32];
            8'h6C: rmux = res_data[95:64];
            8'h70: rmux = res_data[127:96];
            8'h80: rmux = cyc_total[31:0];
            8'h84: rmux = cyc_total[63:32];
            8'h88: rmux = cyc_gen[31:0];
            8'h8C: rmux = cyc_gen[63:32];
            8'h90: rmux = {27'd0, r_cyc_idx};
            8'h94: rmux = cyc_round;
            8'h98: rmux = cyc_wall[31:0];
            8'h9C: rmux = cyc_wall[63:32];
            8'hA0: rmux = r_mem_addr;
            8'hA4: rmux = {31'd0, r_mem_sel};
            8'hA8: rmux = r_mem_w[0];
            8'hAC: rmux = r_mem_w[1];
            8'hB0: rmux = r_mem_w[2];
            8'hB4: rmux = r_mem_w[3];
            8'hC0: rmux = mem_rbuf[31:0];
            8'hC4: rmux = mem_rbuf[63:32];
            8'hC8: rmux = mem_rbuf[95:64];
            8'hCC: rmux = mem_rbuf[127:96];
            8'hD0: rmux = r_mem_rcnt;
            8'hF0: rmux = KCLK32;
            default: rmux = 32'd0;
        endcase
    end

    always @(posedge aclk) begin
        if (rst) begin
            s_axi_arready <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rresp   <= 2'b00;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (!s_axi_arready && s_axi_arvalid && !s_axi_rvalid)
                s_axi_arready <= 1'b1;
            else
                s_axi_arready <= 1'b0;

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;
                s_axi_rdata  <= rmux;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
endmodule
