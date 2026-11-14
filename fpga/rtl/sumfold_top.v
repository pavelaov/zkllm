// sumfold_top.v : ZCU102 PL-only board top for SumFold. 125 MHz differential sysclk (G21/F21, LVDS_25) -> IBUFDS -> MMCME4_ADV -> BUFG -> kernel clock JTAG-to-AXI Master IP (module jtag_axi_0, AXI4-Lite, 32-bit address/data) -> sumfold_axil Everything runs in the kernel clock domain. Reset is released from the synchronized MMCM LOCKED. LEDs: [0] heartbeat, [1] busy, [2] done, [3] error, [4] MMCM locked, [7:5] off. MMCM settings (VCO 800..1600 MHz, PFD = 125 MHz with D = 1): KCLK_MHZ 100: M = 8,  O = 10 | 125: M = 8, O = 8 | 150: M = 12, O = 10
//   200: M = 8, O = 5            | 250: M = 12, O = 6 | 300: M = 12, O = 5
`timescale 1ns / 1ps

module sumfold_top #(
    parameter LANES    = 8,
    parameter LOG_NMAX = 16,
    parameter KCLK_MHZ = 250
) (
    input  wire       sysclk_p,
    input  wire       sysclk_n,
    output wire [7:0] led
);
    localparam integer MMCM_M = (KCLK_MHZ == 100 || KCLK_MHZ == 125 || KCLK_MHZ == 200) ? 8 : 12;
    localparam integer MMCM_O = (125 * MMCM_M) / KCLK_MHZ;
    localparam real    M_REAL = MMCM_M;
    localparam real    O_REAL = MMCM_O;

    // synthesis-time parameter check (elaboration fails on an unsupported clock)
    generate
        if (125 * MMCM_M != MMCM_O * KCLK_MHZ) begin : g_bad_kclk
            KCLK_MHZ_must_be_100_125_150_200_250_or_300 u_err ();
        end
    endgenerate

    // ------------------------------------------------------------ clocking
        // G21/F21 sit in a HIGH_DENSITY (HD) I/O bank, which has no MMCM site: an HD clock pin cannot drive an MMCM directly (DRC PLHDIO-4). The input clock therefore goes IBUFDS -> BUFG (global clock network) -> MMCME4_ADV CLKIN1. (Implementation fix, 2026-09-27, fpga_impl agent.)
    wire clk125_ibuf, clk125, clk_fb, clk_mmcm, kclk, mmcm_locked;
    IBUFDS #(.DIFF_TERM("FALSE"), .IOSTANDARD("LVDS_25")) u_ibufds (
        .I(sysclk_p), .IB(sysclk_n), .O(clk125_ibuf));
    BUFG u_bufg_in (.I(clk125_ibuf), .O(clk125));

    MMCME4_ADV #(
        .BANDWIDTH         ("OPTIMIZED"),
        .COMPENSATION      ("AUTO"),
        .DIVCLK_DIVIDE     (1),
        .CLKFBOUT_MULT_F   (M_REAL),
        .CLKFBOUT_PHASE    (0.0),
        .CLKIN1_PERIOD     (8.000),
        .CLKOUT0_DIVIDE_F  (O_REAL),
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE     (0.0),
        .STARTUP_WAIT      ("FALSE")
    ) u_mmcm (
        .CLKIN1      (clk125),
        .CLKIN2      (1'b0),
        .CLKINSEL    (1'b1),
        .CLKFBIN     (clk_fb),
        .CLKFBOUT    (clk_fb),
        .CLKFBOUTB   (),
        .CLKOUT0     (clk_mmcm),
        .CLKOUT0B    (),
        .CLKOUT1     (),
        .CLKOUT1B    (),
        .CLKOUT2     (),
        .CLKOUT2B    (),
        .CLKOUT3     (),
        .CLKOUT3B    (),
        .CLKOUT4     (),
        .CLKOUT5     (),
        .CLKOUT6     (),
        .LOCKED      (mmcm_locked),
        .PWRDWN      (1'b0),
        .RST         (1'b0),
        .DADDR       (7'd0),
        .DCLK        (1'b0),
        .DEN         (1'b0),
        .DI          (16'd0),
        .DO          (),
        .DRDY        (),
        .DWE         (1'b0),
        .PSCLK       (1'b0),
        .PSEN        (1'b0),
        .PSINCDEC    (1'b0),
        .PSDONE      (),
        .CDDCREQ     (1'b0),
        .CDDCDONE    (),
        .CLKFBSTOPPED(),
        .CLKINSTOPPED()
    );
    BUFG u_bufg_k (.I(clk_mmcm), .O(kclk));

    // ------------------------------------------------------------ reset from LOCKED
    (* ASYNC_REG = "TRUE" *) reg [2:0] lock_sync = 3'b000;
    reg [7:0] rst_cnt = 8'd0;
    reg       aresetn = 1'b0;
    always @(posedge kclk or negedge mmcm_locked) begin
        if (!mmcm_locked) begin
            lock_sync <= 3'b000;
            rst_cnt   <= 8'd0;
            aresetn   <= 1'b0;
        end else begin
            lock_sync <= {lock_sync[1:0], 1'b1};
            if (!lock_sync[2]) begin
                rst_cnt <= 8'd0;
                aresetn <= 1'b0;
            end else if (rst_cnt != 8'hFF) begin
                rst_cnt <= rst_cnt + 1'b1;
                aresetn <= 1'b0;
            end else begin
                aresetn <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------ JTAG-to-AXI master
    wire [31:0] awaddr, wdata, araddr, rdata;
    wire [2:0]  awprot, arprot;
    wire [3:0]  wstrb;
    wire [1:0]  bresp, rresp;
    wire        awvalid, awready, wvalid, wready, bvalid, bready;
    wire        arvalid, arready, rvalid, rready;

    jtag_axi_0 u_jtag_axi (
        .aclk         (kclk),
        .aresetn      (aresetn),
        .m_axi_awaddr (awaddr),
        .m_axi_awprot (awprot),
        .m_axi_awvalid(awvalid),
        .m_axi_awready(awready),
        .m_axi_wdata  (wdata),
        .m_axi_wstrb  (wstrb),
        .m_axi_wvalid (wvalid),
        .m_axi_wready (wready),
        .m_axi_bresp  (bresp),
        .m_axi_bvalid (bvalid),
        .m_axi_bready (bready),
        .m_axi_araddr (araddr),
        .m_axi_arprot (arprot),
        .m_axi_arvalid(arvalid),
        .m_axi_arready(arready),
        .m_axi_rdata  (rdata),
        .m_axi_rresp  (rresp),
        .m_axi_rvalid (rvalid),
        .m_axi_rready (rready)
    );

    // ------------------------------------------------------------ SumFold
    wire st_busy, st_done, st_error;
    sumfold_axil #(.LANES(LANES), .LOG_NMAX(LOG_NMAX), .KCLK_MHZ(KCLK_MHZ)) u_sumfold (
        .aclk         (kclk),
        .aresetn      (aresetn),
        .s_axi_awaddr (awaddr),
        .s_axi_awprot (awprot),
        .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),
        .s_axi_wdata  (wdata),
        .s_axi_wstrb  (wstrb),
        .s_axi_wvalid (wvalid),
        .s_axi_wready (wready),
        .s_axi_bresp  (bresp),
        .s_axi_bvalid (bvalid),
        .s_axi_bready (bready),
        .s_axi_araddr (araddr),
        .s_axi_arprot (arprot),
        .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),
        .s_axi_rdata  (rdata),
        .s_axi_rresp  (rresp),
        .s_axi_rvalid (rvalid),
        .s_axi_rready (rready),
        .o_busy       (st_busy),
        .o_done       (st_done),
        .o_error      (st_error)
    );

    // ------------------------------------------------------------ LEDs
    reg [26:0] hb = 27'd0;
    always @(posedge kclk) hb <= hb + 1'b1;

    reg [4:0] led_r = 5'd0;
    always @(posedge kclk) led_r <= {lock_sync[2], st_error, st_done, st_busy, hb[26]};
    assign led = {3'b000, led_r};
endmodule
