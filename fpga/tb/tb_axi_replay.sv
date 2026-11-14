// tb_axi_replay.sv : replays a recorded AXI4-Lite register trace against sumfold_axil + sumfold_core, i.e. the exact register transactions that the JTAG-to-AXI master issues on the board.
//
// Trace file (+TRACE=<file>), one operation per line, hex fields:
//   W <addr> <data>   write
//   R <addr> <x>      read, prints "RD <addr> <data>"
//   P <addr> <mask>   poll: read until (data & mask) != 0 or data[2] (ERROR) is set, prints the final
// "RD <addr> <data>" (one line per poll, like poll_status in the board script) The trace is produced from the generated board Tcl by fpga/sim/replay/replay.py, which also plays the RD values back through the same Tcl script to obtain the SFB_* lines of a board run.
//
// Bus behaviour (+WMODE=<k>): 0 = AWVALID and WVALID together (default), 1 = WVALID k=+WDLY cycles after AWVALID, 2 = AWVALID +WDLY cycles after WVALID. BREADY/RREADY are held high like the JTAG-to-AXI master. +GAP=<k> idle cycles between transactions (default 8). Compile with -d NETLIST_DUT when sumfold_axil is a post-synthesis/post-route funcsim netlist (the +POWERON0 hierarchy references are RTL-only; a netlist models power-on state through INIT/GSR).
`timescale 1ns / 1ps

module tb_axi_replay;
    parameter LANES    = 8;
    parameter LOG_NMAX = 16;
    parameter KCLK_MHZ = 250;

    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg aresetn = 1'b0;

    reg  [31:0] awaddr = 0, wdata = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, arvalid = 0;
    wire        bready = 1'b1, rready = 1'b1;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;
    wire        o_busy, o_done, o_error;

`ifdef NETLIST_DUT
    sumfold_axil dut (                     // netlist: parameters are already bound
`else
    sumfold_axil #(.LANES(LANES), .LOG_NMAX(LOG_NMAX), .KCLK_MHZ(KCLK_MHZ)) dut (
`endif
        .aclk(clk), .aresetn(aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b000), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b000), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .o_busy(o_busy), .o_done(o_done), .o_error(o_error));

    integer wmode = 0, wdly = 2, gap = 8, nerr = 0;

        // +POWERON0: model the FPGA power-on state of storage that the RTL never initializes but that the configured device holds at zero (LUTRAM/BRAM INIT = 0): the challenge RAM (pass 0 reads ch_ram[15] as a don't-care r) and every lane memory. Only needed when a post-synthesis netlist lets idle-cycle values leak into results (without it such leaks show up as X).
    integer pz = 0, zi;
`ifndef NETLIST_DUT
    initial begin
        if ($value$plusargs("POWERON0=%d", pz) && pz != 0) begin
            for (zi = 0; zi < 16; zi++) dut.u_core.ch_ram[zi] = 128'd0;
            for (zi = 0; zi < 64; zi++) dut.u_core.res_ram[zi] = 128'd0;
            for (zi = 0; zi < 32; zi++) dut.u_core.cyc_ram[zi] = 32'd0;
            $display("POWERON0 applied");
        end
    end
    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl++) begin : g_pz
            integer k;
            initial begin
                #0;
                if (pz != 0) begin
                    for (k = 0; k < (1 << (LOG_NMAX - 1 - $clog2(LANES))); k++) begin
                        dut.u_core.g_lane[gl].u_lane.u_fe.mem[k] = 128'd0;
                        dut.u_core.g_lane[gl].u_lane.u_fo.mem[k] = 128'd0;
                        dut.u_core.g_lane[gl].u_lane.u_ge.mem[k] = 128'd0;
                        dut.u_core.g_lane[gl].u_lane.u_go.mem[k] = 128'd0;
                    end
                    dut.u_core.g_lane[gl].u_lane.hf = 128'd0;
                    dut.u_core.g_lane[gl].u_lane.hg = 128'd0;
                end
            end
        end
    endgenerate
`endif

    task automatic axi_write(input [31:0] a, input [31:0] d);
        integer k;
        bit aw_done, w_done;
        aw_done = 0; w_done = 0;
        @(posedge clk);
        awaddr <= a; wdata <= d;
        if (wmode == 1) begin
            awvalid <= 1;
            repeat (wdly) @(posedge clk);
            wvalid <= 1;
        end else if (wmode == 2) begin
            wvalid <= 1;
            repeat (wdly) @(posedge clk);
            awvalid <= 1;
        end else begin
            awvalid <= 1; wvalid <= 1;
        end
        // independent AW / W handshakes (AXI4-Lite)
        for (k = 0; k < 1000 && !(aw_done && w_done); k++) begin
            @(posedge clk);
            if (awvalid && awready) begin aw_done = 1; awvalid <= 0; end
            if (wvalid && wready)   begin w_done = 1;  wvalid <= 0; end
        end
        if (!(aw_done && w_done)) begin nerr++; $display("ERR write handshake timeout at %08h", a); end
        for (k = 0; k < 1000 && !bvalid; k++) @(posedge clk);
        if (!bvalid) begin nerr++; $display("ERR no BVALID at %08h", a); end
        else if (bresp != 2'b00) $display("ERR bresp=%0d at %08h", bresp, a);
        @(posedge clk);
        repeat (gap) @(posedge clk);
    endtask

    task automatic axi_read(input [31:0] a, output [31:0] d);
        integer k;
        @(posedge clk);
        araddr <= a; arvalid <= 1;
        for (k = 0; k < 1000; k++) begin
            @(posedge clk);
            if (arvalid && arready) begin arvalid <= 0; break; end
        end
        for (k = 0; k < 1000 && !rvalid; k++) @(posedge clk);
        if (!rvalid) begin nerr++; $display("ERR no RVALID at %08h", a); end
        d = rdata;
        @(posedge clk);
        repeat (gap) @(posedge clk);
    endtask

    initial begin : main
        string trace, op;
        integer fd, code, nops, npoll, i;
        reg [31:0] a, x, v;
        if (!$value$plusargs("TRACE=%s", trace)) begin $display("ERR +TRACE missing"); $finish; end
        if (!$value$plusargs("WMODE=%d", wmode)) wmode = 0;
        if (!$value$plusargs("WDLY=%d", wdly)) wdly = 2;
        if (!$value$plusargs("GAP=%d", gap)) gap = 8;
        fd = $fopen(trace, "r");
        if (fd == 0) begin $display("ERR cannot open trace %s", trace); $finish; end
        // reset like sumfold_top: aresetn released after the MMCM lock counter
        repeat (20) @(posedge clk);
        aresetn <= 1'b1;
        repeat (20) @(posedge clk);
        nops = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s %h %h\n", op, a, x);
            if (code != 3) continue;
            nops++;
            if (op == "W") begin
                axi_write(a, x);
            end else if (op == "R") begin
                axi_read(a, v);
                $display("RD %08h %08h", a, v);
            end else if (op == "P") begin
                npoll = 0;
                do begin axi_read(a, v); npoll++; end while (!(v & x) && !v[2] && npoll < 100000);
                $display("RD %08h %08h", a, v);
            end else begin
                $display("ERR bad op %s", op); nerr++;
            end
        end
        $fclose(fd);
        $display("REPLAY_DONE ops=%0d errors=%0d time_ns=%0t", nops, nerr, $time);
        $finish;
    end
endmodule
