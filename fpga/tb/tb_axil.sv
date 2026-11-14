// tb_axil.sv : smoke test of the AXI4-Lite register map (sumfold_axil) with a simple BFM. Runs one GEN + AUTO job (case given by +CASE, default g_L2_n6) and one MEM write/read, and checks ID, CFG, STATUS, all round messages, lane finals and the cycle registers.
`timescale 1ns / 1ps

module tb_axil;
    parameter LANES    = 2;
    parameter LOG_NMAX = 16;
    parameter KCLK_MHZ = 300;
    localparam [31:0] CFG_EXP = (KCLK_MHZ << 16) | (LOG_NMAX << 8) | LANES;

    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg aresetn = 1'b0;

    reg  [31:0] awaddr = 0, wdata = 0, araddr = 0;
    reg         awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;
    wire        o_busy, o_done, o_error;

    sumfold_axil #(.LANES(LANES), .LOG_NMAX(LOG_NMAX), .KCLK_MHZ(KCLK_MHZ)) dut (
        .aclk(clk), .aresetn(aresetn),
        .s_axi_awaddr(awaddr), .s_axi_awprot(3'b000), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arprot(3'b000), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .o_busy(o_busy), .o_done(o_done), .o_error(o_error));

    // ------------------------------------------------------------ AXI4-Lite BFM
    task automatic axi_write(input [31:0] a, input [31:0] d);
        @(negedge clk); awaddr = a; wdata = d; awvalid = 1; wvalid = 1;
        forever begin @(negedge clk); if (awready && wready) break; end
        @(negedge clk); awvalid = 0; wvalid = 0; bready = 1;
        forever begin if (bvalid) break; @(negedge clk); end
        if (bresp != 2'b00) $display("ERR bresp=%0d at %08h", bresp, a);
        @(negedge clk); bready = 0;
    endtask
    task automatic axi_read(input [31:0] a, output [31:0] d);
        @(negedge clk); araddr = a; arvalid = 1;
        forever begin @(negedge clk); if (arready) break; end
        @(negedge clk); arvalid = 0; rready = 1;
        forever begin if (rvalid) break; @(negedge clk); end
        d = rdata;
        @(negedge clk); rready = 0;
    endtask
    task automatic write128(input [31:0] base, input [127:0] v);
        axi_write(base + 0,  v[31:0]);
        axi_write(base + 4,  v[63:32]);
        axi_write(base + 8,  v[95:64]);
        axi_write(base + 12, v[127:96]);
    endtask
    task automatic read128(input [31:0] base, output [127:0] v);
        reg [31:0] w0, w1, w2, w3;
        axi_read(base + 0,  w0);
        axi_read(base + 4,  w1);
        axi_read(base + 8,  w2);
        axi_read(base + 12, w3);
        v = {w3, w2, w1, w0};
    endtask

    reg [127:0] g_ch [0:15];
    reg [127:0] g_msg[0:47];
    reg [127:0] g_fin[0:15];
    reg [127:0] g_f  [0:1023];

    initial begin : main
        string gold, name, cases, fn;
        integer fd, code, n, mem, step, R, i, k, t, errors;
        reg [63:0] sf, sg;
        reg [31:0] v;
        reg [127:0] d;
        errors = 0;
        if (!$value$plusargs("GOLD=%s", gold)) gold = "../golden";
        if (!$value$plusargs("CASE=%s", name)) name = "g_L2_n6";
        cases = $sformatf("%s/cases_L%0d.txt", gold, LANES);
        fd = $fopen(cases, "r");
        R = -1;
        while (!$feof(fd) && R < 0) begin
            string nm;
            code = $fscanf(fd, "%s %d %d %d %h %h %d\n", nm, n, mem, step, sf, sg, k);
            if (code == 7 && nm == name) R = k;
        end
        $fclose(fd);
        if (R < 0) begin $display("ERR case %s not found", name); $finish; end
        fn = $sformatf("%s/%s_ch.hex", gold, name);  $readmemh(fn, g_ch);
        fn = $sformatf("%s/%s_msg.hex", gold, name); $readmemh(fn, g_msg);
        fn = $sformatf("%s/%s_fin.hex", gold, name); $readmemh(fn, g_fin);
        fn = $sformatf("%s/%s_f.hex", gold, name);   $readmemh(fn, g_f);
        for (i = 0; i < 3 * R; i++) if ($isunknown(g_msg[i])) begin errors++; $display("ERR golden msg %0d not loaded", i); end
        for (i = 0; i < 2 * LANES; i++) if ($isunknown(g_fin[i])) begin errors++; $display("ERR golden final %0d not loaded", i); end

        repeat (10) @(negedge clk);
        aresetn = 1;
        repeat (60) @(negedge clk);

        axi_read(32'h000, v);
        if (v !== 32'hEDCE2026) begin errors++; $display("ERR ID=%08h", v); end
        axi_read(32'h004, v);
        if (v !== CFG_EXP) begin errors++; $display("ERR CFG=%08h", v); end
        $display("AXI ID=%08h", 32'hEDCE2026);

        axi_write(32'h010, n);
        axi_write(32'h018, sf[31:0]); axi_write(32'h01C, sf[63:32]);
        axi_write(32'h020, sg[31:0]); axi_write(32'h024, sg[63:32]);
        axi_write(32'h008, 32'h2);                          // GEN
        t = 0;
        do begin axi_read(32'h00C, v); t++; end while (!v[4] && t < 1000);
        if (!v[4]) begin errors++; $display("ERR GEN_DONE not set, STATUS=%08h", v); end
        axi_read(32'h088, v);
        $display("AXI CYC_GEN=%0d", v);

        // MEM read-back of f[1] (generator placement through the register interface)
        axi_write(32'h0A0, 1); axi_write(32'h0A4, 0); axi_write(32'h0B8, 2);
        repeat (20) @(negedge clk);
        read128(32'h0C0, d);
        if (d !== g_f[1]) begin errors++; $display("ERR MEM read f[1]=%032h exp %032h", d, g_f[1]); end

        for (i = 0; i < R; i++) begin
            axi_write(32'h040, i);
            write128(32'h044, g_ch[i]);
            axi_write(32'h054, 1);
        end
        axi_write(32'h008, 32'h1);                          // START, AUTO
        t = 0;
        do begin axi_read(32'h00C, v); t++; end while (!v[1] && t < 5000);
        if (!v[1] || v[2]) begin errors++; $display("ERR job STATUS=%08h", v); end
        axi_read(32'h014, v);
        if (v !== R) begin errors++; $display("ERR ROUND=%0d", v); end

        for (i = 0; i < 3 * R; i++) begin
            axi_write(32'h060, i);
            read128(32'h064, d);
            if (d !== g_msg[i]) begin errors++; $display("ERR msg word %0d got %032h exp %032h", i, d, g_msg[i]); end
        end
        for (i = 0; i < 2 * LANES; i++) begin
            axi_write(32'h060, 48 + i);
            read128(32'h064, d);
            if (d !== g_fin[i]) begin errors++; $display("ERR final %0d got %032h exp %032h", i, d, g_fin[i]); end
        end
        for (i = 0; i <= R; i++) begin
            axi_write(32'h090, i);
            axi_read(32'h094, v);
            $display("AXI CYC_ROUND[%0d]=%0d", i, v);
        end
        axi_read(32'h080, v);
        $display("AXI CYC_TOTAL=%0d", v);
        $display("AXI_RESULT case=%s L=%0d result=%s errors=%0d", name, LANES, errors == 0 ? "PASS" : "FAIL", errors);
        $finish;
    end
endmodule
