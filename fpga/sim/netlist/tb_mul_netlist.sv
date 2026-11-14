// tb_mul_netlist.sv : streams the golden gl_mul vectors (fpga/sim/golden/field_mul.hex, {a, b, y}) through a gl_mul netlist (write_verilog -mode funcsim) at II = 1 and checks every output LAT cycles later. +GAP=1 inserts one cycle with random operands between vectors.
`timescale 1ns / 1ps
module tb_mul_netlist;
    parameter LAT = 11;
    localparam NV = 512;
    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg  [191:0] v [0:NV-1];
    reg  [63:0]  a = 0, b = 0;
    wire [63:0]  y;
    gl_mul dut (.clk(clk), .a(a), .b(b), .y(y));
    reg  [63:0]  exp_q [0:4095];
    reg          vld_q [0:4095];
    integer t, errs = 0, chk = 0, gap = 0, issued = 0, dump = 0;
    string gold;
    initial begin
        if (!$value$plusargs("GOLD=%s", gold)) gold = "../golden";
        if (!$value$plusargs("GAP=%d", gap)) gap = 0;
        if (!$value$plusargs("DUMP=%d", dump)) dump = 0;
        $readmemh($sformatf("%s/field_mul.hex", gold), v);
        if ($isunknown(v[0]) || $isunknown(v[NV-1])) begin $display("MULNET_RESULT FAIL (vectors)"); $finish; end
        for (t = 0; t < 4096; t++) vld_q[t] = 0;
        // let the global set/reset pulse of glbl finish
        #200;
        for (t = 0; t < 2 * NV * (gap + 1) + 64; t++) begin
            @(negedge clk);
            if (dump) $display("Y %0d %016h", t, y);
            if (t >= LAT && vld_q[t - LAT]) begin
                chk++;
                if (y !== exp_q[t - LAT]) begin
                    errs++;
                    if (errs <= 8) $display("ERR t=%0d got %016h exp %016h diff %016h", t - LAT, y, exp_q[t - LAT], y ^ exp_q[t - LAT]);
                end
            end
            if (issued < NV && (gap == 0 || (t % (gap + 1)) == 0)) begin
                {a, b} = v[issued][191:64];
                exp_q[t] = v[issued][63:0];
                vld_q[t] = 1;
                issued++;
            end else begin
                a = {$urandom, $urandom} % 64'hFFFFFFFF00000001;
                b = {$urandom, $urandom} % 64'hFFFFFFFF00000001;
                vld_q[t] = 0;
            end
        end
        $display("MULNET_RESULT %s gap=%0d checked=%0d errors=%0d", (errs == 0 && chk == NV) ? "PASS" : "FAIL", gap, chk, errs);
        $finish;
    end
endmodule
