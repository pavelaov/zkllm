// tb_mul_impulse.sv : impulse response of a gl_mul netlist (or RTL). For each bit pair (p, q) one cycle with a = 2^p, b = 2^q (zeros before and after). A cycle-equivalent implementation shows a single non-zero output y = 2^(p+q) mod p exactly LAT cycles after the impulse; a mis-pipelined partial product shows up as a non-zero output at another cycle (or as a wrong value). Also prints the raw product register (dut.u_red.Q, legacy netlist naming) when it exists.
`timescale 1ns / 1ps
module tb_mul_impulse;
    parameter LAT = 11;
    localparam [127:0] PM = 128'h0000_0000_0000_0000_FFFF_FFFF_0000_0001;
    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg  [63:0] a = 0, b = 0;
    wire [63:0] y;
    gl_mul dut (.clk(clk), .a(a), .b(b), .y(y));
    integer pi, qi, t, p, q, nz, bad = 0, pairs = 0;
    integer bits [0:7];
    reg [127:0] e;
    initial begin
        bits[0] = 0; bits[1] = 8; bits[2] = 17; bits[3] = 25; bits[4] = 34; bits[5] = 42; bits[6] = 51; bits[7] = 63;
        #200;
        for (pi = 0; pi < 8; pi++) for (qi = 0; qi < 8; qi++) begin
            p = bits[pi]; q = bits[qi];
            e = (128'd1 << (p + q)) % PM;
            @(negedge clk); a = 64'd1 << p; b = 64'd1 << q;
            @(negedge clk); a = 0; b = 0;
            nz = 0;
            for (t = 1; t < LAT + 12; t++) begin
                if (y !== 64'd0) begin
                    nz++;
                    $display("IMP p=%0d q=%0d t=%0d y=%016h exp=%016h %s", p, q, t, y, e[63:0],
                             (t == LAT && y === e[63:0]) ? "OK" : "WRONG");
                    if (!(t == LAT && y === e[63:0])) bad++;
                end
                @(negedge clk);
            end
            if (nz != 1) begin bad++; if (nz == 0) $display("IMP p=%0d q=%0d no output", p, q); end
            pairs++;
        end
        $display("MULIMP_RESULT %s pairs=%0d bad=%0d", bad == 0 ? "PASS" : "FAIL", pairs, bad);
        $finish;
    end
endmodule
