// tb_field.sv : streams one vector per cycle (II = 1) through every arithmetic unit and checks each output exactly LAT cycles later against the golden model (fpga/host/make_golden.py).
`timescale 1ns / 1ps
`include "sumfold_defs.vh"

module tb_field;
    localparam NV = 512;
    reg clk = 1'b0;
    always #2 clk = ~clk;

    reg [191:0] v_add  [0:NV-1];
    reg [191:0] v_sub  [0:NV-1];
    reg [191:0] v_mul  [0:NV-1];
    reg [127:0] v_m7   [0:NV-1];
    reg [191:0] v_red  [0:NV-1];
    reg [383:0] v_emul [0:NV-1];

    reg  [63:0]  add_a, add_b, sub_a, sub_b, mul_a, mul_b, m7_a;
    reg  [127:0] red_x, em_a, em_b;
    wire [63:0]  add_y, sub_y, mul_y, m7_y, red_y;
    wire [127:0] em_y;

    gl_add       u_add (.clk(clk), .a(add_a), .b(add_b), .y(add_y));
    gl_sub       u_sub (.clk(clk), .a(sub_a), .b(sub_b), .y(sub_y));
    gl_mul       u_mul (.clk(clk), .a(mul_a), .b(mul_b), .y(mul_y));
    gl_mul7      u_m7  (.clk(clk), .a(m7_a), .y(m7_y));
    gl_reduce128 u_red (.clk(clk), .x(red_x), .y(red_y));
    ext_mul      u_em  (.clk(clk), .a(em_a), .b(em_b), .y(em_y));

    integer t, k;
    integer err_add = 0, err_sub = 0, err_mul = 0, err_m7 = 0, err_red = 0, err_em = 0;
    integer chk_add = 0, chk_sub = 0, chk_mul = 0, chk_m7 = 0, chk_red = 0, chk_em = 0;
    string gold;

    initial begin
        if (!$value$plusargs("GOLD=%s", gold)) gold = "../golden";
        $readmemh($sformatf("%s/field_add.hex", gold), v_add);
        $readmemh($sformatf("%s/field_sub.hex", gold), v_sub);
        $readmemh($sformatf("%s/field_mul.hex", gold), v_mul);
        $readmemh($sformatf("%s/field_mul7.hex", gold), v_m7);
        $readmemh($sformatf("%s/field_red.hex", gold), v_red);
        $readmemh($sformatf("%s/field_emul.hex", gold), v_emul);
        for (t = 0; t < NV; t = t + 1)
            if ($isunknown(v_add[t]) || $isunknown(v_sub[t]) || $isunknown(v_mul[t]) ||
                $isunknown(v_m7[t]) || $isunknown(v_red[t]) || $isunknown(v_emul[t])) begin
                $display("ERR golden vector %0d not loaded", t);
                $display("FIELD_RESULT FAIL");
                $finish;
            end
        for (t = 0; t < NV + 64; t = t + 1) begin
            @(negedge clk);
            // ---- check outputs of the vector issued LAT cycles ago
            k = t - `GL_ADD_LAT;
            if (k >= 0 && k < NV) begin chk_add++; if (add_y !== v_add[k][63:0]) begin err_add++; if (err_add < 4) $display("ERR add k=%0d got %016h exp %016h", k, add_y, v_add[k][63:0]); end end
            k = t - `GL_SUB_LAT;
            if (k >= 0 && k < NV) begin chk_sub++; if (sub_y !== v_sub[k][63:0]) begin err_sub++; if (err_sub < 4) $display("ERR sub k=%0d got %016h exp %016h", k, sub_y, v_sub[k][63:0]); end end
            k = t - `GL_MUL_LAT;
            if (k >= 0 && k < NV) begin chk_mul++; if (mul_y !== v_mul[k][63:0]) begin err_mul++; if (err_mul < 4) $display("ERR mul k=%0d got %016h exp %016h", k, mul_y, v_mul[k][63:0]); end end
            k = t - `GL_MUL7_LAT;
            if (k >= 0 && k < NV) begin chk_m7++; if (m7_y !== v_m7[k][63:0]) begin err_m7++; if (err_m7 < 4) $display("ERR mul7 k=%0d got %016h exp %016h", k, m7_y, v_m7[k][63:0]); end end
            k = t - `GL_RED_LAT;
            if (k >= 0 && k < NV) begin chk_red++; if (red_y !== v_red[k][63:0]) begin err_red++; if (err_red < 4) $display("ERR red k=%0d got %016h exp %016h", k, red_y, v_red[k][63:0]); end end
            k = t - `EXT_MUL_LAT;
            if (k >= 0 && k < NV) begin chk_em++; if (em_y !== v_emul[k][127:0]) begin err_em++; if (err_em < 4) $display("ERR emul k=%0d got %032h exp %032h", k, em_y, v_emul[k][127:0]); end end
            // ---- drive vector t
            if (t < NV) begin
                {add_a, add_b} = v_add[t][191:64];
                {sub_a, sub_b} = v_sub[t][191:64];
                {mul_a, mul_b} = v_mul[t][191:64];
                m7_a           = v_m7[t][127:64];
                red_x          = v_red[t][191:64];
                {em_a, em_b}   = v_emul[t][383:128];
            end
        end
        $display("UNIT gl_add       lat=%0d checked=%0d errors=%0d", `GL_ADD_LAT,  chk_add, err_add);
        $display("UNIT gl_sub       lat=%0d checked=%0d errors=%0d", `GL_SUB_LAT,  chk_sub, err_sub);
        $display("UNIT gl_mul       lat=%0d checked=%0d errors=%0d", `GL_MUL_LAT,  chk_mul, err_mul);
        $display("UNIT gl_mul7      lat=%0d checked=%0d errors=%0d", `GL_MUL7_LAT, chk_m7,  err_m7);
        $display("UNIT gl_reduce128 lat=%0d checked=%0d errors=%0d", `GL_RED_LAT,  chk_red, err_red);
        $display("UNIT ext_mul      lat=%0d checked=%0d errors=%0d", `EXT_MUL_LAT, chk_em,  err_em);
        if (err_add + err_sub + err_mul + err_m7 + err_red + err_em == 0 &&
            chk_add == NV && chk_mul == NV && chk_em == NV && chk_red == NV)
            $display("FIELD_RESULT PASS");
        else
            $display("FIELD_RESULT FAIL");
        $finish;
    end
endmodule
