// ext_arith.v : quadratic extension Ext = F_p[X]/(X^2 - 7), element word {c1[63:0], c0[63:0]}. Semantics: src/edgecert/field.py (eadd, esub, emul Karatsuba form). II = 1.
`timescale 1ns / 1ps
`include "sumfold_defs.vh"

// ---------------------------------------------------------------- ext_add (latency EXT_ADD_LAT)
module ext_add (
    input  wire         clk,
    input  wire [127:0] a,
    input  wire [127:0] b,
    output wire [127:0] y
);
    gl_add u0 (.clk(clk), .a(a[63:0]),   .b(b[63:0]),   .y(y[63:0]));
    gl_add u1 (.clk(clk), .a(a[127:64]), .b(b[127:64]), .y(y[127:64]));
endmodule

// ---------------------------------------------------------------- ext_sub (latency EXT_SUB_LAT)
// y = a - b
module ext_sub (
    input  wire         clk,
    input  wire [127:0] a,
    input  wire [127:0] b,
    output wire [127:0] y
);
    gl_sub u0 (.clk(clk), .a(a[63:0]),   .b(b[63:0]),   .y(y[63:0]));
    gl_sub u1 (.clk(clk), .a(a[127:64]), .b(b[127:64]), .y(y[127:64]));
endmodule

// ---------------------------------------------------------------- ext_mul (latency EXT_MUL_LAT)
// Karatsuba: t0 = a0*b0, t1 = a1*b1, t2 = (a0+a1)(b0+b1) c0 = t0 + 7*t1 ,  c1 = (t2 - t0) - t1 7*t1 is computed multiplier-free as (t1 << 3) - t1 mod p (gl_mul7).
module ext_mul (
    input  wire         clk,
    input  wire [127:0] a,
    input  wire [127:0] b,
    output wire [127:0] y
);
    localparam LA  = `GL_ADD_LAT;
    localparam LS  = `GL_SUB_LAT;
    localparam L7  = `GL_MUL7_LAT;
    localparam C0L = `GL_MUL7_LAT + `GL_ADD_LAT;   // c0 post-multiply latency
    localparam C1L = 2 * `GL_SUB_LAT;              // c1 post-multiply latency

    wire [63:0] a0 = a[63:0], a1 = a[127:64];
    wire [63:0] b0 = b[63:0], b1 = b[127:64];

    // pre-add stage
    wire [63:0] sa, sb;
    gl_add u_sa (.clk(clk), .a(a0), .b(a1), .y(sa));
    gl_add u_sb (.clk(clk), .a(b0), .b(b1), .y(sb));
    wire [255:0] ab_d;
    pipe_delay #(.W(256), .D(LA)) u_dab (.clk(clk), .d({a1, a0, b1, b0}), .q(ab_d));

    // three base products
    wire [63:0] t0, t1, t2;
    gl_mul u_m0 (.clk(clk), .a(ab_d[191:128]), .b(ab_d[63:0]),   .y(t0));  // a0*b0
    gl_mul u_m1 (.clk(clk), .a(ab_d[255:192]), .b(ab_d[127:64]), .y(t1));  // a1*b1
    gl_mul u_m2 (.clk(clk), .a(sa),            .b(sb),           .y(t2));

    // c0 = t0 + 7*t1
    wire [63:0] t1x7, t0_d7, c0;
    gl_mul7 u_m7 (.clk(clk), .a(t1), .y(t1x7));
    pipe_delay #(.W(64), .D(L7)) u_dt0 (.clk(clk), .d(t0), .q(t0_d7));
    gl_add u_c0 (.clk(clk), .a(t0_d7), .b(t1x7), .y(c0));

    // c1 = (t2 - t0) - t1
    wire [63:0] u1, t1_ds, c1_raw, c1;
    gl_sub u_s1 (.clk(clk), .a(t2), .b(t0), .y(u1));
    pipe_delay #(.W(64), .D(LS)) u_dt1 (.clk(clk), .d(t1), .q(t1_ds));
    gl_sub u_s2 (.clk(clk), .a(u1), .b(t1_ds), .y(c1_raw));
    pipe_delay #(.W(64), .D(C0L - C1L)) u_dc1 (.clk(clk), .d(c1_raw), .q(c1));

    assign y = {c1, c0};
endmodule
