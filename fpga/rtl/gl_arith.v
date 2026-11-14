// gl_arith.v : Goldilocks (p = 2^64 - 2^32 + 1) pipelined base-field units. Semantics: src/edgecert/field.py (fadd, fsub, fmul/reduce128, mul-by-W with W = 7). Inputs must be canonical (< p); outputs are canonical. All units have II = 1.
`timescale 1ns / 1ps
`include "sumfold_defs.vh"

// ---------------------------------------------------------------- pipe_delay
// Plain delay line (maps to SRL or FFs). D = 0 is a wire.
module pipe_delay #(
    parameter W = 1,
    parameter D = 1
) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (D == 0) begin : g_wire
            assign q = d;
        end else begin : g_reg
            reg [W-1:0] r [0:D-1];
            integer i;
            always @(posedge clk) begin
                r[0] <= d;
                for (i = 1; i < D; i = i + 1)
                    r[i] <= r[i-1];
            end
            assign q = r[D-1];
        end
    endgenerate
endmodule

// ---------------------------------------------------------------- gl_add (latency 2)
// s = a + b (65 bit); s >= p  <=>  s + (2^32 - 1) >= 2^64, and then s - p = (s + 2^32 - 1) mod 2^64.
module gl_add (
    input  wire        clk,
    input  wire [63:0] a,
    input  wire [63:0] b,
    output reg  [63:0] y
);
    reg  [64:0] s;
    wire [64:0] w = s + 65'h0_0000_0000_FFFF_FFFF;
    always @(posedge clk) begin
        s <= {1'b0, a} + {1'b0, b};
        y <= w[64] ? w[63:0] : s[63:0];
    end
endmodule

// ---------------------------------------------------------------- gl_sub (latency 2)
// d = a - b (65 bit two's complement). On borrow, a - b + p = (d mod 2^64) - (2^32 - 1).
module gl_sub (
    input  wire        clk,
    input  wire [63:0] a,
    input  wire [63:0] b,
    output reg  [63:0] y
);
    reg [64:0] d;
    always @(posedge clk) begin
        d <= {1'b0, a} - {1'b0, b};
        y <= d[64] ? (d[63:0] - 64'h0000_0000_FFFF_FFFF) : d[63:0];
    end
endmodule

// ---------------------------------------------------------------- gl_mul7 (latency 3)
// 7*a = (a << 3) - a (67 bit), then fold the top 3 bits with 2^64 = 2^32 - 1 (mod p). y2 < 2^64 + 6*2^32 < 2p, so one conditional subtraction gives the canonical value.
module gl_mul7 (
    input  wire        clk,
    input  wire [63:0] a,
    output reg  [63:0] y
);
    reg  [66:0] x1;
    reg  [64:0] y2;
    wire [2:0]  h  = x1[66:64];
    wire [63:0] hm = {29'd0, h, 32'd0} - {61'd0, h};          // h * (2^32 - 1)
    wire [64:0] w  = y2 + 65'h0_0000_0000_FFFF_FFFF;
    always @(posedge clk) begin
        x1 <= {a, 3'b000} - {3'b000, a};
        y2 <= {1'b0, x1[63:0]} + {1'b0, hm};
        y  <= w[64] ? w[63:0] : y2[63:0];
    end
endmodule

// ---------------------------------------------------------------- gl_reduce128 (latency 4)
// Multiplier-free reduction, identical to field.reduce128: x = hi*2^64 + lo, hi = hh*2^32 + hl ;  x = lo - hh + hl*(2^32 - 1) (mod p) Stage 1: d = lo - hh, m = hl*(2^32-1) ; stage 2: t = d (+p on borrow) ; stage 3: u = t + m (u <= 2p - 2) ; stage 4: y = u >= p ? u - p : u.
module gl_reduce128 (
    input  wire         clk,
    input  wire [127:0] x,
    output reg  [63:0]  y
);
    reg  [64:0] d1;
    reg  [63:0] m1, t2, m2;
    reg  [64:0] u3;
    wire [64:0] w = u3 + 65'h0_0000_0000_FFFF_FFFF;
    always @(posedge clk) begin
        d1 <= {1'b0, x[63:0]} - {33'd0, x[127:96]};
        m1 <= {x[95:64], 32'd0} - {32'd0, x[95:64]};
        t2 <= d1[64] ? (d1[63:0] - 64'h0000_0000_FFFF_FFFF) : d1[63:0];
        m2 <= m1;
        u3 <= {1'b0, t2} + {1'b0, m2};
        y  <= w[64] ? w[63:0] : u3[63:0];
    end
endmodule

// ---------------------------------------------------------------- gl_mul (latency GL_MUL_LAT)
// 64x64 -> 128 product with registered inputs and a GL_MUL_PIPE-cycle product pipeline, followed by gl_reduce128. The default product is an explicit 12-DSP48E2 decomposition (see below); the behavioural "a*b + NP registers" form is kept only as GL_MUL_LEGACY_INFER because Vivado 2022.2 mis-pipelines it (board bring-up bug of 2026-09-27, docs/notes/fpga_board.md).
module gl_mul (
    input  wire        clk,
    input  wire [63:0] a,
    input  wire [63:0] b,
    output wire [63:0] y
);
    localparam NP = `GL_MUL_PIPE;
    reg  [63:0] a_r, b_r;
    always @(posedge clk) begin
        a_r <= a;
        b_r <= b;
    end
    genvar k;
`ifdef GL_MUL_SPLIT3
        // Variant (NOT netlist-verified, same automatic wide-multiplier pipelining as the LEGACY form): three 26-bit column products a_i * b (each a 4-DSP PCIN>>17 cascade, 12 DSP48E2 in total), NP-2 pipeline registers per column, then two registered fabric additions.
    wire [89:0] c0 = a_r[25:0]  * b_r;
    wire [89:0] c1 = a_r[51:26] * b_r;
    wire [75:0] c2 = a_r[63:52] * b_r;
    wire [255:0] cst [0:NP-2];
    assign cst[0] = {c2, c1, c0};
    generate
        for (k = 0; k < NP - 2; k = k + 1) begin : g_mp
            (* shreg_extract = "no" *) reg [255:0] q;
            always @(posedge clk) q <= cst[k];
            assign cst[k+1] = q;
        end
    endgenerate
    wire [89:0] c0p = cst[NP-2][89:0];
    wire [89:0] c1p = cst[NP-2][179:90];
    wire [75:0] c2p = cst[NP-2][255:180];
    (* shreg_extract = "no" *) reg [115:0] s01;
    (* shreg_extract = "no" *) reg [75:0]  c2q;
    (* shreg_extract = "no" *) reg [127:0] prod_q;
    always @(posedge clk) begin
        s01    <= {26'd0, c0p} + {c1p, 26'd0};
        c2q    <= c2p;
        prod_q <= {12'd0, s01} + {c2q, 52'd0};
    end
    gl_reduce128 u_red (.clk(clk), .x(prod_q), .y(y));
`elsif GL_MUL_LEGACY_INFER
        // LEGACY (board bring-up bug, kept only to reproduce it; never use in a hardware build): one behavioural 64x64 product followed by NP registers. Vivado 2022.2 splits it into a 16-DSP48E2 cascade and redistributes the NP registers itself. The resulting netlist is NOT cycle-equivalent to this RTL: in a stream, the partial product a[50:34]*b[63:51] (weight 2^85) is taken from the NEXT operand pair and a[63:51]*b[16:0] (weight 2^51) from the PREVIOUS one (impulse test fpga/sim/netlist/tb_mul_impulse.sv). RTL simulation cannot see this; the board run of 2026-09-27 and the netlist replay (docs/notes/fpga_board.md) do.
    wire [127:0] prod = a_r * b_r;
    wire [127:0] stage [0:NP];
    assign stage[0] = prod;
    generate
        for (k = 0; k < NP; k = k + 1) begin : g_mp
            (* shreg_extract = "no" *) reg [127:0] q;
            always @(posedge clk) q <= stage[k];
            assign stage[k+1] = q;
        end
    endgenerate
    gl_reduce128 u_red (.clk(clk), .x(stage[NP]), .y(y));
`else
        // Default (fix of 2026-09-27): explicit DSP48E2 decomposition. Every multiplier is at most 26 x 17 bits (one DSP48E2 each, unsigned inside the signed 27 x 18 array) and every pipeline register is written here, so synthesis only has to absorb registers into DSP slices; nothing relies on the tool's automatic pipelining of a wide multiplier (see the LEGACY branch). a = a0 + a1*2^26 + a2*2^52 (26, 26, 12 bits);  b = b0 + b1*2^17 + b2*2^34 + b3*2^51 (17, 17, 17, 13)
    //   column i: c_i = a_i * b by a 4-stage cascade  P_j = (P_{j-1} >> 17) + a_i*b_j  (PCIN>>17);
        // c_i = {P_3, P_2[16:0], P_1[16:0], P_0[16:0]}  (a_i * b < 2^90, P_3 = c_i >> 51 < 2^39) prod = c_0 + c_1*2^26 + c_2*2^52 Schedule (cycle 0 = a_r/b_r): M_i0 @1, P_i0 @2 | M_i1 @2, P_i1 @3 | M_i2, M_i3 @3, stage 2 is combinational into stage 3, P_i3 @4 | s01 = c0 + c1*2^26 @5 | prod @6 = NP. Latency is unchanged (GL_MUL_LAT = 1 + 6 + GL_RED_LAT); 12 DSP48E2 per gl_mul instead of 16.
    generate
        if (NP != 6) begin : g_bad_np
            GL_MUL_PIPE_must_be_6_for_the_explicit_multiplier u_err ();
        end
    endgenerate
    reg [63:0] a_d1, a_d2, b_d1, b_d2;
    always @(posedge clk) begin
        a_d1 <= a_r;
        a_d2 <= a_d1;
        b_d1 <= b_r;
        b_d2 <= b_d1;
    end
    wire [89:0] col [0:2];
    generate
        for (k = 0; k < 3; k = k + 1) begin : g_col
            localparam integer LO  = 26 * k;
            localparam integer AWK = (k == 2) ? 12 : 26;
            wire [AWK-1:0] x0 = a_r[LO +: AWK];
            wire [AWK-1:0] x1 = a_d1[LO +: AWK];
            wire [AWK-1:0] x2 = a_d2[LO +: AWK];
            reg  [47:0] m0, m1, m2, m3, p0, p1, p3;
            reg  [16:0] p0_lo_d, p0_lo_dd, p1_lo_d, s2_lo;
            wire [47:0] s2 = (p1 >> 17) + m2;
            always @(posedge clk) begin
                m0       <= x0 * b_r[16:0];        // @1
                m1       <= x1 * b_d1[33:17];      // @2
                m2       <= x2 * b_d2[50:34];      // @3
                m3       <= x2 * b_d2[63:51];      // @3
                p0       <= m0;                    // @2
                p1       <= (p0 >> 17) + m1;       // @3
                p3       <= (s2 >> 17) + m3;       // @4
                s2_lo    <= s2[16:0];              // @4
                p1_lo_d  <= p1[16:0];              // @4
                p0_lo_d  <= p0[16:0];              // @3
                p0_lo_dd <= p0_lo_d;               // @4
            end
            assign col[k] = {p3[38:0], s2_lo, p1_lo_d, p0_lo_dd};
        end
    endgenerate
    reg [115:0] s01;
    reg [75:0]  c2q;
    reg [127:0] prod_q;
    always @(posedge clk) begin
        s01    <= {26'd0, col[0]} + {col[1], 26'd0};     // @5
        c2q    <= col[2][75:0];
        prod_q <= {12'd0, s01} + {c2q, 52'd0};          // @6
    end
    gl_reduce128 u_red (.clk(clk), .x(prod_q), .y(y));
`endif
endmodule
