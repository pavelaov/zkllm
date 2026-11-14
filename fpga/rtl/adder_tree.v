// adder_tree.v : pipelined binary integer adder tree (N a power of two, latency LG).
`timescale 1ns / 1ps
module adder_tree #(
    parameter N  = 8,
    parameter LG = 3,
    parameter W  = 80
) (
    input  wire              clk,
    input  wire [N*W-1:0]    din,
    output wire [W+LG-1:0]   dout
);
    localparam OW = W + LG;
    // heap layout: node 1 is the root, leaves are nodes N .. 2N-1
    wire [OW-1:0] val [1:2*N-1];
    genvar k;
    generate
        for (k = N; k < 2 * N; k = k + 1) begin : g_leaf
            assign val[k] = din[(k-N)*W +: W];      // zero-extended to OW bits
        end
        for (k = 1; k < N; k = k + 1) begin : g_node
            reg [OW-1:0] s;
            always @(posedge clk) s <= val[2*k] + val[2*k+1];
            assign val[k] = s;
        end
    endgenerate
    assign dout = val[1];
endmodule
