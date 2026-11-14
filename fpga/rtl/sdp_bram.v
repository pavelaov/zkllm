// sdp_bram.v : simple dual-port block RAM, one write port, one read port, read latency 2 (array register + output register, mapped to the BRAM36 DOB_REG). CASCADE_HEIGHT limits the BRAM read cascade: with the default (unlimited) Vivado chains up to 8 RAMB36 for a 32K-deep bank (LANES = 1), a 3.1 ns path; deeper banks then use a LUT mux after the output registers instead.
`timescale 1ns / 1ps
module sdp_bram #(
    parameter DW = 128,
    parameter AW = 12,
    parameter CASCADE = 2
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [DW-1:0] wdata,
    input  wire [AW-1:0] raddr,
    output reg  [DW-1:0] rdata
);
    (* ram_style = "block", cascade_height = CASCADE *) reg [DW-1:0] mem [0:(1<<AW)-1];
    reg [DW-1:0] rd1;
    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        rd1   <= mem[raddr];
        rdata <= rd1;
    end
endmodule
