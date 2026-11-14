// sumfold_defs.vh : shared latency constants for the SumFold RTL (spec/sumfold_kernel.md). All arithmetic units are fully pipelined (II = 1). Every latency below is exact and is checked cycle-by-cycle by fpga/tb/tb_field.sv.
`ifndef SUMFOLD_DEFS_VH
`define SUMFOLD_DEFS_VH

`define GL_P              64'hFFFFFFFF00000001

// Goldilocks base-field units
`define GL_ADD_LAT        2
`define GL_SUB_LAT        2
`define GL_MUL7_LAT       3
`define GL_RED_LAT        4
// Product pipeline depth after the registered 64x64 inputs. The default gl_mul places all six stages explicitly in its 12-DSP48E2 decomposition (gl_arith.v); it requires GL_MUL_PIPE = 6.
`ifndef GL_MUL_PIPE
`define GL_MUL_PIPE       6
`endif
// input register + product pipeline + reduction
`define GL_MUL_LAT        (1 + `GL_MUL_PIPE + `GL_RED_LAT)

// Quadratic extension F_p[X]/(X^2-7), Karatsuba (3 gl_mul + mul-by-7)
`define EXT_ADD_LAT       `GL_ADD_LAT
`define EXT_SUB_LAT       `GL_SUB_LAT
`define EXT_MUL_LAT       (`GL_ADD_LAT + `GL_MUL_LAT + `GL_MUL7_LAT + `GL_ADD_LAT)

// Lane memory read latency: BRAM array register + BRAM output register + one fabric register
`define RAM_RD_LAT        3

// Lane pipeline (cycles counted from the core issue register, see sumfold_lane.v header):
//   lane input register              1
//   memory read                      RAM_RD_LAT                     -> S0
//   fold: ext_sub + in-reg + ext_mul + ext_add  (2 + 1 + LE + 2)   -> S1
//   eval (s2 path): in-reg + pre-op(4) + ext_mul + product reg + accumulate
`define SF_S0_LAT         (1 + `RAM_RD_LAT)
`define SF_S1_LAT         (`SF_S0_LAT + `EXT_SUB_LAT + 1 + `EXT_MUL_LAT + `EXT_ADD_LAT)
`define SF_EVAL_LAT       (`SF_S1_LAT + 1 + `EXT_ADD_LAT + `EXT_SUB_LAT + `EXT_MUL_LAT + 2)
`define SF_FOLD_LAT       (`SF_S1_LAT + 1)

`endif
