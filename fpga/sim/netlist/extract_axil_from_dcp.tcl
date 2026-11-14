# extract_axil_from_dcp.tcl : functional netlist of the whole SumFold AXI slave + core (cell u_sumfold, module sumfold_axil) from a routed board checkpoint, for fpga/sim/replay/replay.py --dut-netlist. usage: vivado -mode batch -source extract_axil_from_dcp.tcl -tclargs <post_route.dcp> <out.v>
set dcp [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
file mkdir [file dirname $out]
open_checkpoint $dcp
set c [get_cells u_sumfold]
puts "CELL=$c REF=[get_property REF_NAME $c]"
write_verilog -cell $c -mode funcsim -force -rename_top sumfold_axil $out
puts "EXTRACT_DONE=$out"
