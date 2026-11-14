# extract_mul_from_dcp.tcl : write the functional netlist of one gl_mul instance (and its DSP register settings) from a routed board checkpoint, to test the multiplier that is actually on the board. usage: vivado -mode batch -source extract_mul_from_dcp.tcl -tclargs <post_route.dcp> <outdir> [cell]
set dcp    [lindex $argv 0]
set outdir [file normalize [lindex $argv 1]]
set cell   [expr {[llength $argv] > 2 ? [lindex $argv 2] : "u_sumfold/u_core/g_lane[0].u_lane/u_C/u_m0"}]
file mkdir $outdir
open_checkpoint $dcp
set c [get_cells $cell]
puts "CELL=$c REF=[get_property REF_NAME $c] ORIG=[get_property ORIG_REF_NAME $c]"
set fp [open [file join $outdir dsp.rpt] w]
foreach d [lsort [get_cells -hier -filter "NAME =~ $cell/* && REF_NAME == DSP48E2"]] {
    puts $fp [format "%s AREG=%s BREG=%s MREG=%s PREG=%s OPMODE=%s" $d [get_property AREG $d] [get_property BREG $d] \
        [get_property MREG $d] [get_property PREG $d] [get_property OPMODEREG $d]]
}
close $fp
write_verilog -cell $c -mode funcsim -force [file join $outdir gl_mul_funcsim.v]
puts "EXTRACT_DONE"
