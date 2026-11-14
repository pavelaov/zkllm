# synth_unit.tcl : synthesize one SumFold unit out of context with the board-build synthesis settings (same part, same RTL, default synth_design options unless extra options are given) and write a functional simulation netlist (write_verilog -mode funcsim). usage: vivado -mode batch -nojournal -log <log> -source synth_unit.tcl -tclargs <top> <outdir> [synth options ...]
set top    [lindex $argv 0]
set outdir [file normalize [lindex $argv 1]]
set extra  [lrange $argv 2 end]
set here [file dirname [file normalize [info script]]]
set rtl  [file normalize [file join $here .. .. rtl]]
file mkdir $outdir
set_param general.maxThreads 8
read_verilog [list $rtl/gl_arith.v $rtl/ext_arith.v $rtl/sdp_bram.v $rtl/adder_tree.v \
                   $rtl/sumfold_lane.v $rtl/sumfold_core.v $rtl/sumfold_axil.v]
synth_design -top $top -part xczu9eg-ffvb1156-2-e -mode out_of_context -include_dirs $rtl {*}$extra
create_clock -period 4.000 -name kclk [get_ports clk]
write_verilog -mode funcsim -force [file join $outdir ${top}_funcsim.v]
set fp [open [file join $outdir dsp.rpt] w]
foreach c [lsort [get_cells -hier -filter {REF_NAME == DSP48E2}]] {
    puts $fp [format "%s AREG=%s BREG=%s ACASCREG=%s BCASCREG=%s MREG=%s PREG=%s OPMODEREG=%s A_INPUT=%s B_INPUT=%s AMULTSEL=%s" $c \
        [get_property AREG $c] [get_property BREG $c] [get_property ACASCREG $c] [get_property BCASCREG $c] \
        [get_property MREG $c] [get_property PREG $c] [get_property OPMODEREG $c] [get_property A_INPUT $c] \
        [get_property B_INPUT $c] [get_property AMULTSEL $c]]
}
close $fp
report_timing_summary -file [file join $outdir timing.rpt]
puts "NETLIST_DONE=[file join $outdir ${top}_funcsim.v]"
