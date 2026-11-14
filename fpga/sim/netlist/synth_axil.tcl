# synth_axil.tcl : out-of-context synthesis of the whole SumFold register interface + core (sumfold_axil) with the board-build synthesis options, then a functional simulation netlist for fpga/sim/replay/replay.py --dut-netlist (post-synthesis netlist of the complete kernel). usage: vivado -mode batch -source synth_axil.tcl -tclargs <outdir> <LANES> <LOG_NMAX> <KCLK_MHZ> [default|perf] [+legacy]
set outdir  [file normalize [lindex $argv 0]]
set lanes   [lindex $argv 1]
set lognmax [lindex $argv 2]
set kclk    [lindex $argv 3]
set strat   [expr {[llength $argv] > 4 ? [lindex $argv 4] : "default"}]
set legacy  [expr {[lsearch $argv "+legacy"] >= 0}]
set here [file dirname [file normalize [info script]]]
set rtl  [file normalize [file join $here .. .. rtl]]
file mkdir $outdir
set_param general.maxThreads 8
read_verilog [list $rtl/gl_arith.v $rtl/ext_arith.v $rtl/sdp_bram.v $rtl/adder_tree.v \
                   $rtl/sumfold_lane.v $rtl/sumfold_core.v $rtl/sumfold_axil.v]
set extra {}
if {$strat eq "perf"} { lappend extra -keep_equivalent_registers }
if {$legacy} { lappend extra -verilog_define GL_MUL_LEGACY_INFER }
synth_design -top sumfold_axil -part xczu9eg-ffvb1156-2-e -mode out_of_context -include_dirs $rtl \
    -generic LANES=$lanes -generic LOG_NMAX=$lognmax -generic KCLK_MHZ=$kclk {*}$extra
puts "DSP48E2=[llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]"
write_verilog -mode funcsim -force [file join $outdir sumfold_axil_funcsim.v]
puts "NETLIST_DONE=[file join $outdir sumfold_axil_funcsim.v]"
