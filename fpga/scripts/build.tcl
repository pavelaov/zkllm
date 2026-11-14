# build.tcl : Vivado 2022.2 flow for the ZCU102 SumFold board design (synthesis, placement, routing, reports, bitstream).
#
# usage (from any directory; OUTDIR is created):
#   vivado -mode batch -nojournal -log <OUTDIR>/vivado.log -source fpga/scripts/build.tcl \
#          -tclargs LANES KCLK_MHZ OUTDIR [STRATEGY] [LOG_NMAX]
#   LANES    : 1 | 2 | 4 | 8
# KCLK_MHZ : 100 | 125 | 150 | 200 | 250 | 300 (MMCM settings in fpga/rtl/sumfold_top.v) STRATEGY : default (default) | perf | perf_rt default = Vivado default directives everywhere
#              perf    = synth -keep_equivalent_registers (keeps the per-lane copies of the
# broadcast address/control registers instead of merging them into one chip-wide driver), place ExtraTimingOpt, phys_opt AggressiveExplore, route AggressiveExplore perf_rt = perf + synth -retiming LOG_NMAX : log2 of the maximum table size (default 16, the paper configuration; smaller values are only used for quick flow smoke tests) No '=' characters in the arguments: the Vivado .bat wrapper splits arguments at '='.
#
# Outputs in OUTDIR:
#   summary.txt              KEY=VALUE lines (flushed after every step; parsed by collect_impl.py)
#   reports/*.rpt            utilization (flat, hierarchical), timing summary, power, clock utilization, DRC, methodology, route status, ... post_route.dcp, sumfold_top.bit, sumfold_top.ltx (if the tool writes one)
#   ip/jtag_axi_0/...        the JTAG-to-AXI master IP, generated and synthesized out of context
#
# This script never opens the hardware manager and never programs a device.

if {[llength $argv] < 3} {
    puts "usage: -tclargs LANES KCLK_MHZ OUTDIR ?STRATEGY? ?LOG_NMAX?"
    exit 2
}
set lanes    [lindex $argv 0]
set kclk     [lindex $argv 1]
set outdir   [file normalize [lindex $argv 2]]
set strategy [expr {[llength $argv] > 3 ? [lindex $argv 3] : "default"}]
set lognmax  [expr {[llength $argv] > 4 ? [lindex $argv 4] : 16}]
set part     xczu9eg-ffvb1156-2-e
set top      sumfold_top

set here [file dirname [file normalize [info script]]]
set repo [file normalize [file join $here .. ..]]
set rtl  [file join $repo fpga rtl]
set xdc  [file join $repo fpga constraints zcu102.xdc]
set rpt  [file join $outdir reports]
file mkdir $outdir
file mkdir $rpt

set_param general.maxThreads 8

# ------------------------------------------------------------------ summary helpers
set ::summ [open [file join $outdir summary.txt] w]
proc kv {k v} {
    puts $::summ "$k=$v"
    flush $::summ
    puts "SFI_$k=$v"
}
proc now {} { return [clock seconds] }
proc step {name body} {
    set t0 [now]
    kv STEP_${name}_START [clock format $t0 -format {%Y-%m-%dT%H:%M:%S}]
    uplevel 1 $body
    set t1 [now]
    kv STEP_${name}_SECONDS [expr {$t1 - $t0}]
}
proc worst_slack {kind} {
    set p [get_timing_paths -$kind -max_paths 1 -nworst 1]
    if {[llength $p] == 0} { return "NA" }
    return [get_property SLACK [lindex $p 0]]
}

kv LANES $lanes
kv KCLK_MHZ $kclk
kv PERIOD_NS [format %.4f [expr {1000.0 / $kclk}]]
kv STRATEGY $strategy
kv LOG_NMAX $lognmax
kv PART $part
kv TOP $top
kv OUTDIR $outdir
kv VIVADO_VERSION [version -short]
kv VIVADO_BUILD [lindex [split [version] "\n"] 1]
kv HOST [info hostname]
kv MAXTHREADS [get_param general.maxThreads]
kv START [clock format [now] -format {%Y-%m-%dT%H:%M:%S}]
set t_start [now]

switch -- $strategy {
    default {
        set synth_extra {}
        set place_dir   Default
        set phys_dir    Default
        set route_dir   Default
    }
    perf {
        set synth_extra {-keep_equivalent_registers}
        set place_dir   ExtraTimingOpt
        set phys_dir    AggressiveExplore
        set route_dir   AggressiveExplore
    }
    perf_rt {
        set synth_extra {-keep_equivalent_registers -retiming}
        set place_dir   ExtraTimingOpt
        set phys_dir    AggressiveExplore
        set route_dir   AggressiveExplore
    }
    default {
        puts "unknown STRATEGY $strategy"
        exit 2
    }
}
kv SYNTH_EXTRA "$synth_extra"
kv PLACE_DIRECTIVE $place_dir
kv PHYS_OPT_DIRECTIVE $phys_dir
kv ROUTE_DIRECTIVE $route_dir

create_project -in_memory -part $part
set_property target_language Verilog [current_project]
set_property default_lib xil_defaultlib [current_project]

# ------------------------------------------------------------------ JTAG-to-AXI master IP
step IP {
    set ipdir [file join $outdir ip]
    file delete -force $ipdir
    file mkdir $ipdir
    create_ip -name jtag_axi -vendor xilinx.com -library ip -module_name jtag_axi_0 -dir $ipdir
    # PROTOCOL: 0 = AXI4, 2 = AXI4LITE (jtag_axi_v1_2 component.xml, choice_pairs_230a70fe)
    set_property -dict [list CONFIG.PROTOCOL {2} CONFIG.M_AXI_DATA_WIDTH {32} CONFIG.M_AXI_ADDR_WIDTH {32}] \
        [get_ips jtag_axi_0]
    report_property [get_ips jtag_axi_0] -file [file join $rpt ip_jtag_axi_0_props.rpt]
    generate_target all [get_ips jtag_axi_0]
    synth_ip [get_ips jtag_axi_0]
    kv IP_PROTOCOL [get_property CONFIG.PROTOCOL [get_ips jtag_axi_0]]
    kv IP_VERSION  [get_property IPDEF [get_ips jtag_axi_0]]
}

# ------------------------------------------------------------------ RTL + constraints
read_verilog [list \
    [file join $rtl gl_arith.v] [file join $rtl ext_arith.v] [file join $rtl sdp_bram.v] \
    [file join $rtl adder_tree.v] [file join $rtl sumfold_lane.v] [file join $rtl sumfold_core.v] \
    [file join $rtl sumfold_axil.v] [file join $rtl sumfold_top.v]]
read_xdc $xdc

# ------------------------------------------------------------------ synthesis
step SYNTH {
    synth_design -top $top -part $part -include_dirs $rtl \
        -generic LANES=$lanes -generic KCLK_MHZ=$kclk -generic LOG_NMAX=$lognmax {*}$synth_extra
}
report_utilization    -file [file join $rpt post_synth_util.rpt]
report_timing_summary -max_paths 10 -file [file join $rpt post_synth_timing_summary.rpt]
kv SYNTH_WNS [worst_slack setup]

# ------------------------------------------------------------------ implementation
step OPT   { opt_design }
step PLACE { place_design -directive $place_dir }
kv PLACE_WNS [worst_slack setup]
step PHYS_OPT { phys_opt_design -directive $phys_dir }
kv PHYS_OPT_WNS [worst_slack setup]
step ROUTE { route_design -directive $route_dir }
set wns [worst_slack setup]
kv ROUTE_WNS $wns
set post_phys 0
if {$wns ne "NA" && $wns < 0} {
    set post_phys 1
    step POST_ROUTE_PHYS_OPT { phys_opt_design -directive AggressiveExplore }
    kv POST_ROUTE_PHYS_OPT_WNS [worst_slack setup]
}
kv POST_ROUTE_PHYS_OPT_RAN $post_phys

# ------------------------------------------------------------------ reports (post-route)
step REPORTS {
    report_route_status                 -file [file join $rpt route_status.rpt]
    report_timing_summary -max_paths 20 -report_unconstrained -warn_on_violation \
                                        -file [file join $rpt timing_summary.rpt]
    report_timing -max_paths 50 -nworst 1 -sort_by slack -path_type full_clock_expanded \
                                        -file [file join $rpt timing_paths_setup.rpt]
    report_timing -hold -max_paths 20 -nworst 1 -sort_by slack \
                                        -file [file join $rpt timing_paths_hold.rpt]
    report_utilization                  -file [file join $rpt util.rpt]
    report_utilization -hierarchical -hierarchical_depth 4 \
                                        -file [file join $rpt util_hier.rpt]
    report_clock_utilization            -file [file join $rpt clock_util.rpt]
    report_clocks                       -file [file join $rpt clocks.rpt]
    report_clock_interaction            -file [file join $rpt clock_interaction.rpt]
    check_timing -verbose               -file [file join $rpt check_timing.rpt]
    report_drc                          -file [file join $rpt drc.rpt]
    report_methodology                  -file [file join $rpt methodology.rpt]
        # Vectorless power analysis with the default switching activity.
    report_power                        -file [file join $rpt power.rpt]
    report_design_analysis -timing -congestion -complexity \
                                        -file [file join $rpt design_analysis.rpt]
}
set wns [worst_slack setup]
set whs [worst_slack hold]
kv FINAL_WNS $wns
kv FINAL_WHS $whs
foreach {k expr_} {
    ROUTE_UNROUTED_NETS {llength [get_nets -hier -quiet -filter {ROUTE_STATUS == UNROUTED}]}
    ROUTE_CONFLICT_NETS {llength [get_nets -hier -quiet -filter {ROUTE_STATUS == CONFLICTS}]}
    CELL_DSP48E2        {llength [get_cells -hier -quiet -filter {REF_NAME == DSP48E2}]}
    CELL_RAMB36         {llength [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.RAMB36*}]}
    CELL_RAMB18         {llength [get_cells -hier -quiet -filter {PRIMITIVE_TYPE =~ BLOCKRAM.BRAM.RAMB18*}]}
    DBG_HUB_CELLS       {llength [get_cells -hier -quiet -filter {REF_NAME =~ dbg_hub*}]}
    KCLK_CLOCK_NAME     {get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT0}]}
    KCLK_PERIOD_TOOL    {get_property PERIOD [get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT0}]]}
} {
    if {[catch {set v [eval $expr_]} e]} { set v "NA" }
    kv $k $v
}

step CHECKPOINT { write_checkpoint -force [file join $outdir post_route.dcp] }

# ------------------------------------------------------------------ bitstream (+ debug probes)
set bit [file join $outdir ${top}.bit]
set ltx [file join $outdir ${top}.ltx]
set bit_ok 0
step BITSTREAM {
    if {[catch {write_bitstream -force $bit} err]} {
        kv BITSTREAM_ERROR [string map {"\n" " "} $err]
    } else {
        set bit_ok 1
    }
}
kv BITSTREAM_OK $bit_ok
kv BITSTREAM [expr {$bit_ok ? $bit : "NA"}]
set ltx_ok 0
if {[catch {write_debug_probes -force $ltx} err]} {
    kv LTX_ERROR [string map {"\n" " "} $err]
} elseif {[file exists $ltx]} {
    set ltx_ok 1
}
kv LTX [expr {$ltx_ok ? $ltx : "NA"}]

kv TOTAL_SECONDS [expr {[now] - $t_start}]
kv END [clock format [now] -format {%Y-%m-%dT%H:%M:%S}]
kv BUILD_STATUS DONE
close $::summ
