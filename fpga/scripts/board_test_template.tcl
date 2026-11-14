# board_test_template.tcl : SumFold on-board test through the JTAG-to-AXI master (hw_axi_1). Generated into a concrete script by fpga/host/jtag_board_test.py (placeholders are upper-case names between double at-signs). Run:  vivado -mode batch -nojournal -log <log> -source <generated>.tcl
#
# Flow: open hw target -> program bitstream (volatile PL configuration only) -> check ID/CFG -> LOGN/seeds -> GEN (on-chip xorshift tables) -> AUTO: load r_0..r_{R-1}, START, poll DONE STEP: START, then per round: poll WAIT, read (s0,s1,s2), compute r_i with the SHA-256 transcript in Tcl (tcllib sha256), write CH[i], START -> read messages, lane finals, cycle counters -> print SFB_<KEY>=<VALUE> lines. DRY_RUN = 1 never opens the hardware manager: AXI accesses go to a register emulator that answers with the golden values, which exercises the full control flow and the Tcl transcript.

set BITSTREAM   {@@BITSTREAM@@}
set PROBES      {@@PROBES@@}
set HW_SERVER   {@@HW_SERVER@@}
set HW_TARGET   {@@HW_TARGET@@}
set HW_DEVICE   {@@HW_DEVICE@@}
set PROGRAM     @@PROGRAM@@
set LOGN        @@LOGN@@
set LANES_EXP   @@LANES@@
set SEED_F      0x@@SEED_F@@
set SEED_G      0x@@SEED_G@@
set MODE        {@@MODE@@}
set CHALLENGES  {@@CHALLENGES@@}
set DRY_RUN     @@DRY_RUN@@
set EMU_MSGS    {@@EMU_MSGS@@}
set EMU_FINALS  {@@EMU_FINALS@@}
set POLL_MAX    @@POLL_MAX@@

set P 0xFFFFFFFF00000001

# ---------------------------------------------------------------- register map (spec section 5)
array set REG {
    ID 0x000 CFG 0x004 CTRL 0x008 STATUS 0x00C LOGN 0x010 ROUND 0x014
    SEED_F_LO 0x018 SEED_F_HI 0x01C SEED_G_LO 0x020 SEED_G_HI 0x024
    CH_IDX 0x040 CH_W0 0x044 CH_WE 0x054 RES_IDX 0x060 RES_W0 0x064
    CYC_TOTAL_LO 0x080 CYC_TOTAL_HI 0x084 CYC_GEN_LO 0x088 CYC_GEN_HI 0x08C
    CYC_ROUND_IDX 0x090 CYC_ROUND 0x094 CYC_WALL_LO 0x098 CYC_WALL_HI 0x09C KCLK 0x0F0
}

proc kv {k v} { puts "SFB_$k=$v" }

# ---------------------------------------------------------------- field and transcript helpers
proc u32le {v} { return [binary format i [expr {$v & 0xFFFFFFFF}]] }
proc u64le {v} { return [binary format ii [expr {$v & 0xFFFFFFFF}] [expr {($v >> 32) & 0xFFFFFFFF}]] }
proc enc {e} { return "[u64le [lindex $e 0]][u64le [lindex $e 1]]" }
proc hex64 {v} { return [format %08x%08x [expr {($v >> 32) & 0xFFFFFFFF}] [expr {$v & 0xFFFFFFFF}]] }
proc ehex {e} { return "[hex64 [lindex $e 1]][hex64 [lindex $e 0]]" }
proc parse_ehex {h} {
    set c1 [expr "0x[string range $h 0 15]"]
    set c0 [expr "0x[string range $h 16 31]"]
    return [list $c0 $c1]
}
proc eadd {a b} {
    global P
    return [list [expr {([lindex $a 0] + [lindex $b 0]) % $P}] [expr {([lindex $a 1] + [lindex $b 1]) % $P}]]
}

set HAVE_SHA256 [expr {![catch {package require sha256}]}]
if {!$HAVE_SHA256} {
    kv STEP_SUPPORTED 0
    kv NOTE "tcllib sha256 not available in this Vivado Tcl; STEP mode cannot compute challenges on the host side"
}

proc tr_init {n claim} {
    global TR_STATE
    set TR_STATE [sha2::sha256 -bin "EDGE-CERT/sumfold/v1[u32le $n][enc $claim]"]
}
proc tr_round {s0 s1 s2} {
    global TR_STATE P
    set TR_STATE [sha2::sha256 -bin "$TR_STATE[enc $s0][enc $s1][enc $s2]"]
    binary scan $TR_STATE iu4 w
    set c0 [expr {((wide([lindex $w 1]) << 32) | [lindex $w 0]) % $P}]
    set c1 [expr {((wide([lindex $w 3]) << 32) | [lindex $w 2]) % $P}]
    return [list $c0 $c1]
}

# ---------------------------------------------------------------- register emulator (DRY_RUN)
array set EMU {}
proc emu_reset {} {
    global EMU LANES_EXP
    array unset EMU
    array set EMU [list 0 0xEDCE2026 4 [expr {$LANES_EXP | (16 << 8) | (250 << 16)}] 12 0 20 0 96 0 144 0]
    set EMU(step_round) -1
}
proc emu_wr {addr data} {
    global EMU EMU_MSGS EMU_FINALS LOGN LANES_EXP
    set addr [expr {$addr + 0}]
    set EMU($addr) $data
    set R [expr {$LOGN - int(log($LANES_EXP) / log(2) + 0.5)}]
    if {$addr == 0x008} {
        set mode [expr {($data >> 4) & 1}]
        if {$data & 2} { set EMU(12) [expr {$EMU(12) | 0x10}] }
        if {$data & 1} {
            if {!$mode} {
                set EMU(12) [expr {($EMU(12) & 0x10) | 0x2}]
                set EMU(20) $R
            } else {
                incr EMU(step_round)
                if {$EMU(step_round) < $R} {
                    set EMU(12) [expr {($EMU(12) & 0x10) | 0x9}]
                    set EMU(20) $EMU(step_round)
                } else {
                    set EMU(12) [expr {($EMU(12) & 0x10) | 0x2}]
                    set EMU(20) $R
                }
            }
        }
    }
}
proc emu_rd {addr} {
    global EMU EMU_MSGS EMU_FINALS
    set addr [expr {$addr + 0}]
    if {$addr >= 0x064 && $addr <= 0x070} {
        set idx $EMU(96)
        if {$idx >= 48} { set h [lindex $EMU_FINALS [expr {$idx - 48}]] } else { set h [lindex $EMU_MSGS $idx] }
        if {$h eq ""} { return 0 }
        set k [expr {($addr - 0x064) / 4}]
        set w [string range $h [expr {24 - 8 * $k}] [expr {31 - 8 * $k}]]
        return [expr "0x$w"]
    }
    if {[info exists EMU($addr)]} { return $EMU($addr) }
    return 0
}

# ---------------------------------------------------------------- AXI access
proc axi_wr {addr data} {
    global DRY_RUN AXI
    if {$DRY_RUN} { emu_wr $addr $data; return }
    create_hw_axi_txn -force sf_wr $AXI -address [format %08X $addr] \
        -data [format %08X [expr {$data & 0xFFFFFFFF}]] -len 1 -type write
    run_hw_axi -quiet [get_hw_axi_txns sf_wr]
}
proc axi_rd {addr} {
    global DRY_RUN AXI
    if {$DRY_RUN} { return [emu_rd $addr] }
    create_hw_axi_txn -force sf_rd $AXI -address [format %08X $addr] -len 1 -type read
    run_hw_axi -quiet [get_hw_axi_txns sf_rd]
    return [expr "0x[get_property DATA [get_hw_axi_txns sf_rd]]"]
}
proc wr128 {base e} {
    axi_wr $base                  [expr {[lindex $e 0] & 0xFFFFFFFF}]
    axi_wr [expr {$base + 4}]     [expr {([lindex $e 0] >> 32) & 0xFFFFFFFF}]
    axi_wr [expr {$base + 8}]     [expr {[lindex $e 1] & 0xFFFFFFFF}]
    axi_wr [expr {$base + 12}]    [expr {([lindex $e 1] >> 32) & 0xFFFFFFFF}]
}
proc rd_res {idx} {
    global REG
    axi_wr $REG(RES_IDX) $idx
    set w0 [axi_rd 0x064]
    set w1 [axi_rd 0x068]
    set w2 [axi_rd 0x06C]
    set w3 [axi_rd 0x070]
    return [list [expr {(wide($w1) << 32) | $w0}] [expr {(wide($w3) << 32) | $w2}]]
}
proc rd64 {lo hi} { return [expr {(wide([axi_rd $hi]) << 32) | [axi_rd $lo]}] }
proc poll_status {mask what} {
    global REG POLL_MAX
    for {set i 0} {$i < $POLL_MAX} {incr i} {
        set s [axi_rd $REG(STATUS)]
        if {$s & 0x4} { kv ERROR "STATUS.ERROR while waiting for $what (STATUS=[format %08X $s])"; return $s }
        if {$s & $mask} { return $s }
    }
    kv ERROR "timeout waiting for $what"
    return -1
}

# ---------------------------------------------------------------- hardware bring-up
kv DRY_RUN $DRY_RUN
kv MODE $MODE
kv LOGN $LOGN
kv SEED_F [hex64 $SEED_F]
kv SEED_G [hex64 $SEED_G]
kv TCL_PATCHLEVEL [info patchlevel]
kv SHA256_TCLLIB $HAVE_SHA256

if {$DRY_RUN} {
    emu_reset
    set AXI emu
} else {
    open_hw_manager
    connect_hw_server -url $HW_SERVER -allow_non_jtag
    set tgt [lindex [get_hw_targets $HW_TARGET] 0]
    if {$tgt eq ""} { kv ERROR "hw target $HW_TARGET not found"; exit 2 }
    current_hw_target $tgt
    open_hw_target
    set dev [lindex [get_hw_devices $HW_DEVICE] 0]
    current_hw_device $dev
    refresh_hw_device -update_hw_probes false $dev
    if {$PROGRAM} {
        set_property PROGRAM.FILE $BITSTREAM $dev
        if {$PROBES ne ""} { set_property PROBES.FILE $PROBES $dev }
        program_hw_devices $dev
        kv PROGRAMMED 1
    }
    refresh_hw_device $dev
    set AXI [lindex [get_hw_axis hw_axi_1] 0]
    if {$AXI eq ""} { kv ERROR "hw_axi_1 not found"; exit 3 }
    reset_hw_axi $AXI
}

# ---------------------------------------------------------------- SYSMON (die temperature, rails)
# Prints SFB_SYSMON_<TAG>_<PROP>; hardware only, never fatal (errors are reported as a value).
proc sysmon_sample {tag} {
    global DRY_RUN
    if {$DRY_RUN} { return }
    if {[catch {
        set sm [lindex [get_hw_sysmons] 0]
        refresh_hw_sysmon $sm
        foreach prop {TEMPERATURE VCCINT VCCAUX VCCBRAM} {
            if {![catch {get_property $prop $sm} v]} { kv SYSMON_${tag}_$prop $v }
        }
    } err]} { kv SYSMON_${tag}_ERR [string map {"\n" " "} $err] }
}
sysmon_sample BEFORE

# ---------------------------------------------------------------- identity
set id  [axi_rd $REG(ID)]
set cfg [axi_rd $REG(CFG)]
kv ID  [format %08X $id]
kv CFG [format %08X $cfg]
set LANES [expr {$cfg & 0xFF}]
set LG_L  [expr {$LANES >= 8 ? 3 : ($LANES >= 4 ? 2 : ($LANES >= 2 ? 1 : 0))}]
set R     [expr {$LOGN - $LG_L}]
kv LANES $LANES
kv KCLK_MHZ [expr {($cfg >> 16) & 0xFFFF}]
kv DEVICE_ROUNDS $R
if {$id != 0xEDCE2026} { kv ERROR "bad ID"; exit 4 }
if {$LANES != $LANES_EXP} { kv ERROR "LANES mismatch (bitstream $LANES, expected $LANES_EXP)"; exit 5 }

# ---------------------------------------------------------------- GEN
set mode_bit [expr {$MODE eq "step" ? 0x10 : 0x0}]
axi_wr $REG(CTRL) [expr {0x4 | $mode_bit}]
axi_wr $REG(LOGN) $LOGN
axi_wr $REG(SEED_F_LO) [expr {$SEED_F & 0xFFFFFFFF}]
axi_wr $REG(SEED_F_HI) [expr {($SEED_F >> 32) & 0xFFFFFFFF}]
axi_wr $REG(SEED_G_LO) [expr {$SEED_G & 0xFFFFFFFF}]
axi_wr $REG(SEED_G_HI) [expr {($SEED_G >> 32) & 0xFFFFFFFF}]
set t0 [clock microseconds]
axi_wr $REG(CTRL) [expr {0x2 | $mode_bit}]
poll_status 0x10 GEN_DONE
kv GEN_HOST_US [expr {[clock microseconds] - $t0}]

# ---------------------------------------------------------------- job
if {$MODE eq "auto"} {
    for {set i 0} {$i < $R} {incr i} {
        axi_wr $REG(CH_IDX) $i
        wr128 $REG(CH_W0) [parse_ehex [lindex $CHALLENGES $i]]
        axi_wr $REG(CH_WE) 1
    }
    set t0 [clock microseconds]
    axi_wr $REG(CTRL) 0x1
    set st [poll_status 0x2 DONE]
    kv JOB_HOST_US [expr {[clock microseconds] - $t0}]
} else {
    if {!$HAVE_SHA256} { kv ERROR "STEP mode needs tcllib sha256"; exit 6 }
    set t0 [clock microseconds]
    axi_wr $REG(CTRL) 0x11
    set tstart [clock microseconds]
    for {set i 0} {$i < $R} {incr i} {
        set st [poll_status 0x8 "WAIT round $i"]
        if {$st < 0 || ($st & 0x4)} { break }
        set tr0 [clock microseconds]
        # host time from the START write that launched this round until WAIT was observed
        kv STEP_POLL_US_$i [expr {$tr0 - $tstart}]
        set s0 [rd_res [expr {3 * $i}]]
        set s1 [rd_res [expr {3 * $i + 1}]]
        set s2 [rd_res [expr {3 * $i + 2}]]
        if {$i == 0} {
            # the claim H = sum_b f(b) g(b) equals s0 + s1 of round 0
            set claim [eadd $s0 $s1]
            kv CLAIM [ehex $claim]
            tr_init $LOGN $claim
        }
        set r [tr_round $s0 $s1 $s2]
        kv CH_$i [ehex $r]
        axi_wr $REG(CH_IDX) $i
        wr128 $REG(CH_W0) $r
        axi_wr $REG(CH_WE) 1
        axi_wr $REG(CTRL) 0x11
        set tstart [clock microseconds]
        kv STEP_HOST_US_$i [expr {$tstart - $tr0}]
    }
    set st [poll_status 0x2 DONE]
    kv JOB_HOST_US [expr {[clock microseconds] - $t0}]
}
kv STATUS [format %08X [axi_rd $REG(STATUS)]]
kv ROUND [axi_rd $REG(ROUND)]

# ---------------------------------------------------------------- results
for {set i 0} {$i < $R} {incr i} {
    for {set k 0} {$k < 3} {incr k} {
        kv MSG_${i}_${k} [ehex [rd_res [expr {3 * $i + $k}]]]
    }
}
for {set l 0} {$l < $LANES} {incr l} {
    kv FIN_F_$l [ehex [rd_res [expr {48 + 2 * $l}]]]
    kv FIN_G_$l [ehex [rd_res [expr {48 + 2 * $l + 1}]]]
}
for {set i 0} {$i <= $R} {incr i} {
    axi_wr $REG(CYC_ROUND_IDX) $i
    kv CYC_PASS_$i [axi_rd $REG(CYC_ROUND)]
}
kv CYC_TOTAL [rd64 $REG(CYC_TOTAL_LO) $REG(CYC_TOTAL_HI)]
kv CYC_WALL  [rd64 $REG(CYC_WALL_LO) $REG(CYC_WALL_HI)]
kv CYC_GEN   [rd64 $REG(CYC_GEN_LO) $REG(CYC_GEN_HI)]
sysmon_sample AFTER

if {!$DRY_RUN} {
    close_hw_target
    disconnect_hw_server
    close_hw_manager
}
kv END 1
