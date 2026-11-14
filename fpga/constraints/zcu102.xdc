# zcu102.xdc : physical and timing constraints for sumfold_top on the ZCU102 (xczu9eg-ffvb1156-2-e).
#
# Pin sources (checked 2026-09-27 against the Vivado 2022.2 board store, <vivado-2022.2>/data/xhub/boards/XilinxBoardStore/boards/Xilinx/zcu102/3.4/part0_pins.xml):
#   sysclk_p  G21  LVDS_25   (index 0)       sysclk_n  F21  LVDS_25   (index 1)
# GPIO_LED_0_LS .. GPIO_LED_7_LS : AG14 AF13 AE13 AJ14 AJ15 AH13 AH14 AL12, LVCMOS33 The 3.4 board.xml declares no interface/frequency for sysclk (only user_si570_sysclk, 300 MHz); 125 MHz for the G21/F21 pair is the ZCU102 user guide value (UG1182, CLK_125) and is the value assumed by the MMCM settings in fpga/rtl/sumfold_top.v (CLKIN1_PERIOD = 8.000).

# ------------------------------------------------------------------ system clock (125 MHz)
set_property PACKAGE_PIN G21     [get_ports sysclk_p]
set_property PACKAGE_PIN F21     [get_ports sysclk_n]
set_property IOSTANDARD  LVDS_25 [get_ports {sysclk_p sysclk_n}]
create_clock -period 8.000 -name sysclk_125 [get_ports sysclk_p]
# The kernel clock (MMCME4_ADV CLKOUT0, KCLK_MHZ) is derived automatically by Vivado. The JTAG TCK clock of the debug hub / BSCAN is created automatically when dbg_hub is inserted.

# ------------------------------------------------------------------ user LEDs (status only)
set_property PACKAGE_PIN AG14 [get_ports {led[0]}]
set_property PACKAGE_PIN AF13 [get_ports {led[1]}]
set_property PACKAGE_PIN AE13 [get_ports {led[2]}]
set_property PACKAGE_PIN AJ14 [get_ports {led[3]}]
set_property PACKAGE_PIN AJ15 [get_ports {led[4]}]
set_property PACKAGE_PIN AH13 [get_ports {led[5]}]
set_property PACKAGE_PIN AH14 [get_ports {led[6]}]
set_property PACKAGE_PIN AL12 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
set_property DRIVE 8             [get_ports {led[*]}]
set_property SLEW  SLOW          [get_ports {led[*]}]
# LEDs are human-visible status outputs; no output timing requirement.
set_false_path -to [get_ports {led[*]}]

# ------------------------------------------------------------------ reset
# MMCM LOCKED is an asynchronous status signal; it drives the asynchronous clear of the reset synchronizer in sumfold_top (lock_sync, rst_cnt, aresetn). LOCKED is not a clocked startpoint, so these clear paths are untimed by construction (a set_false_path -from LOCKED is rejected by Vivado with Constraints 18-513); the synchronized aresetn is fully timed in the kernel clock domain.

# ------------------------------------------------------------------ bitstream
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
