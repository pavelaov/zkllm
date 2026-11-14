# sumfold_top_pins.xdc : reference pin/clock constraints for sumfold_top on the ZCU102 (rev 1.x), taken from the probed board facts in docs/PAPER_BRIEF.md section 7. The implementation flow may copy this file to fpga/constraints/.
set_property PACKAGE_PIN G21      [get_ports sysclk_p]
set_property PACKAGE_PIN F21      [get_ports sysclk_n]
set_property IOSTANDARD  LVDS_25  [get_ports {sysclk_p sysclk_n}]
create_clock -period 8.000 -name sysclk_125 [get_ports sysclk_p]
# the kernel clock is derived automatically from the MMCME4_ADV (CLKOUT0)

set_property PACKAGE_PIN AG14 [get_ports {led[0]}]
set_property PACKAGE_PIN AF13 [get_ports {led[1]}]
set_property PACKAGE_PIN AE13 [get_ports {led[2]}]
set_property PACKAGE_PIN AJ14 [get_ports {led[3]}]
set_property PACKAGE_PIN AJ15 [get_ports {led[4]}]
set_property PACKAGE_PIN AH13 [get_ports {led[5]}]
set_property PACKAGE_PIN AH14 [get_ports {led[6]}]
set_property PACKAGE_PIN AL12 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[*]}]
set_false_path -to [get_ports {led[*]}]
