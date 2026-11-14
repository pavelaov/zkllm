@echo off
rem usage: run_mulnet_dir.cmd <netlist dir containing gl_mul_funcsim.v>
cd /d %~dp0
set XB=<vivado-2022.2>\bin
set D=%1
if not exist work_%D% mkdir work_%D%
cd work_%D%
call %XB%\xvlog.bat ../%D%/gl_mul_funcsim.v > xvlog_net.out 2>&1
call %XB%\xvlog.bat --sv ../tb_mul_netlist.sv ../tb_mul_impulse.sv > xvlog_tb.out 2>&1
call %XB%\xelab.bat -debug off -L unisims_ver -s tb_mulnet work.tb_mul_netlist work.glbl > xelab_net.out 2>&1
call %XB%\xelab.bat -debug off -L unisims_ver -s tb_mulimp work.tb_mul_impulse work.glbl > xelab_imp.out 2>&1
call %XB%\xsim.bat tb_mulnet -R -testplusarg "GOLD=../../golden" -testplusarg "GAP=0" > xsim_gap0.out 2>&1
call %XB%\xsim.bat tb_mulnet -R -testplusarg "GOLD=../../golden" -testplusarg "GAP=1" > xsim_gap1.out 2>&1
call %XB%\xsim.bat tb_mulimp -R > xsim_imp.out 2>&1
