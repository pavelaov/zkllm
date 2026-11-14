// tb_top.sv : elaboration + clocking smoke test of sumfold_top (unisim MMCME4_ADV, IBUFDS, BUFG)
// with the idle jtag_axi_0 stub. Checks that the MMCM locks, reset is released and the kernel
// clock period matches KCLK_MHZ.
`timescale 1ps / 1ps
module tb_top;
    parameter KCLK_MHZ = 300;
    reg sysclk_p = 1'b0;
    always #4000 sysclk_p = ~sysclk_p;      // 125 MHz
    wire sysclk_n = ~sysclk_p;
    wire [7:0] led;

    sumfold_top #(.LANES(2), .LOG_NMAX(16), .KCLK_MHZ(KCLK_MHZ)) dut (
        .sysclk_p(sysclk_p), .sysclk_n(sysclk_n), .led(led));

    realtime t_a, t_b;
    initial begin
        wait (led[4] === 1'b1);                // MMCM locked (synchronized)
        $display("TOP locked at %0t ps", $realtime);
        wait (dut.aresetn === 1'b1);
        $display("TOP aresetn released at %0t ps", $realtime);
        @(posedge dut.kclk); t_a = $realtime;
        repeat (100) @(posedge dut.kclk);
        t_b = $realtime;
        $display("TOP kclk period_ps=%0.1f expected_ps=%0.1f", (t_b - t_a) / 100.0, 1.0e6 / KCLK_MHZ);
        if ((t_b - t_a) / 100.0 > 0.99e6 / KCLK_MHZ && (t_b - t_a) / 100.0 < 1.01e6 / KCLK_MHZ &&
            led[3] === 1'b0)
            $display("TOP_RESULT PASS");
        else
            $display("TOP_RESULT FAIL");
        $finish;
    end
    initial begin
        #200000000;
        $display("TOP_RESULT FAIL (timeout)");
        $finish;
    end
endmodule
