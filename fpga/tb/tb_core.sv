// tb_core.sv : drives sumfold_core directly and checks every round message and lane final against the golden model (fpga/host/make_golden.py -> fpga/sim/golden/). One elaboration per LANES value; the case list is cases_L<LANES>.txt. Output lines (parsed by fpga/sim/run_sim.py): CASE name=<> L=<> n=<> src=<gen|mem> mode=<auto|step> R=<> result=<PASS|FAIL> errors=<k> CYC  name=<> L=<> n=<> pass=<i> kind=<eval0|fused|fold> pairs=<p> cycles=<c> CYCT name=<> L=<> n=<> total=<> wall=<> gen=<>
`timescale 1ns / 1ps

module tb_core;
    parameter LANES     = 1;
    parameter LOG_NMAX  = 16;
    parameter DRAIN_ADJ = 0;
    localparam LG_L  = (LANES >= 8) ? 3 : (LANES >= 4) ? 2 : (LANES >= 2) ? 1 : 0;
    localparam MAXC  = 400000;

    reg clk = 1'b0;
    always #2 clk = ~clk;

    reg          rst = 1'b1;
    reg          cmd_start = 0, cmd_gen = 0, cmd_soft = 0, cfg_step = 0;
    reg  [7:0]   cfg_logn = 0;
    reg  [63:0]  seed_f = 64'd1, seed_g = 64'd2;
    reg          ch_we = 0;
    reg  [3:0]   ch_idx = 0;
    reg  [127:0] ch_data = 0;
    reg  [6:0]   res_idx = 0;
    wire [127:0] res_data;
    reg  [4:0]   cyc_idx = 0;
    wire [31:0]  cyc_round;
    wire [63:0]  cyc_total, cyc_gen, cyc_wall;
    reg          mem_we = 0, mem_re = 0, mem_sel = 0;
    reg  [LOG_NMAX-1:0] mem_addr = 0;
    reg  [127:0] mem_wdata = 0;
    wire [127:0] mem_rdata;
    wire         mem_rvalid;
    wire         st_busy, st_done, st_error, st_wait, st_gen_done;
    wire [7:0]   st_round;

    sumfold_core #(.LANES(LANES), .LOG_NMAX(LOG_NMAX), .DRAIN_ADJ(DRAIN_ADJ)) dut (
        .clk(clk), .rst(rst),
        .cmd_start(cmd_start), .cmd_gen(cmd_gen), .cmd_soft_reset(cmd_soft),
        .cfg_step(cfg_step), .cfg_logn(cfg_logn), .cfg_seed_f(seed_f), .cfg_seed_g(seed_g),
        .ch_we(ch_we), .ch_idx(ch_idx), .ch_data(ch_data),
        .res_idx(res_idx), .res_data(res_data),
        .cyc_idx(cyc_idx), .cyc_round(cyc_round), .cyc_total(cyc_total), .cyc_gen(cyc_gen), .cyc_wall(cyc_wall),
        .mem_we(mem_we), .mem_re(mem_re), .mem_sel(mem_sel), .mem_addr(mem_addr), .mem_wdata(mem_wdata),
        .mem_rdata(mem_rdata), .mem_rvalid(mem_rvalid),
        .st_busy(st_busy), .st_done(st_done), .st_error(st_error), .st_wait(st_wait),
        .st_gen_done(st_gen_done), .st_round(st_round));

    reg [127:0] g_ch  [0:15];
    reg [127:0] g_msg [0:47];
    reg [127:0] g_fin [0:15];
    reg [127:0] g_f   [0:(1<<LOG_NMAX)-1];
    reg [127:0] g_g   [0:(1<<LOG_NMAX)-1];

    integer n_pass = 0, n_fail = 0;

    // ------------------------------------------------------------ host-side primitives
    task automatic pulse_start();
        @(negedge clk); cmd_start = 1; @(negedge clk); cmd_start = 0;
    endtask
    task automatic pulse_gen();
        @(negedge clk); cmd_gen = 1; @(negedge clk); cmd_gen = 0;
    endtask
    task automatic soft_reset();
        @(negedge clk); cmd_soft = 1; @(negedge clk); cmd_soft = 0;
        repeat (4) @(negedge clk);
    endtask
    task automatic write_ch(input integer idx, input [127:0] d);
        @(negedge clk); ch_idx = idx; ch_data = d; ch_we = 1; @(negedge clk); ch_we = 0;
    endtask
    task automatic read_res(input integer idx, output [127:0] d);
        @(negedge clk); res_idx = idx; repeat (3) @(negedge clk); d = res_data;
    endtask
    task automatic read_cyc(input integer idx, output [31:0] d);
        @(negedge clk); cyc_idx = idx; repeat (3) @(negedge clk); d = cyc_round;
    endtask
    task automatic mem_write(input sel, input integer addr, input [127:0] d);
        @(negedge clk); mem_sel = sel; mem_addr = addr; mem_wdata = d; mem_we = 1;
        @(negedge clk); mem_we = 0;
    endtask
    task automatic mem_read(input sel, input integer addr, output [127:0] d);
        integer t;
        @(negedge clk); mem_sel = sel; mem_addr = addr; mem_re = 1;
        @(negedge clk); mem_re = 0;
        t = 0;
        while (!mem_rvalid && t < 64) begin @(negedge clk); t++; end
        @(negedge clk);
        d = mem_rdata;
    endtask
    task automatic wait_flag(input integer which, output integer ok);   // 0 done, 1 wait, 2 gen_done
        integer t;
        t = 0;
        ok = 0;
        while (t < MAXC) begin
            @(negedge clk);
            if ((which == 0 && st_done) || (which == 1 && st_wait) || (which == 2 && st_gen_done)) begin
                ok = 1;
                t = MAXC;
            end
            t++;
        end
    endtask

    // ------------------------------------------------------------ one test case
    task automatic run_case(input string gold, input string name, input integer n, input integer mem,
                            input integer step, input [63:0] sf, input [63:0] sg, input integer R);
        integer errors, i, k, l, ok, N, pairs;
        reg [127:0] d;
        reg [31:0]  cyc;
        string kind, fn;
        errors = 0;
        N = 1 << n;
        for (i = 0; i < 48; i++) g_msg[i] = 'x;
        for (i = 0; i < 16; i++) begin g_ch[i] = 'x; g_fin[i] = 'x; end
        fn = $sformatf("%s/%s_ch.hex", gold, name);  $readmemh(fn, g_ch);
        fn = $sformatf("%s/%s_msg.hex", gold, name); $readmemh(fn, g_msg);
        fn = $sformatf("%s/%s_fin.hex", gold, name); $readmemh(fn, g_fin);
        fn = $sformatf("%s/%s_f.hex", gold, name);   $readmemh(fn, g_f);
        fn = $sformatf("%s/%s_g.hex", gold, name);   $readmemh(fn, g_g);
        // guard against vacuous comparisons: every expected word must be known
        for (i = 0; i < 3 * R; i++) if ($isunknown(g_msg[i])) begin errors++; $display("ERR %s: golden msg word %0d not loaded", name, i); end
        for (i = 0; i < R; i++) if ($isunknown(g_ch[i])) begin errors++; $display("ERR %s: golden challenge %0d not loaded", name, i); end
        for (i = 0; i < 2 * LANES; i++) if ($isunknown(g_fin[i])) begin errors++; $display("ERR %s: golden final %0d not loaded", name, i); end
        for (i = 0; i < N; i++) if ($isunknown(g_f[i]) || $isunknown(g_g[i])) begin errors++; $display("ERR %s: golden table word %0d not loaded", name, i); i = N; end

        soft_reset();
        @(negedge clk);
        cfg_logn = n; seed_f = sf; seed_g = sg; cfg_step = step;
        repeat (2) @(negedge clk);

        if (mem) begin
            for (i = 0; i < N; i++) begin
                mem_write(0, i, g_f[i]);
                mem_write(1, i, g_g[i]);
            end
        end else begin
            pulse_gen();
            wait_flag(2, ok);
            if (!ok || st_error) begin errors++; $display("ERR %s: GEN did not complete (err=%0d)", name, st_error); end
        end
        // read-back spot checks of the table placement (lane/bank/address mapping)
        for (k = 0; k < 4; k++) begin
            i = (k == 0) ? 0 : (k == 1) ? N - 1 : (k == 2) ? N / 2 : (N > 2 ? 3 : 1);
            mem_read(0, i, d);
            if (d !== g_f[i]) begin errors++; $display("ERR %s: readback f[%0d] got %032h exp %032h", name, i, d, g_f[i]); end
            mem_read(1, i, d);
            if (d !== g_g[i]) begin errors++; $display("ERR %s: readback g[%0d] got %032h exp %032h", name, i, d, g_g[i]); end
        end

        if (!step) begin
            for (i = 0; i < R; i++) write_ch(i, g_ch[i]);
            pulse_start();
            wait_flag(0, ok);
            if (!ok) begin errors++; $display("ERR %s: AUTO job timeout", name); end
        end else begin
            // the testbench plays the host: read message i, then supply challenge r_i
            pulse_start();
            for (i = 0; i < R; i++) begin
                wait_flag(1, ok);
                if (!ok) begin errors++; $display("ERR %s: STEP wait timeout at round %0d", name, i); i = R; end
                else begin
                    if (st_round !== i) begin errors++; $display("ERR %s: ROUND=%0d expected %0d", name, st_round, i); end
                    for (k = 0; k < 3; k++) begin
                        read_res(3 * i + k, d);
                        if (d !== g_msg[3 * i + k]) begin errors++; $display("ERR %s: STEP msg[%0d][%0d] got %032h exp %032h", name, i, k, d, g_msg[3*i+k]); end
                    end
                    write_ch(i, g_ch[i]);
                    pulse_start();
                end
            end
            wait_flag(0, ok);
            if (!ok) begin errors++; $display("ERR %s: STEP job timeout", name); end
        end
        if (st_error) begin errors++; $display("ERR %s: STATUS.ERROR set", name); end
        if (st_round !== R) begin errors++; $display("ERR %s: final ROUND=%0d expected %0d", name, st_round, R); end

        // every round message and every lane final
        for (i = 0; i < R; i++)
            for (k = 0; k < 3; k++) begin
                read_res(3 * i + k, d);
                if (d !== g_msg[3 * i + k]) begin errors++; $display("ERR %s: msg[%0d][%0d] got %032h exp %032h", name, i, k, d, g_msg[3*i+k]); end
            end
        for (l = 0; l < LANES; l++) begin
            read_res(48 + 2 * l, d);
            if (d !== g_fin[2 * l]) begin errors++; $display("ERR %s: lane %0d final f got %032h exp %032h", name, l, d, g_fin[2*l]); end
            read_res(48 + 2 * l + 1, d);
            if (d !== g_fin[2 * l + 1]) begin errors++; $display("ERR %s: lane %0d final g got %032h exp %032h", name, l, d, g_fin[2*l+1]); end
        end

        // cycle counters
        for (i = 0; i <= R; i++) begin
            read_cyc(i, cyc);
            pairs = (i == 0) ? (1 << (n - LG_L - 1)) : (1 << (n - LG_L - i));
            kind = (i == 0) ? "eval0" : (i == R) ? "fold" : "fused";
            $display("CYC name=%s L=%0d n=%0d pass=%0d kind=%s pairs=%0d cycles=%0d", name, LANES, n, i, kind, pairs, cyc);
        end
        $display("CYCT name=%s L=%0d n=%0d total=%0d wall=%0d gen=%0d", name, LANES, n, cyc_total, cyc_wall, mem ? 0 : cyc_gen);
        $display("CASE name=%s L=%0d n=%0d src=%s mode=%s R=%0d result=%s errors=%0d",
                 name, LANES, n, mem ? "mem" : "gen", step ? "step" : "auto", R, errors == 0 ? "PASS" : "FAIL", errors);
        if (errors == 0) n_pass++; else n_fail++;
    endtask

    // ------------------------------------------------------------ main
    initial begin : main
        string gold, cases, name;
        integer fd, code, n, mem, step, R;
        reg [63:0] sf, sg;
        if (!$value$plusargs("GOLD=%s", gold)) gold = "../golden";
        if (!$value$plusargs("CASES=%s", cases)) cases = $sformatf("%s/cases_L%0d.txt", gold, LANES);
        $display("TB tb_core LANES=%0d LOG_NMAX=%0d DRAIN_ADJ=%0d cases=%s", LANES, LOG_NMAX, DRAIN_ADJ, cases);
        repeat (20) @(negedge clk);
        rst = 0;
        repeat (100) @(negedge clk);
        fd = $fopen(cases, "r");
        if (fd == 0) begin $display("ERR cannot open %s", cases); $finish; end
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%s %d %d %d %h %h %d\n", name, n, mem, step, sf, sg, R);
            if (code == 7) run_case(gold, name, n, mem, step, sf, sg, R);
        end
        $fclose(fd);
        // invalid configuration must raise ERROR and not start
        @(negedge clk); cfg_logn = LOG_NMAX + 1;
        pulse_start();
        repeat (4) @(negedge clk);
        if (st_error && !st_busy) $display("CHECK invalid_logn result=PASS");
        else begin $display("CHECK invalid_logn result=FAIL"); n_fail++; end
        $display("SUMMARY L=%0d pass=%0d fail=%0d", LANES, n_pass, n_fail);
        $finish;
    end
endmodule
