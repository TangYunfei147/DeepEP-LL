# 生产LL性能协议

两kernel分支使用相同tests/elastic/test_ep_low_latency.py；统一入口tests/ll_stage_timer/run.py。
默认10预热、100采集、每次Exp+scalar all_reduce，按kernel/task统计最后50条。
内部timer为独立稳定尾段，不混入perf样本。Event/API/task时间不相互替代。
对比指南见../ll_stage_timer/PARITY.md；当前泛化默认G4；显式配置的G16与NDR G4仍可作为用户选择的比较变量。
