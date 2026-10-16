# 逐次事件分支验证

本地CPU阶段/事件/报告/dispatch选择回归：152 passed，10 skipped（没有本地C++编译器）。覆盖逐次metadata/write顺序、单核轨道、事件计数与容量、错误/未闭合/乱序/旧格式拒绝、紧凑导出、实际事件计算的带宽、不同shape的离线报告。C++用例在有G++的环境执行真实DeviceTimer及Host分发夹具。

可复现命令：

```bash
python -m pytest tests/ll_stage_timer tests/ll_rdma/test_dispatch_variants.py tests/ll_rdma/test_transport.py tests/elastic/test_ll_st_contract.py -q
```

A3仅允许编译与CPU检查，不运行设备。编译入口compile_validation.sh使用当前显式CANN环境及SHMEM_ROOT指向的已识别SDK，构建timer ON完整wheel和timer OFF设备对象，不安装/重建SDK，不安装wheel，不修改驱动或系统依赖。日志及产物位于当前工程内。普通任务执行与依赖等待互斥，kernel单独概览；等待前后使用同一task ID，没有旧父阶段插桩。

目前尚未完成本次A3编译；不得将CPU模型通过标为CANN或设备验证通过。没有目标机正确性或性能结论。所有同步模式均有插桩开销，默认mega主动排空普通阶段；正式基线使用timer OFF。