# 逐次事件分支验证

本地CPU阶段/事件/报告/dispatch选择回归：152 passed，10 skipped（没有本地C++编译器）。覆盖逐次metadata/write顺序、单核轨道、事件计数与容量、错误/未闭合/乱序/旧格式拒绝、紧凑导出、实际事件计算的带宽、不同shape的离线报告。C++用例在有G++的环境执行真实DeviceTimer及Host分发夹具。

可复现命令：

```bash
python -m pytest tests/ll_stage_timer tests/ll_rdma/test_dispatch_variants.py tests/ll_rdma/test_transport.py tests/elastic/test_ll_st_contract.py -q
```

A3仅允许编译与CPU检查，不运行设备。编译入口compile_validation.sh使用当前显式CANN环境及SHMEM_ROOT指向的已识别SDK，构建timer ON完整wheel和timer OFF设备对象，不安装/重建SDK，不安装wheel，不修改驱动或系统依赖。日志及产物位于当前工程内。普通任务执行与依赖等待互斥，kernel单独概览；等待前后使用同一task ID，没有旧父阶段插桩。

A3源码提交471f8b90ee823eb1857791cead5884d011560b73：CPU/G++ 115项全部通过，包括等待切分/同task ID续记、容量错误、实际Host分发夹具和异步通信协议模型。timer ON完整wheel构建通过；timer OFF独立设备构建21个对象通过；smoke Host源文件G++语法编译通过。该验证对应最终相同的C++源码；后续仅补充文档和Python错误消息。没有初始化或运行NPU。没有目标机正确性或性能结论。所有同步模式均有插桩开销，默认mega主动排空普通阶段；正式基线使用timer OFF。
A3工作区：/home/tyf/Q4_remote_test/DeepEP-LL-event-timer，独立Git worktree，使用Git bundle取得同一分支。复用原cold工程内已识别的原生HNS_1825 SDK，无source_patches，未重建SDK或安装wheel。生产架构保持Ascend950，未为A3改目标。

日志：.logs/event-validation/{cpu,wheel,off}.log；产物：.artifacts/event-wheels/ascend_deepep-0.1.0.dev5-cp314-cp314-linux_aarch64.whl。此wheel只对应A3 Python环境，不作为用户目标环境的通用安装包。目标机仍通过Git更新并重建匹配wheel。