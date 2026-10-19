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

## 2026-10-19 Start/End配对修复

目标EP16的dispatch诊断导出报告core 0 flags=6（2=意外嵌套Start，4=不匹配End），不是容量溢出。首跳包FillTail在send_tail活动期间调用带send_route计时的CalTokenSendExpertCnt，普通任务发生嵌套。修复在q计算前结束send_tail，计算后开始下一段尾部写入；保留严格检查、schema4、ABI5、全部容量和通信操作。

已检查dispatch发送/接收/转发和combine直接回传、partial归约、源端输出及共用Packet路径。combine的依赖等待由DeviceTimer暂停执行片段，结束后同task ID续记，无需放宽嵌套限制。新增CPU/C++回归直接提取生产FillTail，覆盖direct/first-hop、非去重、多个成员和三种模式；另验证combine/relay的连续依赖等待，以及原非法send_tail→send_route仍严格报告flags=6。

本地相关完整回归163 passed（含真实DeviceTimer的C++执行、通信协议模型、Host分发和离线报告），无跳过。Windows便携Zig C++仅用于CPU夹具；项目根使用quote include避免VERSION遮蔽标准库version头。A3 SSH连接被关闭，本轮未完成CANN构建或NPU验证；此前A3编译记录不等同于本次修复的设备验收。更新后需重建timer ON wheel并用新run-name重新采集，失败旧run没有可分析的完整事件数据。
