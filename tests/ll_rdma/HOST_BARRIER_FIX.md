# NDR 初始化和销毁：现有 HCCL Host 协调

2026-10-22方案，参考 DJLRY/DeepEP 的
`codex/megamoe-hierarchical-roce@bfb6fe8935df07e52d5768e73426dab03050fbd0`。
SHMEM源码、gitlink、SDK、构建脚本和身份校验规则全部保持原样。
本方案替代此前生产路径的DeepEP设备barrier；不需要重建SDK。

## 生命周期

NDR (`DEEPEP_LL_BACKEND=rdma`) 必须使用已建立的HCCL ProcessGroup，
不新建Gloo、TCPStore或任何通信域。无group/其他backend在SHMEM建域前明确拒绝。

1. 配置一致性检查后，利用同一HCCL组交换Linux boot_id、四个可见性变量和本地设备号。
2. 校验每物理主机恰好16个连续global rank、同主机映射一致及设备不重复。
   缺失host身份、交错rank、每机8卡、单机32rank均拒绝。
3. 创建SHMEM Session及workspace；NDR构造失败保留进程owner，禁止新建runtime。
4. 初始化barrier/公开barrier/销毁均先调用设备级synchronize，等待本卡所有stream。
5. HCCL all_gather_object交换操作名及完成/失败状态；操作顺序不一致或任一rank失败，
   不进入native释放。HCCL自身异常同样失败封锁，不回退。
6. 全rank成功后调用native barrier/close(host_coordinated=True)。NDR native不再启动
   设备barrier。直接native调用不带协调确认会拒绝；该参数是内部契约，不可用它绕过协调。

失败后Python runtime永久封锁，保留owner及尚未释放资源；ensure_alive/barrier/close
均拒绝重入，必须重启作业。native清理失败仍保留已有poisoned保护。
已成功完成的部分清理无法回滚，不声称所有清理异常都能保留完整workspace。

所有rank须按同一顺序调用，不允许其他线程并发提交算子。LL kernel内部的QP quiet
和body/flag顺序保持原样，device完成才能作为Host汇合前提。设备故障后HCCL可能也失败，
此时受原ProcessGroup超时约束，不能保证收齐所有rank的错误描述，但不会当作成功释放。

生产libdeepep_kernels不再编入host_barrier.cpp；该文件仅保留给独立双rank RDMA
原语诊断程序，其设备测试不能替代HCCL生命周期验收。

## 更新与测试

每台机器使用原Python环境及未修改SDK：

```bash
git pull --ff-only
source tests/ll_rdma/env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

全rank必须更新完整wheel，Python/native接口一起更新。不要运行旧SDK overlay，
不要为此执行SDK --rebuild。若曾执行已撤回SDK修改，应先核实SDK状态。

保留原EP32/每机16进程配置，使用新run-name、先B1，加
`--skip-perf-test --trace-ops`。日志将经过SHMEM建域/分配，然后输出
`HCCL barrier completion exchange enter`；生产路径不应出现DeepEP设备barrier kernel。
正确性通过后恢复原B矩阵和性能采样。

## 验证边界

CPU故障注入运行生产Python生命周期方法，覆盖全部流完成顺序、peer失败、HCCL失败、
操作错序、native错误、重复调用封锁和拓扑校验。C++/CANN/NPU未在本机验收。
参考分支也不是目标机已通过证明，仍需两机设备验证。
