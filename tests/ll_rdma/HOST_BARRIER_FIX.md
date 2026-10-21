# NDR Host barrier：只修改 DeepEP，SDK 保持原样

此方案取代已撤回的 `4a2d736` SDK overlay 方案。**禁止对 SHMEM 应用补丁；
SHMEM锁定revision、源码、构建脚本和SDK身份规则保持原样。无需重建原SDK。**

锁定SDK的Host barrier在EP32上直接轮询跨机peer地址，不适用于本作业的
MTE | ROCE混合拓扑。DeepEP现在自己编译并启动
`deepep_ndr_host_barrier_kernel`，不调用或修改SDK的跨机Host barrier。

实现：`csrc/kernels/runtime/host_barrier.{cpp,h}`，编入DeepEP共享Device对象，
timer ON/OFF复用同一对象。Session初始化/关闭在独立stream执行并检查完成状态，
错误不再被SDK的void barrier吞掉。RDMA测试原语也使用同一DeepEP实现。

- 全MTE team沿用SDK设备同步算法；含ROCE peer时使用binary dissemination。
- 仅AIV0推进generation和使用SDK UB/QP，两端保留核内同步。
- 机内使用MTE可达GM写，跨机只调用原SDK的QP put/quiet，无RDMA get/atomic。
- 所有轮询均读本地同步槽，NIC可写视图绕过L2。put完成后才能复用generation源。
- 使用显式team映射，每轮独立缓存行；排空ROCE peer已有QP写入后发布到达。
- 不支持的peer、generation溢出和槽位越界明确trap；原SDK CQ异常/超时处理保留。

Host barrier必须与本rank通信kernel串行。没有修改dispatch/combine算法、ABI5、
核数、G策略或SDK原生每peer 56 QP限制。SDK自身其他barrier接口仍保持原行为。

## 部署

每台机器在原测试环境和仓库根目录执行，保持原先通过校验的未修改SDK：

```bash
git pull --ff-only
source tests/ll_rdma/env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF  # 打点时全rank统一ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_rdma/build_smoke.sh
```

不要执行上一版文档的SDK `--rebuild` 或overlay步骤。
若已经执行过已撤回版本的SDK修改，应先核实SDK状态；不能把被改SDK当作原SDK。
所有rank更新完整wheel；新的Host依赖新的DeepEP Device导出符号，禁止Host/Device混装。
使用新的run-name，先执行跨机双rank原语及两机各16卡EP32 B1正确性，再恢复性能矩阵。

## 验证范围

CPU模型覆盖延迟消息、连续generation、非连续team映射和EP2..256。
提供直接编译实际DeepEP helper的C++线程模拟及Session同步失败故障注入测试。
本地无C++/CANN/NPU，相关编译/设备项未验收；CPU模型不能替代目标机验证。
