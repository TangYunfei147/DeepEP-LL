# NDR 多机初始化 Host barrier 修复

适用分支 `kernel/ll-ndr-hns1825`，锁定 SHMEM
`4dd80e2129da42bd773982c2bb3758af1ed0bba8`。两机各 16 卡为 EP32。

## 问题与修复范围

原 SDK 的 Host `barrier_on_stream_kernel` 对 EP32 选择 v3，直接 Scalar
轮询 `aclshmem_ptr(sync_pool, peer)`；ROCE-only peer 的地址不能直接读取。
这与初始化后的 `507015`、`The address for scalar to access GM is invalid`
及该 kernel 名称一致。SDK 的 void Host barrier 又忽略同步状态，使错误延后暴露。

工程管理一个精确版本的源码 overlay，仅替换 Host barrier translation unit 的调用：

- 全 MTE team 保留原算法；含非 MTE peer 时使用 binary dissemination。
- 仅 AIV0 推进 generation、使用 SDK UB scratch/QP；其余 AIV/AIC 在两端参与核内同步。
- MTE peer 使用可达 GM 写；ROCE peer 使用 QP0 put + quiet，所有等待只读取本地同步槽。
- 每轮一个缓存行槽，team 显式映射支持非连续成员；发布前 drain ROCE peer 的全部 QP。
- 每次 put 完成后才允许复用 generation 源，NIC 可写同步槽的读写视图绕过 L2。
- 非 MTE/ROCE peer、generation 溢出和槽位超限明确 trap。SDK HNS CQ 错误/超时处理保留。
- Session 初始化及关闭使用自有 stream，检查同步错误；关闭 barrier 失败不释放 workspace。

只适用于 Host 串行同步边界；不得与尚在运行的通信 kernel 并发使用同一 QP。
其他 SDK device barrier 接口未修复，不得把它们视为支持跨机。
不改变 LL dispatch/combine、ABI5、G策略、实际AIV数或原生每peer 56 QP限制。

## SDK 身份与部署

补丁保存在主仓库 `scripts/patches/ndr_host_barrier.inc`，
`scripts/build_shmem.py` 在锁定子仓上精确应用，可重复执行。
未知 barrier 修改直接拒绝；其他源码修改仍需要原有显式授权开关。
manifest 的 `source_patches` 记录 `ndr-host-barrier-v1` 和规范化补丁 SHA256；
未打补丁或补丁版本不符的 HNS SDK 不得复用。QP扩容补丁没有恢复。
不提交子仓本地修改或构建产物；Git 获取主仓库即可重现 overlay。

所有节点先停止使用该 SDK 的作业，再在各自工程根目录执行：

```bash
git pull --ff-only
git submodule update --init --recursive --depth 1
source tests/ll_rdma/env.sh
# 激活原测试所用的工程内 Python 环境
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --rebuild
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF  # 打点测试则各节点统一 ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_rdma/build_smoke.sh
python tests/ll_rdma/preflight.py
```

首次修复必须显式重建 SDK，不能只重装 wheel。之后匹配 SDK 自动校验复用。
保持原子仓revision，不更新到上游最新提交；不要伪造manifest绕过校验。

先按 README 执行跨机双rank原语，再执行两机 EP32 的 B1 正确性，
通过后恢复原 B/shape 矩阵和性能采样。使用新的 run-name。
单机 EP16 应继续只启用 MTE。确认所有 rank 同提交、同SDK及同wheel。

## 验证边界

CPU 回归覆盖延迟消息、连续generation、非连续team映射、补丁重复应用、
未知修改拒绝及旧SDK身份拒绝。Host生命周期故障注入覆盖barrier失败不释放内存。
CPU模型不验证RDMA硬件可见性、CANN编译或多机设备正确性。
本地没有CANN/NPU，目标机必须完成上述验收。
