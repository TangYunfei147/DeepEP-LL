# 撤销历史 QP 扩容后的 SDK 迁移

2026-10-16：两条 kernel 分支撤销自行加入的 SHMEM ROCE QP 扩容。锁定的 SHMEM revision 不升级，HNS_1825 后端及机内 MTE/跨机 RDMA 协议保留。LL 的 72 核控制容量不是 SDK QP 容量，不随此迁移删除。

## 先检查并保留本地修改

在目标机仓库根目录执行，下列操作由用户在目标环境进行。先停止使用当前 SDK 的所有任务；更新源码不会自动删除或重建正在使用的 SDK。

```bash
git status --short
git branch --show-current
git -C third-party/shmem status --short
# 确认当前分支为 kernel/ll-ndr-hns1825 或 kernel/ll-configurable-groups。
git pull --ff-only
```

若当前工作区有其他任务修改，保留并检查更新是否冲突；不要 reset、clean 或覆盖文件。Git 无法快进或与本地改动冲突时先解决具体冲突，不强行更新。全 rank 使用同一个分支提交。

## 已有子仓的三处精确撤销

这部分仅记录已删除历史补丁的逆向迁移，不是当前构建步骤中的自动补丁机制。先查看 `git -C third-party/shmem diff`，只在确认为旧脚本生成的表达式时逐处替换。下表路径均相对 `third-party/shmem`。

| 文件 | 旧生成表达式 | 恢复为锁定 SDK 原生表达式 |
|---|---|---|
| `src/host/init/shmem_init.cpp` | `if (!is_valid_qp_num(qp_num, 72U)) { // DeepEP: one ROCE QP per actual AIV` | `if (!is_valid_qp_num(qp_num, ACLSHMEM_MAX_QP_NUM)) {` |
| `src/host/transport/device_rdma/device_rdma_transport_manager_v2.cpp` | `if (qpNum_ == 0 || qpNum_ > 72U) { // DeepEP ROCE-only capacity` | `if (qpNum_ == 0 || qpNum_ > ACLSHMEM_MAX_QP_NUM) {` |
| `src/host/utils/exception/shmem_exception_rdma_dump.cpp` | `info.count > 72U /* DeepEP ROCE-only capacity */` | `info.count > ACLSHMEM_MAX_QP_NUM` |

第一处只针对其下一行为 `SHM_LOG_ERROR("invalid ROCE qp num: " << qp_num);` 的 ROCE 校验，不能修改 UDMA 校验。原生表达式已存在时无需修改。若表达式不存在、出现多次、上下文不匹配或与其他改动重叠，停止并检查具体差异；不做模糊替换。只编辑这三处对应文本，保留其他修改，禁止整文件恢复。

```bash
git -C third-party/shmem diff --check
git -C third-party/shmem diff
```

确认三处扩容已撤销、其他改动保持原样。其他源码修改不会再获得隐式豁免；默认构建会拒绝，只有明确审查后才能使用既有的 --allow-local-changes 流程并记录真实来源。不要为构建通过而删除其他修改或篡改 manifest，也不能用该选项继续保留已撤销的扩容。

此次本地检查涉及的三个 checkout 均未初始化 SHMEM 子仓，没有需要回撤的本地三处修改；这不代表目标机不存在旧补丁。

## 显式重建 SDK 和 wheel

旧扩容 SDK 不可复用，即使子仓源码已恢复，旧二进制也不会因此变成原生 SDK。先确认任务已经停止，然后在仓库根目录执行：

```bash
# 使用目标机实际的 CANN 环境位置。
source /path/to/ascend-toolkit/set_env.sh
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
# 仅在上述源码检查完成、使用者已停止后执行。
python scripts/build_shmem.py --rebuild
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
# 诊断采样选择 ON，正式未插桩性能基线选择 OFF；各 rank 一致。
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
git rev-parse HEAD
```

NDR 默认构建 HNS_1825，泛化默认构建 MTE；不再应用 QP 扩容。后续原生 SDK 身份完全匹配时，`python scripts/build_shmem.py` 可以安全复用；未知或损坏产物不会获准复用。不要删除旧产物、修改身份字段或重新登记旧 SDK 来绕过检查。各 rank 必须使用匹配的 SDK、wheel、核心 ABI 与 timer schema。

## 可运行范围与验证边界

锁定提交 `4dd80e2129da42bd773982c2bb3758af1ed0bba8` 的 `include/host/shmem_host_def.h:34` 定义 `ACLSHMEM_MAX_QP_NUM=56`（每 peer）；`src/host/init/shmem_init.cpp:221-224,722` 校验 ROCE QP 为1..56，transport manager 第269行及异常dump第163行使用同一上限。

跨机 RDMA 每实际 AIV 独占 QP，因此 LL 的合法实际 AIV 数为16..56偶数；实际核数超过56时必须在初始化前明确报错，不截断、不共享 QP，也不自动减少启动核数。EP≤16 的 NDR 机内 MTE及泛化 MTE 不因该 QP 限制降低原有 LL 核控制容量。RDMA 原语也受相同原生限制，原生 SDK 的资源数量约束不等于目标 HNS provider 已通过验收。

本次维护只进行本地 CPU 检查与源码审计，没有 SSH、CANN 编译或设备验证。目标机重建、正确性与性能验收仍由用户执行；旧的设备验证记录仅说明当时版本，不作为撤销补丁后的验收结果。
