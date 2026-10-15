# HNS_1825 NPU Direct RDMA LL 验收

分支：`kernel/ll-configurable-groups`。通信路径、真实 SDK API、顺序和剩余风险见 [AUDIT.md](AUDIT.md)。本目录覆盖生产 dispatch/combine 的 RDMA 适配；双 rank 程序只测传输原语。正式统计协议入口见 [PROFILING.md](../ll_probe/PROFILING.md)。

## 获取与构建

每台目标主机通过 Git 获取同一提交，后续只通过 Git 更新：

```bash
git clone --branch kernel/ll-configurable-groups https://github.com/TangYunfei147/DeepEP-LL.git
cd DeepEP-LL
git submodule update --init --recursive --depth 1
python3 -m venv --system-site-packages .venv-ll-rdma
source .venv-ll-rdma/bin/activate
# 先加载本机 CANN，或显式设置 CANN_ENV；相对 CANN_ENV 从仓库根解析。
# 如 provider 不在 CANN 的库路径中，先设置 DEEPEP_RDMA_PROVIDER_PATH。
source tests/ll_rdma/env.sh
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --rdma-backend HNS_1825
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_RDMA=ON DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_rdma/build_smoke.sh
```

若 SDK build/install 已存在，先停止使用该 SDK 的任务，再显式加 `--rebuild`；一致的 SDK 可用 `--if-needed --rdma-backend HNS_1825` 复用。只使用 `third-party/shmem/install/shmem`。ROCE 72-QP 补丁会在子仓产生预期的三处工作区修改，脚本保留用户其他修改并拒绝无授权覆盖；无需另建子仓或把 SDK 产物提交 Git。

`env.sh` 把日志、缓存和临时文件放到当前工程，并添加 SDK 库路径，不默认选卡。目标 runner 必须显式 `--devices`。Python runner 同样固定工程内可写路径，所有相对配置/输出均基于脚本定位的工程根目录。

## 环境前检

```bash
python tests/ll_rdma/preflight.py
# 可补充本机已确认的 PCI BDF 和平台拓扑文件，不能用它们猜测网卡型号：
python tests/ll_rdma/preflight.py --npu-bdf "$NPU_BDF" --nic-bdf "$NIC_BDF" --topology "$TOPOLOGY_FILE"
```

保留 `.logs/ll-rdma-preflight.json`。检查 1825 驱动/provider 与 CANN 匹配、Hcomm 必需符号、RDMA 端口状态、NPU/NIC PCI 祖先和 NUMA、平台每 NPU 的 ROCE IP 映射。`inventory_complete_requires_smoke` 仅表示基本前检完成；HBM 注册、通道建立、实际亲和性及有序可见性仍需下面的原语测试。前检失败不启动 kernel，不转 MTE。

## 两个真实 rank 的原语测试

同机两卡，以下 IP 是 rank0 主机可达的 bootstrap 地址，端口自行选择空闲值；设备列表是实际选用的两张物理卡：

```bash
python tests/ll_rdma/run_smoke.py --devices 0,1 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
```

跨主机各一卡，在两台主机分别执行，使用相同 endpoint：

```bash
# 主机 0
python tests/ll_rdma/run_smoke.py --rank 0 --devices 0 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
# 主机 1
python tests/ll_rdma/run_smoke.py --rank 1 --devices 0 --endpoint "tcp://$RANK0_IP:29672" --rounds 20
```

默认按当前设备可用 AIV 数创建 QP。`--cores` 可明确测试不超过当前可用数量的 16..72 偶数；它只控制这个原语程序，不改变生产 LL 的实际核数查询。每个核有独占 QP；交换 512/1536/15872B 包，检查 put、get、计数先行发布和 ready，逐轮改变数据并重复复用缓冲。两 rank 都必须输出 PASS，所有核状态为 0，无 CQ error/trap/timeout。原始日志在 `.logs/rdma-smoke-*`。跨机时两台机器都要检查成功，单台 PASS 不代表对端整体完成。

A5 仅允许 `--validation`，强制四个可见性变量为 `0,1`，最多两个真实设备 rank：

```bash
python tests/ll_rdma/run_smoke.py --validation --endpoint "tcp://$RANK0_IP:29672"
```

当前 A5 前检未枚举到 RDMA 设备，原语实际收发未执行；不能把编译或 MTE 结果记为 RDMA 通过。

## 目标机完整 LL 正确性

编辑仓内 `target16.json`/`target32.json` 的 `master_addr`，必要时修改空闲端口。每机 16 张真实卡；EP32 使用两机，每机分别传 `--node-rank 0`、`--node-rank 1`。所有 rank 安装相同 Git 提交、核心协议 3、逻辑组参数、RDMA 编译选项和 SDK。

```bash
# 无设备启动的计划，可从任意 cwd 调用脚本。
python tests/ll_rdma/run_ll.py --backend rdma --devices 0,1 --validation \
  --config tests/ll_rdma/target16.json --plan

# 目标单机 EP16；这是目标机命令，不在 A5 执行。
python tests/ll_rdma/run_ll.py --backend rdma --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --config tests/ll_rdma/target16.json --node-rank 0 --skip-perf-test --route normal
python tests/ll_rdma/run_ll.py --backend rdma --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --config tests/ll_rdma/target16.json --node-rank 0 --skip-perf-test --route hotspot

# 目标双机 EP32：两机均执行，NODE_RANK 分别设置为 0 / 1。
python tests/ll_rdma/run_ll.py --backend rdma --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --config tests/ll_rdma/target32.json --node-rank "$NODE_RANK" --skip-perf-test --route normal
# normal 全 rank 通过后，两机用 --route hotspot 再执行一次。
```

矩阵覆盖 BF16/FP8、H=256/7168 的尾块、B=1/7/65、正常及热点路由；同一 Buffer 连续三次 dispatch 使用不同路由，再对匹配 handle 连续三次 combine，覆盖双窗口复用。也可按公开约束扩大矩阵。CPU 路由/布局回归覆盖逻辑组 1..16 和实际核数 16..72；它不替代目标机上所选 G/N 的数值验收。

通过判据：全部 rank 正确性为 passed；dispatch 行数、来源、专家计数、scale、handle 元数据一致；combine 与独立 CPU 参考满足既有 BF16/FP8 容差；无自动 HT/MTE 回退；无 CQ 错误、超时、卡住或复用后结果变化。失败后保留原始输出，停止该轮，不自动重试正确性或采样。

公开 API 使用方式不变。自行创建 Buffer 时，在启动进程前选择 `DEEPEP_LL_BACKEND=rdma` 并设置 `ElasticBuffer(..., use_1g_huge_pages=False)`；不支持的 LL 请求会显式失败。默认 `mte` 保留既有后端。

## 性能采样与回传

正确性全通过后，移除 `--skip-perf-test` 运行相同目标配置。严格保留 10 warmup、100 capture、每次采集前 Exp＋scalar all_reduce、每 kernel/task 独立取最后 50 条。默认 Exp 为 118260×30720 FP32（约 13.53GiB/卡）；不要把 A5 的缩小负载用于目标性能结论。严格基线使用 stage timer OFF。

用同一目标配置、page policy（示例均为普通页）、G/N、dtype、路由、频率策略和采样协议，将 `--backend rdma` 改为 `--backend mte` 获取对照。MTE 只在目标互联确实能建立原映射时可比；MTE unsupported/失败不能当作加速证据。首版 RDMA 有逐包完成等待，可能更慢，必须如实报告。

统计的主指标是 profiler 原始 kernel/task 时长。trace 中 API/Event 记录另存；本入口没有用 Event 或完整 API 墙钟替代 kernel 指标，也没有宣称测得这些独立指标。需要 API/Event 对照时单独采集、明确口径，不把它们合并到 kernel 分布。

回传：所有主机的 Git SHA、SDK `deepep-sdk.json`、构建开关/核心协议/G/N、前检 JSON、同机和跨机原语日志、完整 LL `cases.json`/summary/每 rank progress 与 loaded-library hashes、原始 profiler task CSV/trace/样本 JSON、失败日志及实际命令。这些原始数据和生成产物保留在工程日志目录，不提交 Git。

CPU 回归（先 source 工程环境，临时文件会位于仓内）：

```bash
python -m pytest -q tests/ll_rdma tests/runtime tests/api \
  tests/elastic/test_low_latency_host.py tests/elastic/test_ll_core_config.py \
  tests/elastic/test_ll16_protocol.py tests/elastic/test_ll_st_contract.py tests/ll_stage_timer/test_timer.py
```
