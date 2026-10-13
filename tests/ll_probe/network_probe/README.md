# LL 建域与网络层诊断

直接编译并调用当前分支 `csrc/runtime/session.cpp` 的 `Session`，沿用 ElasticBuffer 的
MTE|UDMA、32 QP、shared jetty、生产 peer mask、默认 8 GiB workspace 和 1 GiB huge page 设置。
不修改生产源码，不运行完整 dispatch/combine，不计时。默认策略不强制 MESH/CLOS。

**输出边界**：查询的是初始化后每个有效 `(peer, qp)` 的 **UDMA 已选网络层**，不是 LL 的
MTE 物理路径，也不发送 UDMA payload。另用 LL 风格 peer 映射 + DataCopy 做 0↔指定 rank
双向 32B 接收校验，确认 MTE 可访问。MTE 的实际网络层字段始终为 null；现有公开查询接口
不能证明它走 MESH 还是 CLOS。物理路径最终需目标端口流量计数等证据。

## 配置与 rank

`config.yaml` 只包含 IPv4 字符串列表，不需要 PyYAML，不支持变量展开或任意 YAML 对象：

```yaml
- 10.0.0.10
- 10.0.0.11
```

IP 是 bootstrap 可达地址，不是指定数据面网络口。列表第一项为 bootstrap 服务端。
默认每机 16 卡：**全局 rank = YAML 主机序号 × 16 + 本地卡号**。
例如 `--peer 19` 表示第二台机器的本地卡 3，与第一台机器的卡 0 比较。

可修改的“宏式常量”在 `defaults.py`：`DEVICES_PER_HOST = 16`、`DEFAULT_PORT = 18679`、
`DEFAULT_TIMEOUT = 180`、`LL_WORKSPACE_BYTES = 8 << 30`。
也可通过 `--devices-per-host` 覆盖每机卡数；全体节点必须一致。生产 LL 域只允许 EP16/32/64/128。
修改 workspace 会偏离 LL 默认配置；正常诊断保留 8 GiB。`--no-huge-pages` 是显式偏离
ElasticBuffer 默认的兼容选项，不自动回退。

## 目标机步骤（由用户执行）

所有机器 clone 同一分支/提交及子模块，均位于授权根目录内：

```bash
git clone --branch experiments/ll-four-layer-probes --recurse-submodules \
  https://github.com/TangYunfei147/DeepEP-LL.git /home/tyf/Q4_remote_test/DeepEP-LL
cd /home/tyf/Q4_remote_test/DeepEP-LL
```

后续更新使用 `git pull --ff-only` 和 `git submodule update --init --recursive`。
在每台机器编辑相同 `config.yaml`。已有仓库直接切到交付分支，不覆盖已有工作。
使用已有 CANN 9.3.0 环境和锁定版本 SHMEM 库（不安装系统依赖）：

```bash
export PROBE_ROOT=/home/tyf/Q4_remote_test
export CANN_ENV="$PROBE_ROOT/env_1009/cann-9.3.0/set_env.sh" # 按本机现有环境调整
# 若尚无探针 SHMEM 库，先按 tests/ll_probe/USER_RUN.md 构建锁定库。
# build_shmem_probe.sh 使用上述默认环境路径；环境不同请按指南设置构建环境。
export SHMEM_LIB_DIR="$PROBE_ROOT/DeepEP-LL/tests/ll_probe/build/shmem/lib"
bash tests/ll_probe/network_probe/build.sh
python3 tests/ll_probe/network_probe/run.py --config tests/ll_probe/network_probe/config.yaml --peer 19 --plan
```

`--plan` 只检查配置，不启动设备。每台机器分别启动一个本地 launcher，无需机器之间 SSH：

```bash
# 第一台（YAML 索引0）；下面的输出目录必须尚不存在
bash tests/ll_probe/network_probe/run.sh --node-index 0 --peer 19 --target-run \
  --output "$PROBE_ROOT/results/network-case1-node0"
# 第二台（YAML 索引1），在第二台的终端执行
bash tests/ll_probe/network_probe/run.sh --node-index 1 --peer 19 --target-run \
  --output "$PROBE_ROOT/results/network-case1-node1"
```

所有节点应在 timeout 内启动。开放第一项 IP 的指定 bootstrap TCP 端口，数据面网络须事先具备。
`--target-run` 仅用于用户目标机器，显式把可见设备扩展到 `0..DEVICES_PER_HOST-1`。
launcher 不修改网络层策略；保留生产默认环境，记录 `DEEPEP_SHMEM_DENSE_QPS`。
SHMEM 日志默认 INFO；`processes.json` 可关联本次各 rank 的日志 PID。报告包含所用 SHMEM
库的 hash 和可取得的 CMake 配置，特别是 multi-link/relay；不把默认单链路库解释为多链路验证。
某个本地 rank 失败会终止本机其他 rank；其他机器由自己的 SHMEM/launcher 超时退出。
所有日志、缓存、临时文件都在 PROBE_ROOT；结果目录包含源码身份、构建身份、逐rank日志、报告。

收集各节点 report.json 后检查完整性（数据不提交 Git）：

```bash
python3 tests/ll_probe/network_probe/merge.py /path/to/node0/report.json /path/to/node1/report.json
```

成功要求所有节点/所有rank正常退出、UDMA查询成功、双向MTE接收正确；单节点 local_success
不代表整个多机任务成功。每个 route 记录实际 layer；公开 layer0 通常为同域 MESH、layer1 为
CLOS，应同时看 `$PROBE_ROOT/.logs/shmem` 中本次进程的 `Build UDMA channel from resolved route`
日志及 `actualNetLayer/netType/netInstanceId/planeId/localEidIndex/remoteEidIndex`。
报告不推断物理跳数、不把 UDMA QP 建链当作该 QP 已发送数据。

## A5_241 验证专用

仅本机两个真实rank，禁止扩展设备。生成单IP YAML 放在 PROBE_ROOT 内，再运行：

```bash
bash tests/ll_probe/network_probe/run.sh --config /home/tyf/Q4_remote_test/local-network.yaml \
  --validation --devices-per-host 2 --peer 1
```

验证使用 2 MiB workspace、关闭 1 GiB 页，仅验证 Session/查询/小包运行链路；EP2 不是原生 LL
支持的通信域，不会伪装成 EP16，不据此得出目标机链路或性能结论。

2026-10-13 验证记录：A5_241 两个真实rank、2 MiB workspace，生产 Session 创建与释放成功；
两方向 QP0 均查询到 layer0，双向32B MTE接收校验通过。锁定 SHMEM
`4dd80e2129da42bd773982c2bb3758af1ed0bba8`，实验库 multi-link=OFF、relay=OFF。
结果保留于授权目录 `project_control/results/network-validation-003`，不提交原始数据。
多机、16卡/机、8 GiB及1 GiB页的目标验收由用户执行，尚无目标机器选路结论。
