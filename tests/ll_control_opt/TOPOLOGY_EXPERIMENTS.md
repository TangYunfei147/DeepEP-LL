# G4 拓扑、SDK 与接收调度实验

> 当前性能测试改用[纯LL初始化修复指南](LL_ONLY_FIX.md)：
> `commits-topology-mte.json`仅3项新SDK候选，关闭HT构建，使用新前缀
> `b256-topology-mte-001`，共18次ST。本文原4项SDK对照保留为历史，
> 不将旧混合MTE/UDMA初始化样本复用为本轮结果。

本轮以已测的 `c1e4faa` kernel 为基础，分别比较 SHMEM 更新与接收调度。
新清单是 `tests/ll_control_opt/commits-topology.json`；原 `commits.json` 和
`commits-cumulative.json` 保留，已有实验续跑继续使用原清单。

## 拓扑与实验对象

采用用户确认的拓扑：每机16个连续 rank，每4个连续 rank 为一组。
每个 NPU 有3个 FullMesh 端口连接本组另外3个 NPU、4个 Clos 端口用于机内其他组，
以及1个跨机端口。以 rank0 为例，本组其他 rank 是1、2、3；同机其他组是4～15。
rank16起属于下一台机器。

| CPU 端点分类 | 判定 | 本轮 EP16 的含义 |
|---|---|---|
| `self` | source等于destination | 本卡逻辑搬运，不能算网络端口流量 |
| `fullmesh` | 同机、同4-rank组、不同rank | 本组另外3个 NPU 的端点流量 |
| `clos` | 同机、不同4-rank组 | 机内其他组的端点流量 |
| `inter_host` | 不同16-rank物理机 | 单机EP16没有这类payload端点 |

这个分类不能确定某个包走哪一个 Clos 端口，也不能由 QP 编号推断物理端口。
单机 EP16 的数据不能验证跨机端口带宽。当前 LL 使用 MTE；更新 SDK 不会自动将
LL DataCopy 改成 UDMA 多 QP 传输。

| 清单标签 | 来源 | 比较目的 |
|---|---|---|
| `baseline` | `c705381`：新SDK，kernel保持`c1e4faa` | 新SDK下的共同参考 |
| `sdk-old` | `c1e4faa`：旧SDK、相同kernel | 评估SDK及必要Host兼容适配的影响 |
| `buffered-c2` | `8a23867`：新SDK下的独立调度候选 | 与新SDK baseline比较 |
| `immediate-c2` | `0e9db99`：新SDK下的独立调度候选 | 与新SDK baseline比较 |

四项不是依次叠加的优化。以清单的 `comparison_base` 为准；`sdk-old` 的差值是
**旧SDK耗时减新SDK baseline耗时**。它不能被误读为新SDK减旧SDK。
接收调度候选应保持源/目的端点、包数量和字节数；主要改变发起、等待与消费的时序。
CPU字节模型不能替代候选的设备正确性验证。

C2的共同触发条件是：本S核完成FIRST_HOP生成后，每实际提交2个DIRECT包执行一次
有限探测；每次检查16个pending槽，最多处理2个ready FIRST_HOP。
`buffered-c2`先装入原有RX lane，lane满即停止；等**本S核**的payload与source metadata
已提交后才转发/输出，不等待所有核或所有rank，也不增加全局屏障。
`immediate-c2`使用相同触发与选择规则，直接调用已有`ConsumeLoaded`转发/输出。
首次回调还需准备前缀/任务表，V_S依赖也可能等待此前发送端的向量工作，不能称为零等待。
收益由OFF完整ST判断；ON诊断中的`SEND_SUBMIT_END`可辅助解释提交时刻变化，不能代替OFF性能。

## 更新工程与准备环境

以下命令从目标机已有 DeepEP-LL 工程根目录执行。先确认本地修改与已有任务；
Git工作区不干净时不要使用 `reset` 或 `clean` 强行覆盖。

```bash
git status --short
git fetch origin
git switch kernel/ll-g4-control-opt
git pull --ff-only
git submodule update --init --recursive --depth 1 -- third-party/shmem

# 使用已经验证过的项目Python/NPU/CANN环境。
source /home/tyf/0env.sh
python -c 'import sys, torch, torch_npu; print(sys.executable); print(torch.__version__)'

mkdir -p .tmp .cache .logs/shmem .logs/ascend
export TMPDIR="$PWD/.tmp" TMP="$PWD/.tmp" TEMP="$PWD/.tmp"
export XDG_CACHE_HOME="$PWD/.cache" PIP_CACHE_DIR="$PWD/.cache/pip"
export SHMEM_LOG_PATH="$PWD/.logs/shmem"
export ASCEND_WORK_PATH="$PWD/.logs/ascend"
export ASCEND_PROCESS_LOG_PATH="$PWD/.logs/ascend"
export DEEPEP_NPU_ARCH=Ascend950
export CMAKE_BUILD_PARALLEL_LEVEL=4
unset SHMEM_ROOT CATLASS_ROOT
```

直接使用上述已验证的项目环境，不另建空venv并假设它会继承另一个venv中的Torch。
套件自身的候选venv使用`--system-site-packages`；需使用base/Conda安装中已有匹配
Torch/torch_npu的Python。如果依赖只装在外层venv里，应先为候选环境显式安装匹配版本，
不能仅凭外层的import检查判定继承成功。
构建依赖须满足当前锁定版本；若该项目环境不满足，先将对应wheel上传到工程的
`.tmp/build-wheels/`，然后在该项目环境中离线安装：

```bash
python -m pip install --no-index --find-links .tmp/build-wheels -r requirements-build.txt
```

没有上传依赖包时不要照此命令期待联网回退。套件自身使用
`--no-index --no-deps`，不会自动下载Python依赖。网络受限时，工程和依赖通过
Git bundle/文件上传准备。新旧SHMEM须分别准备，不能把单一pin的`--offline-deps`
bundle统一传给这个包含两版SDK的四项套件；旧版的单项准备方式见下节。

## 新 SDK 与旧缓存隔离

本轮将 SHMEM 从 `4dd80e2129da42bd773982c2bb3758af1ed0bba8` 更新至官方master的
`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`。API适配、SDK校验和已知变化见
[SHMEM_MAINLINE_20261025.md](../../docs/SHMEM_MAINLINE_20261025.md)。

如果当前工程没有旧 `build/install`，或者已经有身份匹配的新 SDK：

```bash
python scripts/build_shmem.py
```

若当前工程还安装着旧 SDK，普通命令会拒绝静默替换。在确认没有任务使用**这个工程的**
SDK后，可显式重建：

```bash
python scripts/build_shmem.py --rebuild
```

本分支的显式重建将原项目SDK构建/安装目录存档到
`.cache/shmem-rebuild/<unique-id>/`。这不是对所有缓存执行重建。
不要修改或删除 `.tmp/ll-control-opt/source-s30`、旧依赖身份的其他source目录、
历史wheel及原始结果；旧SDK候选可能仍需要它们。
也可在新的工程内准备新 SDK，避免替换仍有使用者的安装。

套件根据所选提交的依赖锁与Git pins选择
`.tmp/ll-control-opt/source-s30-deps-<dependency-id>`。
旧 `source-s30` 只有身份匹配时才复用；旧、新 SDK 不共用同一个可变安装目录。
套件每次构建仍执行标准 SDK 校验及真实 CMake/depfile 检查，**不会自动添加
`--rebuild`，也不会将已知旧版本 SDK 复制到新版本缓存中**。

已有旧suite缓存时优先保留并复用，无需重复下面的准备。
全新工程没有旧SDK/cache时，可先独立准备旧版。以下创建步骤**仅适用于
`source-s30`不存在时**；存在时不要覆盖它：

```bash
OLD_REV=c1e4faaf32806da84f8818e1ac3eb34864b7958d
OLD_SOURCE=.tmp/ll-control-opt/source-s30
if [ ! -e "$OLD_SOURCE" ]; then
  git worktree add --detach "$OLD_SOURCE" "$OLD_REV" &&
  git -C "$OLD_SOURCE" submodule update --init --recursive --depth 1 -- third-party/shmem &&
  (cd "$OLD_SOURCE" && python scripts/build_shmem.py)
fi
python tests/ll_control_opt/build.py \
  --revision "$OLD_REV" --send-cores 30 --resume
```

最后一条完成旧wheel及其manifest登记；随后四项套件可以同时找到旧SDK缓存和根工程
已准备的新SDK。若现有`source-s30`身份不符，保留它并另选全新工作目录准备，
不要强制checkout/reset已有缓存。

无法通过Git取得旧SHMEM、但已上传完整旧版依赖bundle时，也可单独构建旧项：

```bash
python tests/ll_control_opt/build.py \
  --revision c1e4faaf32806da84f8818e1ac3eb34864b7958d \
  --send-cores 30 --offline-deps .tmp/shmem-old-bundle --resume
```

该bundle必须匹配旧pin `4dd80e`；新pin `8e3fc610`在根工程另行准备。
完成后运行下面的suite时不再传`--offline-deps`，避免同一bundle被用于另一版SDK。

## 单机24次成对测试

设置本机16卡、真实master地址与空闲端口。这里显式使用单机；续跑结果功能也仅支持单机。

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
export SHMEM_PORT=29601
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-001 --resume --plan
```

计划应为4个wheel、24次ST调用：4候选×3seed×2轮。每次调用均包含dispatch和combine，
没有 `--dispatch-only` 或kernel打点。默认S30/R26、G4、Release、Ascend950、timer OFF。

确认计划后，去掉 `--plan` 即一次完成构建、安装、正确性、性能和汇总：

```bash
python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-001 --resume
```

也可先完整编译，再在设备可用时统一启动：

```bash
python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --build-only --resume --send-cores 30

python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --run-only --send-cores 30 --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --seeds 0 17 101 --rounds 2 --order alternating --run-prefix b256-topology-001
```

每次拉起前检查所选NPU是否空闲；忙或无法确认时停止，不终止其他任务。
中断后沿用相同清单、seed列表、S/R和run prefix，加入 `--resume-results`：

```bash
python tests/ll_control_opt/suite.py \
  --commits tests/ll_control_opt/commits-topology.json \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" --send-cores 30 \
  --seeds 0 17 101 --rounds 2 --order alternating \
  --run-prefix b256-topology-001 --resume-results
```

完整summary即使没有 `.done` 也会经全部rank/原始样本校验后复用。
不完整项保留，整项写入新的attempt；不会补采几条数据凑到100。
新的恢复汇总位于 `sessions/<session-id>/`，原结果与原汇总不覆盖。

## 采样与结果判读

固定shape为 EP16、B=M=256、H=3584、K=8、E=384、FP8、`do_cpu_sync=0`。
ST保持10次预热、100次采集、每rank尾50，以及每次Exp+scalar all_reduce。
每个请求seed先做3个正确性路由，正式性能使用最后一个路由：

| 请求seed | 实际采样路由seed |
|---:|---:|
| 0 | 2 |
| 17 | 19 |
| 101 | 103 |

第1轮每个seed按清单正序，第2轮反序；每个候选只减去**同轮、同seed**的reference。
不跨seed做差，也不把时间相邻候选默认当作因果对照。
反序能减轻位置偏差，不能消除温度、前序任务或跨session影响。

主要结果在：

```text
results/ll-control-opt/b256-topology-001/
  baseline/seed-0/round-001/summary.csv
  sdk-old/seed-0/round-001/summary.csv
  <candidate>/seed-<requested>/round-NNN/summary.csv
  comparison.csv
  comparison.json
```

主指标仍是用户指定的 `summary.csv` 中 `mean_kernel_us`：全部rank各自尾50样本的
算术平均。dispatch与combine分开；FP8配置的combine实际为BF16输入/输出。
`reference_delta_us/pct` 为当前减reference，负数表示当前耗时较低。

每个新调用在启动前写入 `*.node-N.build.json`，记录源提交、依赖ID和实际校验的
kernel/SDK hashes。比较表同时列出当前和reference的SDK身份。
历史结果缺少sidecar时保留 `legacy-unavailable`，不能用现在的SDK填充旧身份。
检查 `source_session_id`、`reference_crosses_sessions` 和SDK字段后，再解释小幅差值。

## CPU端点字节分类

下面的脚本只用标准Python生成与正式ST一致的uniform路由，不导入Torch，不启动NPU：

```bash
python tests/ll_control_opt/topology_workload.py \
  --world-size 16 --seeds 0 17 101 \
  --output-dir results/ll-control-opt/topology-cpu-001
```

输出 `summary.json` 和 `topology_counts.csv`，按source、destination、端点类别及
`initial_direct`／`initial_first_hop`／`relay`统计包数与LL packet字节。
使用请求seed+2，packet布局来自当前shape；输出目录须为新目录。

如需查看32-rank时的理论跨机端点分布，可另外运行：

```bash
python tests/ll_control_opt/topology_workload.py \
  --world-size 32 --seeds 0 17 101 \
  --output-dir results/ll-control-opt/topology-cpu-ep32-001
```

这仍只是CPU模型，不表示进行了双机测试。计数含LL packet布局/flag字节，
不含网络封装、重传或独立控制通信，也不测等待时间。
buffered/immediate调度应保留同一端点/字节多重集；这个脚本不是候选kernel的执行模拟器。
不能把总逻辑字节除以完整dispatch平均耗时，称为单端口线速或真实端口利用率。

## 官方SDK带宽／时延微基准

优先使用锁定SDK已有入口，不另建长runner：

- [官方perftest集合](../../third-party/shmem/examples/shmem_perftest/README.md)。
- [MTE perftest说明](../../third-party/shmem/examples/shmem_perftest/mte_perftest/README.md)、
  [运行入口](../../third-party/shmem/examples/shmem_perftest/mte_perftest/run.sh)、
  [device源码](../../third-party/shmem/examples/shmem_perftest/mte_perftest/mte_perftest_kernel.cpp)。
- [UDMA perftest说明](../../third-party/shmem/examples/shmem_perftest/udma_perftest/README.md)：
  作为另一种传输引擎单独记录，不能当作当前LL MTE路径的测量结果。

按官方说明在**独立的SDK实验目录**编译示例，不在套件正在使用的SDK build/install树内
原地重编。先核实设备/PE映射，再分别选本组FullMesh端点和同机跨组Clos端点；
跨机路径需要真正跨机设备运行。扫描包大小、并发核数和收发方向时记录实际参数。
SDK文档标明 `ascendc_perftest` 不支持Ascend950，不应把默认 `all` 作为本机的完整对照。

SDK微基准与完整LL的包布局、同步和重叠不同，应分别保留结果。
新SDK有QP/多链路接口也不等于已测出4个Clos端口各自的利用率；那需要相应端口遥测或
明确经过验证的链路选择。CPU端点分类和SDK微基准可帮助定位，不能替代这种观测。

## 当前验证边界

当前27机器不可访问，本次没有SSH、CANN完整wheel构建、NPU正确性或目标性能验证。
`8e3fc610`涉及SDK接口变化；全部7个dispatch入口、完整wheel链接及设备收发均需目标机
按上述步骤验证。旧SDK的历史编译通过不能证明新SDK可用。
本地CPU测试与plan检查只验证脚本、身份校验和逻辑约束，**目前不宣称SDK或调度候选有性能收益**。
