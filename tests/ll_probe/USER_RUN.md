# 在目标机上执行

**性能采样已统一参考协议，正式测试请使用[统一profiler入口](PROFILING.md)。默认预热10次、采集100次、按kernel/task统计最后50条；下文旧Event命令用于诊断和正确性检查。**

新版四层测试及2/4/8/16/32通信域请先阅读 [four_layers/USER_RUN.md](four_layers/USER_RUN.md)。以下保留早期双卡搬运实验和依赖准备说明。

当前任务是交付经 A5_241 验证的实验脚本，由用户在目标机运行。验证环境不用于生成目标互联结论。

## 1. 独立发送量带宽实验（同机两卡）

先按[路径及SDK流程](PATHS.md)准备仓内依赖，加载本机已有CANN环境。以下命令均从工程根执行，不要求目标机与验证机目录相同。

先编译、再跑接收端正确性验证：

```bash
# 从工程根目录执行
bash tests/ll_probe/build_probe.sh
bash tests/ll_probe/run_peer_validation.sh
```

通过判据是 PASS、receiver checks before/after、normal cleanup 与退出码 0。当前设备对固定为 0→1；这是同进程 IPC 映射的同机 P2P 实验，不覆盖跨节点或多 peer 拓扑。不支持的 peer 路径会失败，不自动降级本卡拷贝。

先查看扫描计划，再由用户执行目标机测量：

```bash
bash tests/ll_probe/run_scan.sh > results/target-plan.json
bash tests/ll_probe/run_scan.sh --execute \
  --bytes 32 128 512 2048 8192 32768 131072 524288 2097152 8388608 33554432 67108864 \
  --cores 1 8 32 --lanes 1 4 --chunk 4096 --modes 0 1 2 3 \
  --warmup 10 --samples 50 \
  --output results/target-peer-001
```

输出目录必须全新。默认扫描 288 个配置，各配置重新初始化进程；初始化与接收端校验不在采样区间。先用少量配置试跑，再执行全扫描。确认对应方向/互联的十进制 GB/s 参考峰值后，追加 --peak-gbps 数值，summary 中 attainment=有效GB/s/参考峰值；比例不裁剪。

mode 0=空 kernel；1=UB 分配/初始化及事件；2=预置 UB→peer GM；3=本卡 GM→UB→peer GM。bytes 是每个 kernel 每个发送 rank 的实际搬运字节数，不是完整 LL 逻辑 payload。每核拥有不同地址，单次发送 iterations=1。保留 mode 0/1 作为参照，不从带宽结果中默认减去它们。

event_us 是 ACL event 包围的 stream 区间，包含事件/调度影响，不等于 profiler 单 kernel task；host_us 包含调用、结束事件提交及同步等待，另列。wire_gbps=bytes/(event_mean_us*1000)。测试 kernel 不包含 LL 路由、量化、去重/relay、接收轮询或 combine 归约，因此不能把它的截距当作当前 LL 固定开销。

## 2. 跨主机两 rank 实验

两节点使用相同源码、CANN 和 SHMEM 锁定提交，工程绝对路径可以不同。依赖统一留在各自工程的 third-party 内：

```bash
git submodule update --init --recursive --depth 1
# 从工程根目录执行
bash tests/ll_probe/build_probe.sh
bash tests/ll_probe/build_shmem_probe.sh
bash tests/ll_probe/build_shmem_driver.sh
```

build_shmem_probe.sh 仅调用 scripts/build_shmem.py --if-needed，SDK固定为 third-party/shmem/install/shmem；不另建实验依赖目录。TCP bootstrap用于建域，数据面为MTE映射；构建身份随结果记录。

目标互联必须允许该映射上的 MTE 直接访问。仅有 RoCE/RDMA put 能力的链路不能自动等价为 LL 的 DataCopy(peer GM, UB)；若初始化/映射/发送失败，保留日志回传，不切换成别的传输后继续标为 LL MTE 带宽。

先在两台主机分别启动正确性检查，使用 rank0 主机可达 IP 和未被占用的端口。每台只运行一个进程，device=0 表示本机可见设备 0，无需相同物理卡序号：

```bash
# 两节点设置相同 ENDPOINT；把占位地址替换为 rank0 的真实可达地址。
ENDPOINT=tcp://RANK0_IP:19731

# 节点0
bash tests/ll_probe/run_shmem_scan.sh --rank 0 --device 0 --endpoint "$ENDPOINT" \
  --execute --validation-only --bytes 32 15392 --cores 8 --lanes 4 --modes 2 3 \
  --output results/target-cross-check-rank0

# 节点1，在另一终端执行
bash tests/ll_probe/run_shmem_scan.sh --rank 1 --device 0 --endpoint "$ENDPOINT" \
  --execute --validation-only --bytes 32 15392 --cores 8 --lanes 4 --modes 2 3 \
  --output results/target-cross-check-rank1
```

两节点都通过后，执行发送量扫描；下面命令在两节点分别运行，设置 RANK=0/1，其余配置与顺序一致。每轮每配置依次创建 SHMEM 作业，不要同时启动另外一轮占用同一 endpoint 的扫描。对端等待初始化超时为30秒，父进程每case总超时120秒，请在30秒内启动双方。

```bash
bash tests/ll_probe/run_shmem_scan.sh --rank "$RANK" --device 0 --endpoint "$ENDPOINT" --execute \
  --bytes 32 128 512 2048 8192 32768 131072 524288 2097152 8388608 33554432 67108864 \
  --cores 1 8 32 --lanes 1 4 --chunk 4096 --modes 0 1 2 3 \
  --warmup 10 --samples 50 \
  --output "results/target-cross-rank${RANK}-001"
```

需要带宽达成率时，在两节点命令中都追加相同 --peak-gbps 数值。rank0 保存发送侧 Event 与 Host 原始样本；rank1 是接收方，不能用它的空 CSV 宣称采样缺失。每个节点的 summary status=local-ranks-passed 只证明本节点正常退出，必须再合并两方证据。

将 rank1 的完整结果目录复制到 rank0 授权工作目录下（不要覆盖已有目录），执行：

```bash
python3 tests/ll_probe/merge_shmem_results.py \
  results/target-cross-rank0-001/summary.json \
  results/target-cross-rank1-001/summary.json \
  --output results/target-cross-merged-001.json
```

合并要求两主机 hostname 不同、rank分别为0/1、配置/样本数/endpoint/二进制一致、双方接收检查及正常退出成立。最终 status=passed 才表示该双rank实验通过。--allow-same-host 只用于启动器验证，输出会标记为same-host验证，不用于目标跨主机报告。一个双rank实验只能刻画该方向的一对链路；反向测试可互换两节点rank身份并使用新目录和endpoint。

A5_241 可运行 --local 进行两个本机独立进程验证；本机通过不证明目标跨节点映射一定可用。目标物理互联验收由上述用户运行完成。

已验证范围：CANN 9.3.0编译、本卡240场景、同机IPC四场景、SHMEM两个独立rank四模式、最小Event采样和双方退出/合并。尚待用户完成目标同机及跨机物理链路检查、正式带宽采样、原生LL多rank测量。最小采样用于检查计时API可运行，没有作为硬件性能结论。

## 3. 当前 LL 小负载与固定开销估计

使用已编译安装的当前仓库及匹配 SHMEM。LL 原生支持 EP16/32/64/128；A5_241 双卡不能运行此项。以下是用户在目标环境执行的模板，H/K/E/dtype 是示例，按实际负载修改，不代表已知目标机规格。

先执行 CPU 计划检查：

```bash
cd .
python3 tests/elastic/test_ep_low_latency.py --plan \
  --num-processes 8 --num-nodes 2 \
  --num-tokens 1 2 4 8 16 32 64 96 128 256 --num-max-tokens-per-rank 256 \
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0
```

目标环境每个节点配置正确的可见卡、MASTER_ADDR、独立 MASTER_PORT 与 SHMEM endpoint；不要在 A5_241 使用 8 卡。目标节点数×本机进程数必须对应受支持的 EP，所有 rank 使用相同源码/二进制/参数。按仓库 tests/README.md 加载 CANN/SHMEM 环境。各节点运行同一命令，仅 NODE_RANK 不同：

```bash
python3 tests/elastic/test_ep_low_latency.py \
  --num-processes "$LOCAL_RANKS" --num-nodes "$NNODES" --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
  --shmem-ip-port "$DEEPEP_SHMEM_ENDPOINT" \
  --num-tokens 1 2 4 8 16 32 64 96 128 256 --num-max-tokens-per-rank 256 \
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0 \
  --num-bytes 8589934592 \
  --output-dir results/target-ll-001
```

固定 M=256 避免 B 与容量布局同时变化；不使用 --skip-check。确认全部 rank 正常退出，summary status=completed、各 case correctness=passed、backend=ll。保存全部 rank 原始样本与日志，不能只凭摘要存在判定成功。

对实际生成的 summary.json 进行拟合：

```bash
python3 tests/ll_probe/analyze_ll_overhead.py \
  /实际运行目录/summary.json --b-min 1 --b-max 32 \
  --output results/ll-fit-small.json
```

拟合使用原始 profiler 样本重算每 rank 均值，再取最大值；与现有 ST 口径一致。按 H/K/E/dtype/M/route/EP 分组，t(B)=alpha+beta×B。alpha 是经验截距估计，不是零 token LL 的实测固定成本。检查残差和不同拟合区间，负截距或低 R²须保留并解释；不能以此单独认定固定成本大小。若需要精确定位，后续加入 LL 阶段插桩，由未插桩生产路径作基线。

分析同时保存每个rank的独立拟合、各B的最慢rank、残差RMSE和截距标准误。若最慢rank切换，将产生警示：max-rank包络的截距可能与单rank固定成本不同。标准误依赖独立、同方差的线性模型假设，不覆盖profiler系统偏差，不能当作硬件固定成本的严格置信保证。通过改变 --b-min/--b-max 检查截距稳定性；正常rank退出仍需单独核对日志。

## 回传材料

回传 peer 扫描目录的 plan.json、summary.json、逐配置 CSV/log；跨主机双方完整目录与 merged.json；LL 完整 summary.json、逐 rank 原始样本/退出日志、源码 revision 与已加载依赖版本。无需透露机型名称；注明方向、同机/跨节点、参考峰值的统计口径即可。
