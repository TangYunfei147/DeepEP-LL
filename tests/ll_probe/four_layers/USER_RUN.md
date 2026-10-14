# 四层测试 v2

源码定位见[脚本与kernel对应表](../KERNEL_MAP.md)，包含Host入口、设备函数名、生产LL内部阶段及快照修改规则。

**性能采样已统一参考协议，正式测试请使用[统一profiler入口](../PROFILING.md)。默认预热10次、采集100次、按kernel/task统计最后50条；下文旧Event命令用于诊断和正确性检查。**

这版以四层为验收边界。通信域覆盖2/4/8/16/32。旧版搬运包的验证不等于四层全覆盖。目标机运行由用户执行，A5仅验证设备0/1子集；跨主机映射和真正N方竞争须在目标机验收。

## 构建

先按上级README通过Git clone本分支，测试目录为 `tests/ll_probe`。沿用上级指南准备同一CANN及锁定SHMEM，再执行：

```bash
# 从工程根目录执行
bash tests/ll_probe/bandwidth/build_sdk.sh
# 以下仍从工程根执行
python3 tests/ll_probe/metadata/prepare_snapshot.py
bash tests/ll_probe/four_layers/build.sh
python3 tests/ll_probe/native_ll/tests/verify_adapter.py
```

prepare_snapshot.py仅改实验快照，不改生产仓库；记录原sender源码SHA256。计时与正确性两个元数据实现使用不同模板名称，避免链接时混用。每次换仓库版本应重审提取边界，不能盲目复用旧二进制。

## 第一层：空载与tiling

通信实验统一入参为src、peer地址表、GM tiling。layer=0是完全空启动；layer=1读取32B tiling中的字段后返回，不创建UB或处理路由。volatile读取防止编译器将基线删除。两者都保留，避免把tiling读取成本藏入启动基线。

## 第二层：真实路由准备提取

```bash
bash tests/ll_probe/four_layers/run.sh tests/ll_probe/metadata/build/metadata_probe 32 7168 6 256 16 10 50 \
  > results/metadata-B32.csv \
  2> results/metadata-B32.log
```

参数为B、H、K、E、逻辑专家分区数、warmup、samples。逻辑专家分区数用于路由分类，完全不创建对应数量的SHMEM rank；此实验仅使用设备0。维持E可被逻辑分区数整除、K≤E，并通过UB预算检查。

提取当前LL Sender::Process路径，保留路由加载、前缀q计算、去重分类、DIRECT任务排序、sourceMask/sourceWeights生成和原UB准备。禁用payload打包/量化/发送；不包括packet tail构造、receiver元数据及跨rank计数交换，不能标为LL全部元数据耗时。

正确性版本输出逐路由q，与CPU黄金结果核对掩码/权重/q。计时版本每核仅额外输出一个checksum，消费q结果以防工作被优化掉；该checksum输出及同步仍包含在计时内，报告为带明确观测开销的提取阶段耗时，不能直接从完整LL减去它并宣称得到纯通信耗时。计时循环每轮检查checksum，检查不在event区间内。

## 第三层：统一通信场景

scenario/tiling key：1=rank0→rank1；2=所有非0 rank→rank0；3=rank0→所有非0 rank；4=所有rank→除自身外全部rank。同一个Host入口按key启动对应模板专门化kernel，GM tiling包含rank/world/单边字节/chunk/lane/layer/key。每个发送方在每个接收方有独立slot，避免多对一数据覆盖。

在每个实际rank的终端运行（RANK/WORLD/DEVICE/ENDPOINT由用户配置），同机和跨主机都使用这一入口；同机每rank选择不同设备，跨主机每rank选择本节点可见设备序号。默认可见设备0/1；用户仅在目标机设置PROBE_VISIBLE_DEVICES显式扩展，例如域8同机设置0,1,2,3,4,5,6,7，DEVICE=RANK。A5_241始终不设置扩展，不使用设备2+。所有rank同一轮使用相同参数和顺序。

```bash
# 按所需通信域和场景逐项运行profiling/profile_probe.py --plan检查参数（见../PROFILING.md）
# 仅用户目标机，以下举例同机EP8：
export PROBE_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

按2/4/8/16/32域、四场景与发送量规划，每配置依次执行layer0/1/3。单边最大64MiB且每rank接收工作区上限256MiB，因此大域的单边上限是min(64MiB,256MiB/world)。

```bash
# 从工程根目录执行
OUT=results/four-scene4-rank${RANK}-001
mkdir "$OUT"
bash tests/ll_probe/four_layers/run.sh tests/ll_probe/communication/build/comm_probe "$RANK" "$WORLD" "$DEVICE" "$ENDPOINT" \
  15392 8 4 4 3 10 50 > "$OUT/samples.csv" 2> "$OUT/run.log"
```

末尾参数依次为每条边发送字节、AIV核数、lane数、scenario、layer、warmup、samples。chunk当前固定4096B。改变字节数扫描曲线；分别执行scene1/2/3/4，layer0/1/3，始终使用新输出目录和作业endpoint。先启动rank0，等run.log出现READY_SHMEM后再启动其他rank。锁定SDK客户端连接仅重试约20秒，不要让它在rank0设备初始化前耗尽窗口；不能混入不同作业。

source每次发送前从本卡GM读数据，通过UB发送到SHMEM的MTE可寻址peer映射。只支持该直接寻址数据面；若目标互联仅支持RDMA put，初始化或传输失败须回传日志再适配，不自动换路径。

CSV记录每rank实际wire字节、Event区间、Host调用至本地stream完成的api_us，以及包含测试barrier的job_join_us。job_join_us明确含测试控制同步，不能标为生产LL接收协议耗时。接收验证在采样前后进行，全输出和末尾guard均校验；两个rank下N对1/1对N退化，不能证明N方竞争。

单边发送量固定时，1对N和N对N的每rank总字节会增长，必须同时记录单边和全作业字节。全作业带宽使用逐轮rank最大Event区间作为明确统计口径，不能用某个rank的字节和整个作业的峰值混算。正式结论要求所有rank正常退出并回传原始CSV/log。

复制所有rank结果到授权根目录后，使用analyze_comm.py审计CSV/log及每条边字节数，再输出逐轮rankMAX、每rank和全作业mean/p50/p95/p99：

```bash
python3 tests/ll_probe/communication/tests/analyze_comm.py results/rank*/samples.csv \
  --output results/comm-summary-001.json
```

不要将不同配置混进同一通配符。需要达成率时追加--peak-gbps及--peak-description，说明匹配方向和拓扑的全作业参考峰值。脚本会拒绝缺rank、样本不对齐、错误字节数、非有限时间或缺失接收/退出记录。

## 第四层：完整dispatch/combine

使用生成的 `tests/ll_probe/native_ll/snapshot/test_ll_four_layers.py`，参数沿用上级指南的原生LL命令，仅替换测试脚本路径：

```bash
bash tests/ll_probe/four_layers/run.sh python3 tests/ll_probe/native_ll/snapshot/test_ll_four_layers.py --plan \
  --num-processes 16 --num-nodes 1 --num-tokens 1 2 4 8 16 32 \
  --num-max-tokens-per-rank 256 --hidden 7168 --num-topk 6 --num-experts 256
```

真实运行时去掉--plan并提供原指南中的node-rank/master/shmem参数及授权目录内output-dir。原生LL仍要求EP16/32/64/128，因此本项目这组通信域中完整LL仅执行EP16/32，EP2/4/8标记unsupported，不降级HT或伪造world_size。目标已有的wheel必须与源码/依赖匹配。PROBE_VISIBLE_DEVICES设置本节点实际使用设备，LOCAL_RANKS=16（每节点固定16个rank），与NNODES乘积分别为16或32；不在A5执行这些原生LL任务。

每个dispatch/combine结果保留原profiler kernel_us，另外保存full_api：单独无profiler采样，计时从invoke之前到返回event等待+device同步完成，包含API内部开销，排除输入准备、专家计算、测试barrier、返回输出销毁。保存每rank原始样本与逐轮rankMAX、mean/p50/p95/p99。此模式是正确性检查之后的稳态复用，不包含Buffer构造/冷启动。

## 建议补充（单独列项）

1. 跨rank计数交换、magic/ready控制协议：与纯本地路由准备分开，需要生产合法EP验收。
2. 接收可消费时间：区分发送MTE完成、接收轮询/解包完成；测试barrier不能替代生产ready协议计时。
3. 冷启动/首次调用与稳态复用：单独记录Buffer建立、注册映射和首次API调用，不混入带宽曲线。
4. 实测dispatch→combine roundtrip：使用同一handle，计时中不包含专家GEMM；不能相加独立测量替代实测。
5. 随机/本地/远端/去重热点路由，FP8量化及packet开销，双向竞争；后续依据首轮结果确定优先级。

四层是归因实验，阶段可重叠、缓存与协议条件可能不同，不强制四层时延相加等于完整算子。请回传完整目录、源码/二进制身份和实际rank布局，无需披露机型名称。
