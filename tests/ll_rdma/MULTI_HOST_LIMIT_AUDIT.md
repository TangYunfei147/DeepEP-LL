# 多机32P特例清理与风险审计（2026-10-29，协议28）

范围：当前 `kernel/ll-ndr-ab-per-core-qp` 的生产LL、Host校验、Python入口、
部署握手及安装检查；对照提交 `7293457`（协议27）。没有修改其他独立工作区或SHMEM。
已按用户要求删除未获确认的EP32专限，并同步其背后的两机算法假设；不是单改条件。

## 已删除或泛化的限制

| 原位置 | 原限制/假设 | 当前处理 |
|---|---|---|
| `dispatch_server_dedup_counts.h::Init` | `RDMA && epWorldSize_ == 32` 才直发count | RDMA跨物理机即启用；现有公开多机域EP32/64/128/256 |
| `ll_count_assignment.h` | 只用核0..15，`host ^ 1`找另一机，`experts/2`当本机专家范围 | 跳过本机16rank后枚举所有远端rank；本机范围按`16 * experts/world`计算；task=core、core+C等分轮 |
| `ep_memory_server_dedup_layout.h` | 只给EP32核算整卡count临时帧 | 所有多机域均核算workBuf临时容量，长期status保留区仍只含原本机计数 |
| `dispatch_server_dedup_send.h` | 仅EP32启用fullmesh首跳；一个token只有一个全局`meshFirstK` | 去掉EP32门槛；每个目标主机的成员mask独立选首跳，消除跨主机复用firstK |
| 旧`ll_route_exchange.h`及layout | `rankSize != 32`拒绝、`rank ^ 16`配对 | 此路径已不再调用，本次删除死代码、专属layout和旧配对模型；历史实现可从7293457读取 |
| `one_click.py`参数与ST调用 | node_rank只能0/1、num_nodes固定2、validate_args固定32 | 新增`--num-nodes`，node_rank按节点数校验，实际EP=16×节点数；默认2只是兼容默认值 |
| `one_click.py`安装身份 | 只查询`ranks_per_group(32)` | 查询本次实际EP对应的group配置，参数身份包含实际num_nodes |
| `target_ready.py` | 只接收节点1，两节点一致就放行 | v2握手收齐所有不同node_rank及相同身份后放行；缺失、重复、身份不一致拒绝 |

另外，`check_installed.py`及`csrc/ops/low_latency.cpp`导入时的Host/device策略互检
原本仅覆盖EP16/32，现补齐EP16/32/64/128/256。
`target32.json`等保留为两机示例；测试fixture中的32也不是生产上限。

## 仍保留且有依据的约束

- 每物理机16个连续rank：用户已确认；`ll_group_policy.h::kLlPhysicalRanksPerHost`。
  `world>16`表示跨机，count读取器的`world>=32`是当前公开域中多机的下界，不是上限。
- 公开LL域16/32/64/128/256及最大256：Python LL API的支持列表、
  `comm_args.h::DISPATCH_DEDUP_MAX_RANK_SIZE`和固定peerMems等数组容量。没有把该任务扩大为新ABI。
- 32B记录/FLAG/对齐、top-k≤32：wire及top-k位图语义，不是32P。
- LL控制容量16..72偶数核；**实际跨机每核QP仍受锁定SDK每peer最多56QP限制**。
  原始依据：`third-party/shmem/include/host/shmem_host_def.h`的`ACLSHMEM_MAX_QP_NUM=56`，
  锁定revision为8e3fc610；`runtime/ll_rdma_qp_config.hpp`在初始化前检查。72核CPU布局覆盖
  不代表本分支当前SDK支持72核跨机运行。
- HT/UDMA的`kQpCount==32`、32B tiling静态断言没有改动；它们不是LL多机rank数约束。

## 正确性审计

1. **不能直接删EP32判断。** 原`host ^ 1`只能访问一台配对机，放开后会漏发其余目标。
   当前压缩远端task空间为`[0,world-16)`，映射时跳过本机区间，核按C步长覆盖，包含所有零计数。
2. **不能跨目标主机共享firstK。** 原token级mesh可能选中另一主机的top-k槽，甚至选到
   本来是DIRECT的远端单条路由，导致发错主机/漏路由。现用各目标主机mask独立选择；
   生产Classify函数体及选取helper的CPU测试检查每条路由恰好出现一次，group不会混机。
3. **分轮不能提前复用GM/UB。** 每轮整帧put后保留quiet，下一轮再使用同核Stage/work；
   MTE3_S先保证注册GM写入完成，S_V保护SDK scratch再次给Vector使用。未借用跨WQE/QP SO。
4. **收发地址与共享commit必须一致。** 远端源整卡帧、同机源旧分片帧均按source-major放置，
   读计划按16-source机界拆分并还原expert-major前缀。最大128记录/核需要最多9段。
   所有接收核快照汇合后才清共享commit，帧首记录的唯一消费者负责清commit，body也清理。
5. **双bank复用仍需要调用约束。** 同通信器、同调用顺序、单stream；源端n+2依赖全部rank
   的n+1计数（包括零），各目标只有完成n及清理才能发布n+1。combine_reuse保持。
6. **部署不能只等节点0/1。** 全节点握手检查相同源码/wheel/SDK身份及运行参数；循环回环测试
   覆盖4/8/16节点、最后节点延迟、缺失节点、重复node_rank和后到节点身份不一致。
   相同身份在网络消息中只携带一份，避免随节点数复制导致合法身份超过64KiB帧上限。

## 剩余风险与验证边界

- 新的EP64/128/256路径尚未执行CANN编译、目标设备正确性或性能验证。CPU模型验证的是
  地址、分工、依赖、计数/成员守恒，不是NIC、缓存或设备运行资格证明。
- 机内旧分片帧仍要求一致的核数、E与计数参与方式；一键入口比较相同参数和active AIV，
  不支持绕过入口后让不同rank使用不同核数/不同B阈值参与方式，不能推导为任意异构调用都正确。
- 首跳策略扩展后，EP64以上可能选择不同的lead专家。q与原权重未修改，combine计算代码
  未修改，但“lead优先、其余成员按原k顺序”的浮点累加次序可能随lead改变。
  **不能宣称与旧域的最小k路径逐bit一致**；需目标机数值检查，尤其是有抵消的权重/输入。
- 每源跨机count帧数为`world-16`。核数不足时分轮，最大轮数`ceil((world-16)/C)`；
  以C56为例EP64/128/256分别最多1/2/5轮。多轮quiet和按专家Vector扫描可能增加count_send，
  需比较完整OFF dispatch；不因删除转发而直接宣称提速。
- 独立首跳选择复用原mask、没有新增UB数组，但需要重新读取各group的成员ID；
  所有group互不重叠，合计最多额外K次ID读取（连同Classify最多2K），EP32也可能有此开销。
- 每主机4个连续4-card fullmesh组的选择启发式来自既有目标拓扑。若其他机器机内拓扑不同，
  目标成员仍正确，但不能保证该启发式最优。未引入额外拓扑检测或猜测实际网络性能。
- 准备握手是运行前身份检查；最后检查通过后节点/网络失效仍可能使HCCL启动失败，
  不把它描述成能容忍故障的原子全机启动，也不会自动重试性能采样。

## 本地检查结果

完整NDR回归303项通过，2项因缺少Torch跳过；随后补充Host/device全域互检与测试指南容量
检查的5项通过，最终握手紧凑帧及多机入口相关49项通过。这些集合重叠，不相加。
生产分工、transport、reader及Vector计数函数体覆盖EP32/64/128/256，16..72核布局、
零计数/热点/长短路由、同核多目标、延迟MTE3、破坏性SDK scratch、共享commit及精确prefix。
生产Classify与逐主机首跳helper覆盖同token多主机/远端单条DIRECT/多个FIRST_HOP，检查成员不丢不重。
指南E1024/H3584/K8/M256、预量化FP8 dispatch+BF16 combine已通过56核Host容量模型。
这些结果没有代替CANN编译或目标NPU数值与性能测试。

## 目标机复现

所有节点更新相同协议28完整wheel；不能混装27。每节点设置NODE为0..NODES-1，
MASTER为节点0地址。下面使用E1024，适配现有全部EP域；用E384时须确保能整除实际EP，
所以EP256需显式选择E256/512/768/1024等合法值。

```bash
git fetch origin &&
git switch kernel/ll-ndr-ab-per-core-qp &&
git pull --ff-only &&
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" ndr-multihost-perf-001 \
  --num-nodes "$NODES" --mode perf --tokens 1 8 32 256 --capacity 256 \
  --hidden 3584 --topk 8 --experts 1024 --dtype fp8
```

`NODES`取现有LL支持的1/2/4/8/16；多机为2/4/8/16。每机仍是显式16设备。
不使用`--skip-build`复用旧协议wheel；失败保留交互shell和日志。采样协议、FP8 dispatch
配BF16 combine、单个≤60000字节回传报告的要求均保持。
