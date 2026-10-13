> 当前逻辑组大小与每机物理卡数已解耦，配置及分支关系见 [LL_KERNEL.md](../../LL_KERNEL.md)。本文原有每机16卡描述仅对应旧部署/专用 run_ll16.sh。

# LL固定值与隐含约束审计

范围：当前生产LL dispatch/combine、其Host绑定与Python选择路径；基于a07737b及本次删除调试代码后的源码。审计不等于目标机设备验收。本轮删除LL_DISPATCH_DEBUG_*、LL_DLOG、轮询打印计数器及调试头文件；不改通信协议、资源上限或收发算法。

## 结论与优先级

没有发现继续把64当作生产启动核数、把32当作实际接收核数，或等待固定核56..63的残留。生产launch使用查询值，分片/等待使用实际N或N/2。仍有以下需要明确的固定约束与输入前提。

### 优先关注：combine实际接收行数10000上限

- [combine配置](../../csrc/kernels/combine/low_latency/ep_memory_server_dedup_combine_config.h)定义`kEpServerDedupCombineMaxTokenNum=10000`。
- [Host BuildLayout](../../csrc/kernels/combine/low_latency/combine_layout.h)使用`min(capacityRows,10000)`检查UB；这并不保证实际接收行数R不超过10000。
- [CombineMemoryServerDedupKernel::Init](../../csrc/kernels/combine/low_latency/combine_server_dedup_kernel.h)读取设备前缀最后一项，实际R>10000直接`Trap()`。
- Python `_combine_low_latency`检查输入与handle的形状一致，但没有针对实际R的10000行上限做完整Host保护。

纯CPU调用现有布局函数得到（N=56、EP16、H7168、K6、E256）：

| B | dispatch布局/UB检查 | combine布局检查 | capacityRows | 设备上限 |
|---:|---|---|---:|---:|
| 32 | 通过 | 通过 | 3072 | 10000 |
| 256 | 通过 | 通过 | 24576 | 10000 |

B256时，如果所有来源都把每个token的6条合法且唯一的路由指向同一rank上的6个专家，该rank可收到16×256×6=24576行，超过设备上限。这个后果由代码和计数推导，未在目标机做故障注入。可能表现为设备错误及其他rank继续等待。

你的此前B32配置容量上界仅3072，因此不会触发这个10000上限。也不能直接用输入Tensor的分配行数大于10000来拒绝所有调用：容量布局的分配行数可能大于真实R。建议后续针对实际R增加一致的超限处理，或支持分块；本轮不改变该限制。

### 优先关注：每个source/expert只有B个槽，路由值前提未完整校验

[Sender](../../csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h)直接把expert ID转为无符号值，再计算目标rank和槽位；`CalTokenSendExpertCnt`在首token的`index<K`时直接返回q=0。[布局](../../csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_data_layout.h)为每个source/expert预留B个槽。

这依赖每个token的expert ID合法且不重复。首token路由`[0,0]`会让两次发送都使用q=0，而计数包含两条；非法负数或≥E的值还可能产生非法目标索引。当前 `_low_latency_plan`主要检查类型/shape/连续性；`_notify_low_latency`过滤无效ID用于计算行数，并没有清除实际发送Tensor里的无效路由，也没有去重。

这是源码推导出的未保护输入前提，尚未进行设备故障注入。原测试生成的是合法唯一路由，因此常规测试通过不能证明异常路由安全；HT支持重复/无效槽的语义也不能直接套到LL。建议明确LL输入契约并增加快速失败或设备错误传播。

### 卡死相关：轮询没有kernel内超时/错误传播

计数的`WaitDispatch/GetCumSum/WaitCumSumFlag`、接收packet的pending循环、combine的`WaitFlags`与源token等待均依赖标志到齐后退出。它们现在等待的是实际参与者，已无旧固定64核问题；但缺包、某rank退出或设备Trap仍可能让其他参与者持续等待。

外部进程超时不等于kernel具备超时协议。建议后续统一错误标志和有界诊断机制；本轮删除的是打印，不改变原等待退出条件。

## 固定值清单

| 固定值 | 实际作用与保护 | 判断 |
|---|---|---|
| N为16..72的偶数、收发1:1 | [ll_core_config.h](../../csrc/kernels/dispatch/low_latency/common/ll_core_config.h)限制支持范围，查询后不满足则报错；72是容量，不是启动数量 | 有保护的支持范围/策略，尚非任意核数、任意比例 |
| CPU计划默认64 | 布局辅助函数兼容无硬件计划；生产调用显式传入查询的N | 不是生产64核残留；新调用点不能遗漏coreNum |
| 每逻辑主机16rank | 路由分组、localRank、ready行16个标志；启动脚本要求每机16进程 | 用户确认的目标拓扑约束，不自动发现任意物理拓扑 |
| 8个通知核 | 最后8个实际核，每核通知组内两个peer；准备发布循环/复制块数/peer步进还存在字面量8 | 当前一致，存在配置耦合；不能仅改kLlNotifyCoreNum |
| 36KiB magic前缀、combine控制1MiB、准备行2304B、ready偏移18432B | 与72核预留和8通知核联动，有静态/Host布局检查；payload从2MiB开始 | 固定线协议，需全体rank同版本重建；修改容量须联查布局 |
| 每rank 8GiB、双窗口各4GiB | Host要求容量恰好8GiB，窗口切换和地址公式共享常量 | 明确工作区约束，不能直接换成小缓冲 |
| peer基址rank×1024偏置 | [Host绑定](../../csrc/ops/low_latency.cpp)在SHMEM映射之后加偏置，Session为8GiB路径额外分配2MiB | 固定地址约定，不能只改一端 |
| dispatch/send UB 190KiB，combine recv UB 240KiB | Host和device使用同一预算，但没有按GetCoreMemSize动态选预算 | 仍依赖目标架构/工具链；核数动态化不代表UB也自动适配 |
| M>32启用去重 | dispatch/combine共用mode函数，取决于容量M而非B | 调优阈值，不是AIV数量 |
| B<512全核计数 | 超过阈值才仅接收半区计数；公开Python LL要求B≤M≤256 | 当前公开LL路径始终全核计数，512分支不能当已覆盖常规路径 |
| K≤32、E≤1024、P≤256及Python EP集合 | 掩码、peer表及资源边界；有多层shape检查 | 有保护的格式/容量上限，不宜单独扩大宏 |
| relay条目expert/q各16bit | 打包为expert|(q<<16)，接收端按0xffff及右移16解码 | 协议格式；需同时满足合法路由与槽位范围 |
| 最多8个搬运lane、M4输入3、M5输入7/输出2，事件7预留 | UB分配与硬事件ID复用共同决定 | 流水资源约定，不是收发核数；调整需联查事件冲突 |
| 32B记录/对齐、512B槽/packet块、480B payload | 两次Copy的64和56分别是256B和224B，合计480B | 字节布局/向量宽度，不能把64/56机械替换成实际核数 |

`comm_args.h`还保留部分旧协议常量，如credit IPC core count和AIV_PER_AICORE；本次核对的生产LL调度链没有用它们决定参与核数，不能仅凭名字把它们算作活跃的固定核数缺陷。

LL的数据面仍要求所有peer有可用的MTE映射，直接DataCopy没有自动RDMA回退；这也是运行前提，但与核数硬编码是不同问题。

## 维护与验证

- 调试头文件及旧DISPATCH_DEBUG.md已删除。Host的`--trace-ops`和`[LL cores]`配置输出保留，它们不是本次删除的kernel编译宏。
- 生成器会清除生成kernel树中源端已删除的文件，避免旧调试头残留；生成快照仍不直接编辑。
- 10000行保护缺口的CPU复核输出保存在工程内`results/ll-hardcoded-audit/capacity.log`，没有启动设备或伪造EP16。
- 调试删除后43项核心/启动/路径CPU回归通过，完整生产wheel与profile探针编译通过，快照适配器和过期头清理检查通过；完整目标机dispatch/combine仍需用户验收。本轮未放宽上述上限，也没有把尚待修复的风险描述为已解决。
