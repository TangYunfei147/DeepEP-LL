# 组内初发与就绪批次动态接收

分支：`kernel/ll-g4-local-send-dynamic-recv`，工作区`DeepEP-LL-g4-dynamic-recv`。

历史从`b3f9004`参考分段打点基线开始，重放组内初发提交为`a2eae05`（与原分支`a192113`代码一致），再演进本次动态接收。原`kernel/ll-g4-shared-recv@a192113`保留，可作静态接收A/B基线。

## 执行语义

S+R仍固定56，默认28/28，G默认4。S核先跨组初发，R核先本组初发并生产前缀。各核完成自己的前段工作、等到前缀及位置表发布后，即可进入动态接收；没有新增全核发送完成屏障。

每16个逻辑输出行组成一批。每批一个32位认领字，低16位对应行；尾批按实际行数屏蔽。先探测包的全部块flag，再CAS认领就绪位，实际拥有`ready & ~old`的行。CAS返回的旧值是唯一归属依据，重复或过期的探测结果不能导致重复消费。

不预占未就绪行，也不领取整段后原地等待。通过共享循环游标每次获取8批探测窗口，到队尾自动回绕。票据只安排探测顺序，不预占行；快核可继续领取后续窗口、认领任意就绪行，避免所有核各自全量扫描。每次就绪探测至多尝试两次CAS，冲突后继续扫描，不在锁上自旋。这避免所有活跃核等待terminal、但FIRST_HOP还未被处理的等待环。

批数少于56时，使用独立的到达顺序admission计数，活跃领取核最多为批数；不按物理核号指定消费者。admission与探测游标独立；游标走过队尾或回绕不会把稍晚完成发送的核排除。若队列已经全被认领，迟到核在分配接收UB之前返回。

## 唯一归属与结束

- 输出行号保持原来的`localExpert → sourceRank → q`次序，领取核不改变输出布局。
- 认领者独占该行的payload消费、输出元数据、FIRST_HOP全部转发和本槽flag清理。
- claimed位本轮只从0变1，不释放、不回收；未就绪行没有所有者。
- 某批全部行被认领时，唯一完成该状态转换的核将批完成计数加1。该计数表示“全部已认领”，并不表示DMA完成。
- 无任务核可以结束；每个持有任务的核必须处理完自己的全部认领行、等待lane并完成最终PIPE_ALL后才返回。下一次同stream kernel依赖整个kernel完成，不能以“全部认领”代替完成事件。

每轮每个magic bank都在前缀发布前初始化队列。首个前缀核复用已有片段总数，通过一次向量归约得到实际T，仅清零header和实际使用的认领字；队列空间仍按最坏接收容量预留。初始化DMA与Scalar原子的边界有显式同步，所有接收核的原有前缀ready等待同时保护位置表和初始化。

## Scalar、原子和内存开销

前缀生产核并行生成一次全局`输出行→packet slot`表，使用向量CreateVecIndex和DMA写入。动态核每批连续加载最多16个位置，省去逐核前缀二分、逐行pending索引初始化及RemovePending操作。

认领与进度均为**本卡GM Scalar原子**，不使用跨卡/RDMA原子，也不通过SetAtomicAdd/SetAtomicMax实现fetch语义。空探测不做行认领CAS，但仍有摊薄的探测票据及完成计数器检查；不是零原子轮询。无竞争且整批就绪时，一批通常1次CAS+1次批完成Add；另有每8批的票据、每32次候选迭代的完成检查，以及小队列admission。部分就绪、争用和尾部扫描会增加操作，不能承诺每批总共只有一次原子。

领取后沿用最多8个预取lane，跨批持续复用。清flag使用独立常量零buffer，按MTE3顺序排在本行输出/relay之后，不在每批添加PIPE_ALL；只在本核结束时统一drain。每核只保留16行描述符及“已知整批被认领”的小位图，不构建全量pending表。

队列内存按容量C计算：

```text
batches = ceil(C/16)
claimBytes = align512(batches*4)
queueBytes = 1536 + claimBytes + align512(C*4)
C = EP * B * min(topK, E/EP)
```

header三个字段分隔512B：探测游标、全批认领数、admission计数。队列复用Dispatch阶段未使用的source-return暂存；空间不足时仅按通用公式扩大该预留，packet base由同一Host/device布局函数计算。队列不挤入1MiB combine控制区，不额外申请Torch设备tensor。Combine在后续串行调用中重用暂存，算法保持原样；原communicator/stream序列约束继续生效。

位置表生成、队列初始化、Scalar CAS和重复扫描均有真实成本。小队列活跃核限额、16行批次会影响并行粒度，不能预设一定比静态分片快。

## 构建和验证

全rank使用同一新分支、同S/R完整wheel。实现身份`g4-local-send-dynamic-recv`，核心标识11000+S，参考分段schema7不变。旧Host/device或其他布局混装拒绝；无需因本次改动强制重建已校验SHMEM。

在目标机工程根目录、匹配CANN/Python环境中：

```bash
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28
export DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

新clone按[LL_KERNEL.md](../../LL_KERNEL.md)准备子仓/SDK。原ST参数可以继续使用，新run-name/目录保存结果。先对随机、热点、空接收rank、同组、跨组、FIRST_HOP及重复窗口验证正确性，再以`a2eae05`或代码相同的`a192113`做同S/R/G/shape/seed/Exp的OFF性能对比，沿用10预热/100采样/每task最后50条。不得把ON诊断当作OFF性能。

ON诊断沿用参考阶段；位置表构造归BUILD_EXPERT_TASKS，探测归RECV_POLL，原子和调度成本包含在LOCAL_WINDOW_COPY/FORMAT_OUTPUT整体内，不新增逐包时钟读。原S/R worker标签仍表示前段角色，不表示固定接收所有者。compare.py允许已知10000/11000身份对照，动态策略改变时不计算dispatch逐核阶段比值；完整perf仍按配置身份核对。

## 接口参考与限制

查阅了[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend/tree/3b25377d04b24fc6154698ded78a2bcb2c59afff)的同步/原子封装和[dc_opt A5实现](https://gitcode.com/zhong-zixin/ops-transformer/blob/35741e3cd6f5ca8e4799a57f90a0c1975153a7ad/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h)，未引入其通信backend或SIMT kernel。

本实现依赖Ascend950 Scalar返回旧值的[AtomicAdd](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/atomic_operations/AtomicAdd.md)和[AtomicCas](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/atomic_operations/AtomicCas.md)。计数字和认领字在dispatch活跃期只通过原子访问，避免与缓存Scalar读写混用；DMA只在消费者放行前初始化这些独立区域。原生编译失败应报告，不静默替换成不返回旧值的DMA原子。

本机无C++编译器/CANN/NPU。Python回归和协议模型不验证NPU原子内存序、MTE可运行性或实际收益。提供了实际调度器的std::atomic多线程测试、双bank变更T测试以及实际布局C++检查，需有编译器的目标环境执行；完整设备正确性和性能仍待验收。

本次本地记录（2026-10-20）：395 passed、53项缺C++编译器跳过、9项Bash/POSIX检查未选。动态队列协议模型42项通过；实际std::atomic调度器及布局11项在跳过项中。CANN编译、NPU正确性和性能均未验证。

后续逐特性 Scalar 优化及本地回归记录见 [SCALAR_AUDIT.md](SCALAR_AUDIT.md)。
最新完整回归为 523 passed、53 skipped、9 deselected；仍无 CANN/NPU 验证。
