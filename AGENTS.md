> 当前为一包深度暂存/put流水，实现身份26、reference schema4；复用原lane，单lane立即提交。S表区分暂存经过、延后间隔、剩余等待及前包提交，不改packet所属核/顺序。详见 [ONE_PACKET_PIPELINE.md](tests/ll_rdma/ONE_PACKET_PIPELINE.md)；覆盖下方旧立即提交及schema2说明。

## 包级reference诊断（2026-10-28）

用户要求：主q排除于调度，S核只细分第二跨机packet并记录跨机/本组提交整体区间与原batch quiet；R核记录成功就绪和实际消费的稳定任务身份/顺序，在UB保存、本核w21后一次DMA导出。schema2每核5120B；R日志只取原lane后的剩余UB，最多128条，必须显式标记截断，不减少lane，不逐包加clock或新增业务屏障。当前通信协议24不变、SDK不改；OFF不含包诊断。交错区间不相加，put不是完成，quiet不是接收消费；w57包含导出，ON完整task扰动须另测。详见tests/ll_rdma/PACKET_TRACE.md。

> 当前协议24：已撤销全部8QP派生调度与缓冲改动，恢复原token分片、核内FIRST_HOP/DIRECT顺序和逐次Vector q；保留路由SO。当前实现以 [PER_CORE_QP_AB.md](tests/ll_rdma/PER_CORE_QP_AB.md) 为准，下方实验历史不作为现行行为。

> 2026-10-28更新：本分支当前协议23，每物理核QP，已恢复逐次Vector求q；不启用q预计算。测试脚本normal路由直接调用按用户tmp.py移植的make_random_expert_indices，默认倍率2、seed偏移0的输出与原脚本一致；hotspot保持原方式。下方协议22/保留q预计算描述属于历史对照7fbc309。

> 当前分支为新增每物理核QP基线，协议22，覆盖下方历史8QP说明。见 [PER_CORE_QP_AB.md](tests/ll_rdma/PER_CORE_QP_AB.md)。原四个A/B分支不改。

> 本分支属于四版独立性能A/B：当前 kernel/ll-ndr-ab-base。当前版本与对照关系以 [tests/ll_rdma/FOUR_WAY_AB.md](tests/ll_rdma/FOUR_WAY_AB.md) 为准，下方历史协议描述不作为当前安装身份。

# 目标RoCE通信契约

## NDR采样方式（2026-10-27最新用户要求）

本分支ST采用`initial-reduce-exp100-batch-v3`：每个操作的采样批次只在开始all_reduce一次，
预热同步后、进入active的profiler.step前固定100次Exp，再连续调用算子；循环内没有Exp或
采样all_reduce。独立timer/reference同样为初始all_reduce、预热、100次Exp、连续稳定/capture。
perf后附加打点时另排100次Exp，共用perf初始all_reduce；不在Exp前缀后额外同步。
正确性/批次边界和算子内部必要复用协调保留，kernel/SDK不改。新旧采样协议不能混作同条件A/B。
详见tests/ll_probe/PROFILING.md；本条覆盖本分支旧“每轮Exp+all_reduce”采样说明。

## SO发布

2026-10-27用户明确提供：跨机RoCE每个WQE的尾片具有SO特性，保证该SO之前的数据
先到达，且片内保序；单片默认2KB。后续本项目通信设计以此目标硬件契约为依据。

优先设计同一WQE发送一个完整消息：数据/元数据在前，末尾追加一个32B FLAG，接收端
观察尾FLAG确认消息就绪。不把现有body/flag两阶段put+quiet视为唯一正确方案。
2KB是网络分片粒度，不是消息大小上限；不自行扩展到跨QP/独立WQE、MTE或其他平台。

SO不替代发送源缓冲生命周期、接收消费完成、旧FLAG清理或epoch/双bank复用证明。
修改前核对实际SDK WQE拆分/SO覆盖和尾FLAG布局，并同步接收端与协议版本；SDK保持
既有实现，不能为该假设擅改SDK。

协议10已实现跨机payload整包单WQE：复用现有packet最后32B flag，接收端保留完整flag
检查（包含尾flag）；内部wire/机内MTE不改。dispatch删除gate副本/清gate/独立gate put，
每批每peer一次quiet；combine Packet每包一次put+quiet。独立路由/计数commit不因单WQE
SO被删。全rank更新完整wheel；具体实现及验证边界见tests/ll_rdma/ROCE_SO_PACKET.md。

详细记录及现有实现区别见[ROCE_SO_CONTRACT.md](tests/ll_rdma/ROCE_SO_CONTRACT.md)。
