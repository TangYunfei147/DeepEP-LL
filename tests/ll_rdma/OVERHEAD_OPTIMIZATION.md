# NDR冗余工作清理：独立提交与验收

对照基线为`85d9cdc`，分支仍为`kernel/ll-ndr-hns1825`。
此系列保持dispatch/combine公开参数、输出tensor形状/类型、handle字段、原始q编号、
权重与归约顺序。核心协议9、timer schema3、物理16卡机界和G4/G16选择保持。
全rank安装当前完整wheel；已有匹配SDK校验复用。

LL生产轮询不使用`GetSystemCycle`，不新增超时、Trap、错误重试或降级。
原mask内容检查移到每token首次消费；原结果初始化错误路径继续检查初始化完成。
阶段/专家计时仅存在于`DEEPEP_LL_STAGE_TIMER`诊断构建。
用户明确保留锁定SHMEM SDK自身实现；SDK源码、gitlink与PI模式规则未改变。
旧HT与已停用Host设备barrier源码不在当前LL构建源列表内。

## 提交清单

按依赖顺序列出代码和验证提交；每个提交都可单独审阅。

| commit | 改动 |
|---|---|
| 82b57f7 | 删除生产LL的cycle时钟和超时轮询 |
| 664d327 | Receiver lane数量不超过实际任务数 |
| cb03413 | 复用metadata构造时已读取的packet role |
| 03c37a1 | Packet提供单次发送免128B恢复路径，默认仍保留复用语义 |
| 87f8363 | RDMA batch初始化时校验/缓存stage，保存已计算的远端地址 |
| 3f58c24 | 两轮quiet复用相同的唯一peer位图 |
| ffb083b | 验证远端分支入口契约与动态地址边界 |
| 348aad8 | combine按实际核数通知/清理，删除重复排空，缓存mask期望值 |
| b2bd180 | 两bank只初始化flags，保留原调用前完成协调 |
| 81c922b | combine显式使用单次发送免恢复接口 |
| cfb0d2e | 将期望flag数装入原pending槽，保持原UB布局和shape准入 |
| 3c2a135 | 生产LL无取时钟调用及构建范围的回归约束 |
| f9b7e18 | 按原展平路由顺序一次生成q表 |
| d8fe5d4 | 稳定DIRECT分组链表与每token一次sourceMask写入 |
| cb25027 | 缓存最近token输入/量化body，保留队列与pack读取依赖 |
| 588b74e | 删除非receiver0在count阶段的无用prefix读取 |
| 7f607e1 | 配对计数最后一次MTE写在已有复用边界退休 |
| 6bbc363 | 仅core0执行路由body准备等待 |
| be56022 | 复用Scalar累计prefix末值，删除重复subtotal归约 |
| 9cedef8 | 发布单份共享只读prefix，保留控制区布局容量 |
| 121530c | 删除已夹取本机专家范围后的重复判断 |
| 93b53c8 | 路由异步读取由原Vector消费者获取完成 |
| ebb20cf | 预量化FP8只清padding，缓存复用保持完整relay尾部字节 |
| c461ac1 | B1沿用q全零特例，跳过空闲发送核的初始化 |
| fa771e3 | 保留原本支持的非对齐H，不缩小输入范围 |

原始提交顺序可用`git log --reverse --oneline 85d9cdc..HEAD`查看。
补充文档和后续验证修正以该日志为准。

## 保留的依赖

- 路由body完成后发布commit；出站commit完成后才能结束其源生命周期。
- 两次计数间覆盖statusBuf前的等待、计数/prefix生产完成和Reset前UB退休。
- 零计数发布、所有计数快照获取、flag清理及下一轮epoch依赖。
- combine prepare→ready→partial读取，现有`combine_reuse`完成协调。
- packet body→gate/flags；两个quiet阶段、GM slot和UB lane复用保护。
- NIC可写数据的L2绕过、注册GM源、SDK CQ/QP语义。

只读prefix合并不合并各消费者会清零的ready/sum矩阵。
共用prefix的保留容量、所有控制偏移、公开sendCountsOut保持原值。
mask期望数使用原pending高16位，token使用低16位；现有B<=65536和flag数量界
保证编码完整，M45布局和Host准入函数与基线完全一致。

q表复用原qDst/qSub作为有界histogram/hash，原qWork保存结果；不增加UB。
原首token的q=0特例保留。DIRECT链表保持原组轮转及组内任务次序。
最近token缓存没有改变发送次序；cache hit以V_S代替原输入等待，保护Scalar尾部改写。
预量化缓存还清理本包不再使用的旧relay条目，保持完整K槽metadata的原定义。

## 开销口径

对EP32、B8/M256、H3584、K8、E384、56 AIV/S28R28：

- 非receiver0死prefix读取减少41,472B/rank/dispatch；共享prefix写出再减少41,472B。
  两者都是本地GM搬运，不是跨机带宽。
- combine少写4,096B ready/rank（含self，其中peer为3,840B），少清4,096B本地prepare尾部。
- combine结果初始化只写61,440B flags，原完整两bank清零为983,040B。
  现在使用独立`combine_clear_result_flags_kernel`，在已有Host协调的准备阶段执行；
  它不属于`combine_server_dedup_kernel`，比较API时应包含其成本。
- 预量化输入每次装载仅初始化144B padding，原为3,840B；有效输入与scale由MTE2写入。
- B8仅8个发送核拥有token，每核最多一个跨机packet，batch没有摊薄该形状的quiet。
  唯一peer缓存主要减少大B的重复Scalar比较，不增加对B8的收益承诺。

这些是逻辑工作量；共享prefix读取竞争、hash查找、缓存命中率与flag初始化kernel
启动成本需要目标机测量。timer ON中部分等待归属改变，不能直接把阶段变短当作OFF收益。

## 目标机步骤

两机工程绝对路径一致，在各自工程根运行。替换示例IP为节点0真实IPv4。
匹配SDK无需`--rebuild-shmem`；本次完整wheel重建不要使用`--skip-build`。

```bash
ndr_overhead_run() {
    git pull --ff-only || return
    bash tests/ll_rdma/one_click.sh "$1" 192.0.2.10 ndr-overhead-001 --mode perf
}
# 节点0使用0；节点1使用1。失败保留当前交互shell和原日志。
if ndr_overhead_run 0; then
    printf '%s\n' 'NDR test completed'
else
    printf '%s\n' 'NDR test failed; inspect project logs'
fi
```

默认覆盖BF16和预量化FP8的dispatch+combine（FP8配置的combine仍是BF16），
保持10预热、100采集、逐kernel最后50条、Exp+scalar all_reduce。
正确性与重复调用检查仍执行。额外覆盖B1/2/4/8/16/31/32及M32/M256时分别更换run名，
使用已有`--tokens`、`--capacity`、`--route`参数；两机参数必须一致。

性能比较固定同一SDK、shape、dtype、路由和采样配置，对照基线`85d9cdc`与最终提交的
完整OFF kernel和API数据。阶段诊断单独用timer模式，不能混入OFF统计。
无需回传全部原始数据；原始数据保留目标机，回传文件仍每个不超过65535字节、总数不超过10。

本地CPU协议/布局/调度模型不能替代CANN编译、NPU精度或性能验收。
