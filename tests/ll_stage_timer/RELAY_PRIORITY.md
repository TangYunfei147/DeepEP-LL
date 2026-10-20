# Relay 提前下发与同批 FIRST_HOP 优先

独立clone：DeepEP-LL-g4-relay-priority，分支kernel/ll-g4-relay-priority。
基线为远端共享接收分支ef5d40cb3a10744776e750d10305c8b1198fb9af。
不读取或复制其他agent的未提交修改，不改其工作区。

## 第一步：relay提前下发

FIRST_HOP仍先快照本地type/destination，再将转发header改为terminal并更新
compact marker。沿用原S→MTE3及MTE2→MTE3依赖后，先下发所有relay写，
再写本地payload/scale、type/destination、relayReadIndex。
wire在第一条relay写之后不再修改；原lane释放事件仍位于全部写之后。
没有新缓冲、屏障、包数或字节数变化；非FIRST_HOP路径不改变。

这是同一MTE3流中的顺序优化，不意味着新增物理通路或保证远端已消费。
它可能提前触发下游工作，也可能延后本地输出；必须以完整dispatch时延判断。
保留schema9打点，relay仍包含在原RECV_OUTPUT包络内，未把下发时间称为链路时长。

## 验证与A/B

第一步提交`ce516ca`，相关CPU检查33 passed。与基线比较可单独判断提前relay的影响。

## 第二步：同批就绪FIRST_HOP优先

复用Probe已有readyMask/keepFlagsMask，不增加包头读取或UB缓冲。
当前协议只有FIRST_HOP首发标记KeepFlags，relay发布前已改为Compact；
消费时仍以真实header判断是否转发，布局提示不替代包类型检查。
仅对已就绪位分两类，各类保留原倒序，先FIRST_HOP后普通包。
最多16个位置编码为一个uint64顺序表，加一个计数，不分配新UB tile。

预取与消费使用同一顺序表，跨批issued/consumed lane环不重置。
整批全部消费下发后，才按原snapshot下标倒序RemovePending，避免
swap-with-last破坏尚未处理的位置；不按优先级顺序删除pending。
不等未就绪FIRST_HOP，本批就绪普通包也全部处理，未引入优先级自旋。
非去重时不使用FIRST_HOP优先标记，保留原就绪包倒序。

此策略不扫描其他核或未来批次，不能解决静态分片的所有负载不均。
多做一个固定小位图调度步骤，收益须与Scalar开销一并测量。
lane释放、最终drain和flag清理不变；schema9阶段仍保留，
RECV_INPUT/RECV_COPY记录顺序会随真实消费顺序变化。

第二步与`ce516ca`比较以分离优先级效果，再与`ef5d40c`比较两步合并效果。

## 部署

全rank重建安装同一提交的完整wheel，无需仅因本改动重建已校验SDK。
G/S/R、包格式、通信协议10000+S和每核UB分配保持一致。
先验证FIRST_HOP/terminal/直接包、零接收、重复窗口、FP8/BF16和合法S/R；
再用相同seed/shape/采样协议、timer OFF、独立结果目录逐提交比较。
当前没有CANN/NPU验证或性能结论。CPU模型不替代设备事件时序验收。

两步合并后的完整本地回归：519 passed、46 skipped、9 deselected。
跳过项为缺少C++编译器或POSIX环境；实际优先级函数的C++穷举测试已提供，
本机未执行。批内模型覆盖深度1/2/4/8、尾批、lane环回绕和pending压缩。
