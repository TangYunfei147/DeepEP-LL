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

全rank重建安装同一提交的完整wheel，无需仅因本改动重建已校验SDK。
G/S/R、包格式、通信协议10000+S和每核UB分配保持一致。
先验证FIRST_HOP/terminal/直接包、零接收、重复窗口、FP8/BF16和合法S/R；
再用相同seed/shape/采样协议、timer OFF、独立结果目录逐提交比较。
当前没有CANN/NPU验证或性能结论。CPU模型不替代设备事件时序验收。
