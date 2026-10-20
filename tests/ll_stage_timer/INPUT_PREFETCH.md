# 两份输入缓冲的一包前瞻预取

分支 `kernel/ll-g4-shared-recv`，A/B 基线 `b33119343b46e13a65afac08b499c52c76cfb410`。
仅改变发送输入调度；保留schema9打点、静态共享接收、组内/组间分工。

## 调度与缓冲

原 FIRST_HOP/DIRECT/本组任务生成器保持原顺序，不建立第二份任务列表。
`SendPacket` 现在提交一份按值保存的5-word描述符：

1. 首包搬入、入队并出队，成为当前包。
2. 获得下一描述符后，先用第二份输入缓冲下发下一包，保持它在队列中。
3. 量化/填写尾部/打包并发送当前包，释放当前输入。
4. 下一包才出队成为当前包，避免它的MTE2→V等待提前阻塞当前包计算。
5. 生成器结束后显式flush最后一个包，再执行原输出lane排空。零包flush为空操作。

一个输入已出队、一个输入在队列内：TQue逻辑深度仍为1，物理输入缓冲仍为2。
输出lane仍由原 `min(8, available)` 公式决定，所有显式UB张量分配大小不变。
新增描述符、LocalTensor引用和事件号为控制状态，不增加packet UB/GM缓冲；
编译器私有寄存器/栈开销仍需目标CANN确认。

## 同步安全

- TQue EnQue/DeQue保持MTE2→V先写后读依赖；下一包DeQue在当前包处理之后。
- 普通/预量化输入一直持有到两条Copy打包指令下发后，才FreeTensor；
  下次AllocTensor通过TQue的V→MTE2事件等待旧读完成。
- 在线量化输入在量化读完成对应的释放点FreeTensor，保留原量化末尾V_S；
  单份quantBuf与输出lane同步顺序保持原实现。
- 普通/预量化模式使用AllocEventID申请两枚MTE2_S事件。每包输入数据和
  scale DMA之后SetFlag；消费只Wait该包的事件。不在消费处新SetFlag，
  否则会把已排在MTE2后面的下一包也等完。事件交替复用前已有匹配Wait；
  最终flush后ReleaseEventID，再进入原drain/TPipe Reset。
- 不使用FetchEventID保存跨调用在途事件，不抢占其他使用者的临时事件号。
- 预量化模式对两份输入缓冲在PREPARE_INPUT一次清零，避免在预取热路径
  先做Duplicate/V→MTE2。每次覆写有效payload/scale/头部和有效尾部项；
  未覆盖的padding/无效尾项不参与消费。普通BF16仍沿用原padding语义。

实现依据公开的
[TQue生命周期](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/tpipe_tque_principles.md)
及 [AllocEventID](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/AllocEventID.md)/
[ReleaseEventID](https://gitcode.com/cann/asc-devkit/blob/9.2.0/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/ReleaseEventID.md)。
源码参考固定9.2.0提交4d9fe405，不修改SDK。

## 打点与收益解释

保留所有阶段ID、累计cycles/calls和前两次区间。TOKEN_INPUT现在计时输入
申请/下发/入队；DeQue移动到前一包处理之后，不能将新旧TOKEN_INPUT区间
直接解释为DMA时延之比。每包仍只记录一次TOKEN_INPUT。
TOKEN_INPUT_WAIT记录该包独立完成事件的剩余等待；TOKEN_QUANT_WAIT保留
原V_S位置。所有包包括最终flush都执行原量化/元数据/打包/发送打点。
最终包flush在原SEND_DRAIN外，避免把计算伪记成排空时间。

预取会将等待转移到任务生成、队列操作等位置，原瓶颈报告只作线索；
报告变小不能独立证明收益。FIRST_HOP实际发出会等待下一描述符生成，
所以序列启动、阶段边界和单包case可能没有收益甚至变慢。预量化路径
还包含清零调度变化，不能把总收益全部归因于MTE2延迟隐藏。

## 验证和A/B

全rank安装同一提交、同S/R/G的完整wheel；核心协议10000+S和schema9不变，
不因本改动强制重建已校验SDK。与基线使用同路由seed、shape、dtype及
Exp/采样协议，保存独立结果目录，使用timer OFF的总dispatch时延判断收益。
ON诊断沿用README和INPUT_WAIT.md，至少3轮用于自动候选判断。

在工程根目录运行：

```bash
python -m pytest tests/elastic/test_ll_input_prefetch.py \
  tests/elastic/test_ll_group_local_send.py tests/ll_stage_timer/test_input_wait.py \
  tests/ll_stage_timer/test_segment_source.py -q
```

覆盖空序列、单包、奇偶长度、跨8-lane边界、重复token及不同包尾；
实际生产调度器另有C++ mock验证按值保存描述符、队列占用和最终flush。
CPU生命周期模型不能验证NPU事件内存序、TPipe调度和性能。
本次完整本地回归491 passed、45 skipped、9 deselected；跳过项为缺C++
编译器或POSIX环境，未进行CANN编译或NPU执行。
目标机必须验证BF16/FP16、预量化FP8、在线量化FP8、FIRST_HOP/terminal、
同组/跨组、空发送分片、重复窗口及合法S/R配置。
