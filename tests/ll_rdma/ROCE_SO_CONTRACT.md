# 目标RoCE：WQE尾片SO与尾FLAG发布

## 来源与范围

2026-10-27，用户在本项目明确给出目标硬件契约：

> 跨机走ROCE的时候，每个WQE的尾片中存在so特性，so能保证该so前面的数据先到达，且片内保序，单片默认2KB。

本项目后续设计采用这一目标RoCE契约。本次没有执行硬件验证，也没有修改SDK、kernel、
wire布局或通信协议；该记录不能作为新方案已经实现或性能改善的证据。
2KB描述网络分片，完整WQE消息可以跨多个分片。它与当前每512B wire块中的480B body
和32B flag是不同层次的概念。

## 后续采用的发布方向

一个消息的有效数据和必要元数据按连续布局放在前面，末尾增加一个32B FLAG：

```text
一个WQE：[data + metadata + 必要padding][32B FLAG]
            可跨多个默认2KB分片         尾部就绪标记
```

按上述SO语义，接收端观察到尾FLAG时，受该尾片SO及片内顺序覆盖的消息前部已到达。
因此，不必仅为body先于FLAG到达，固定采用两次put并在中间插入quiet。
这个方向不同于“整个消息一次发出，首FLAG为0；quiet后另发首FLAG”的现有门控实现。

实现时须落实以下具体契约，而非简单删除现有调用：

1. 实际SDK产生的WQE、分片及SO覆盖范围与消息边界对应。不得把任意一次API调用
   无条件等同于一个WQE，也不推导不同QP或独立WQE之间的全局顺序。
2. 32B FLAG处于消息末尾，满足布局、对齐及尾片可见性要求；在接收者读取FLAG前，
   数据/metadata已由生产者写入稳定的注册源GM。
3. 接收端检查新的尾FLAG，并通过既有缓存绕过/失效与流水依赖读取数据。不能继续
   使用旧“逐512B块flag归约”的判断而只改发送长度。
4. 到达保证与缓冲复用分开：NIC仍可能读取源GM时不能覆盖；相应CQ完成/quiet或
   等价生命周期证明仍需成立。接收FLAG可见也不代表接收者已经消费或释放窗口。
5. 旧FLAG清理或epoch识别、零消息、同stream顺序及窗口复用分别证明。变更布局时
   同步Host容量、sender/receiver和协议版本，全部rank使用匹配wheel。

## 当前代码仍是什么

- dispatch主跨机路径：`DispatchRdmaBatch::Enqueue/Flush`，完整packet首FLAG置0后put；
  body quiet后另put首32B FLAG，再quiet保护源暂存复用。
- `LlTransport::Packet`的跨机路径：按480B body/32B flag分段，两阶段put和quiet；
  当前跨机combine仍调用该路径。
- 两者接收端均沿用既有packet flags。它们尚未改成这里描述的连续消息尾32B FLAG。

后续性能判断需要比较完整OFF task及正确性。减少WQE、独立flag写和中间完成等待
是可分析的工作量变化；不能仅据代码结构给出已达成的加速比。
