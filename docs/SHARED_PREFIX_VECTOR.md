# R核计数、共享前缀及接收任务向量化

## 提交1：R核承担全部源端计数，UB预算230KiB

S核不再扫描/统计路由或发布count，仅保留发送scratch初始化。
R核分别均分本组与远端专家：本组分工与原来一致，继续生成稳定q和本组发送描述符；
远端专家分配给同一批R核统计。每专家唯一生产者，包括零计数，协议不遗漏ready。
每核最多一个本组区间和两个远端区间，count记录连续，且本组区间在最前。
R核的count记录UB峰值、局部描述符保留与发送/接收布局预算使用同一分工公式。

`kEpServerDedupCountUbBytes=230*1024`，Host检查和Device布局共用。
不修改含义不同的`UB_SINGLE_DMA_SIZE_MAX`，不增加事件上限（overlap仍最多6个lane）。
此预算是用户指定目标配置；编译通过不代替目标设备UB容量及最大合法shape验证。
S+R仍56，combine分工不变。协议15000+S拒绝旧预算/计数分工wheel混装。
timer schema14保持；S核不再记录SEND_COUNTS，R核记录真实统计/提交范围。
S核提前提交的数据仍等待接收准备后消费，count数量/原始q不依赖payload先后顺序。

前置提交保留原56份前缀发布，以便独立对照计数迁移和UB预算的影响。
可能的负收益是R核计数更重、前缀发布更晚；需同时观察SEND_COUNTS、WAIT_RECV_COUNTS、
首次接收和OFF整体耗时，不能仅凭S核开始更早判定收益。

## 提交2：单份共享GM前缀，可选parallel/r0

编译环境变量`DEEPEP_LL_PREFIX_MODE=parallel`（默认）或`r0`。
这是构建选项，不是运行时开关。Python构建指纹、Host CMake、Device CMake、
直接编译检查同步传入；全rank必须重装同模式完整wheel。
parallel协议16000+S，r0协议17000+S，Host/device及ST全rank检查拒绝混装。
ST结果记录prefix_mode；可跨模式比较，但保留各自模式，不自动调整。

parallel：R核按8条记录/32B整块均分前缀，避免不同生产核写同一GM块。
每核计算局部前缀、汇总片段总数、加偏移后只写共享表一次。
active生产核可能少于R（小E场景），非生产R核仍参与源端计数、本组发送和接收。
r0：原接收组编号0（物理核S）独自收集全部count、生成整表；
其他R核不收集目标count或生成前缀，只读共享表；所有R核仍做源端SEND_COUNTS。
跳过跨R核片段总数发布/轮询。R0最多1024条count，超过255时使用压紧ready后长计数Sum，
不向基础ReduceSum的8位repeat参数传入溢出长度。

两种模式均只保留一份GM前缀，全部56核各自加载到私有UB。
ready仍为每消费核独占的一行，消费者只清自己的ready；保持原发布顺序和源缓冲复用依赖。
没有新增全核barrier。串行fallback也改读同一张表，不能保留旧副本偏移。
共享表在本次kernel消费者完成前不复写；仍遵守原双窗口magic和同stream调用序列。
并行模式减少MTE3发布工作，R0同时减少片段汇总协调，但可能集中形成慢核；
性能必须由目标机两模式OFF对照确认。

打点schema14/ID不变，BROADCAST_PREFIX现在表示共享片段发布；r0不产生GET_CUMSUM等
跨生产核汇总阶段。未出现的阶段不得按旧算法解释为采集丢失。

```bash
export DEEPEP_LL_PREFIX_MODE=parallel  # 或 r0
export DEEPEP_NPU_ARCH=Ascend950 DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

保持原S/R配置、SDK及ST参数；每模式用不同run-name，OFF正式性能和ON诊断分开。
