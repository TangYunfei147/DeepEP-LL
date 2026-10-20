> 当前演进分支为`kernel/ll-g4-local-send-dynamic-recv`：保留组内初发，接收改为就绪批次动态认领，协议11000+S。以[DYNAMIC_RECEIVE.md](DYNAMIC_RECEIVE.md)为准；下文静态接收说明为父版本记录。

# 原接收核承担本组初发

当前分支：`kernel/ll-g4-shared-recv`；本次源端改动基线：`b3f9004`（保留参考分段打点）。默认G4，编译配置S+R=56，默认S/R=28/28；按实际G/EP/shape计算，不按示例规模特化。

## 调度

- S核：统计并发布本源rank指向其他组的专家count，执行跨组DIRECT/FIRST_HOP，再按原共享接收流程参与接收。
- R核：统计并发布本源rank指向本组的专家count，同时生成稳定本组任务；完成本核本组初发后，调用原来的RecvCnt、CalcCumsum、CompleteCumsum和Receiver。
- 本组包括同卡目标；跨组去重首跳仍由S核发出，目标组内relay仍由原Receiver执行。本次未实现模型中的独立C核池，也没有在轮询失败后临时找任务。
- 组内初发是有界的前置工作：R核不是等全体组内发送结束才进入前缀，而是各自发送完就推进原流程；这可能推迟部分前缀/relay，需要目标测量。

接收器源码、接收侧count链、packet格式、GM窗口、最终56核接收分片及combine算法保持原样。改变的是源端专家count和初发包的所有者；所有专家每个source仍发布且只发布一份count，包括0。

## 减少Scalar开销

1. 每个本组专家唯一归属一个R核。向量mask筛选该核专家范围，CreateVecIndex+稳定GatherMask保留原始展平routeIndex。
2. 在现有直方图累计中直接得到q：`q=histogram[expert]`，再累计count。保存两列各uint32的routeIndex与`expert|(q<<16)`；q沿用原始路由槽语义，首token遵循原0约定。
3. R核发送直接遍历紧凑描述符，不再执行逐任务组判断、Classify或CalTokenSendExpertCnt前缀归约。
4. S核Classify直接分离本组mask，缓存任务只含远端DIRECT。源mask和权重仍由原token所有者S核统一写；权重批量Copy，DIRECT不再逐项回写mask/weight，FIRST_HOP保留原选中位置的1.0语义。
5. 非去重路径同样向量过滤本组路由，再遍历紧凑远端索引。向量读取从32B对齐位置开始，边界mask排除所属区间前的条目，保留原始q及远端顺序。

这减少重复前缀计算，但仍有描述符写入、索引和数据搬入等固定成本；没有宣称Scalar开销为零或OFF性能提升。

## Count与UB

`common/ll_group_local_send.h`统一Host/device的专家分工：R核切分本组专家区间，S核切分删除本组后的远端专家序列。远端分片最多映射成两个实际专家区间，连续count记录带record偏移，DMA仍写原来的目标专家/来源rank槽。

status记录和两列本组描述符独立保留在scratch之外。count临时池与本组发送池通过共享模式复用stagePool地址；发送池有独立分配游标，不消耗后续RecvCnt的游标。所有发送lane及MTE读取退休后Reset本组发送池，并重设CTRL[60]=0再运行原计数归约。无本组任务时直接跳过发送池准备。

Host同时校验新的count峰值、本组发送和跨组发送的UB预算。热点下本核任务可能达到完整B×K，描述符按最坏情况分配；不能靠均匀路由低估容量。新增暂存可能使部分原来贴近UB上限的shape被明确拒绝，不静默减少lane到0、不越界或改协议。旧按B选择count参与者的常量仅保留给历史提取探针，生产源端count改用本次分工。

## 构建、诊断和A/B

所有rank在本分支安装相同完整wheel，S/R一致：

```bash
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28
export DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

可使用任意合法正整数S/R且和56，不依据B256/H3584/K8/E384等示例特化；无需强制重建已校验SDK，新clone仍按标准子仓/SDK流程准备。核心标识改为10000+S，以拒绝新device与旧Host UB预算混装。参考分段schema7不变，接收端wire协议不变。

A/B基线为b3f9004，保持S/R/G、dtype、实际EP、路由种子、shape、Exp和采样参数相同。沿用现有ST入口，使用不同run-name，先正确性和窗口复用，再按10预热/100采样/每task最后50条统计OFF性能。用户示例可以作为一个case，不能代替小B、非去重、热点、跨组、同组及重复调用覆盖。

ON诊断中，原R核的worker标签仍为cumsum，但会出现SEND_TOKENS和对应TOKEN_*实际组内发送区间；之后才是CAL_CUMSUM/LOCAL_WINDOW_COPY。本组q计算已融合到COUNT_COMPUTE，不应在R核看到逐包CAL_TOKEN。compare.py识别9000/10000两种已知schema7身份；dispatch逐核源端工作分配变化时不输出逐核加速比。

源码指纹测试仅更新三个获授权变更的源端文件；接收器和combine指纹仍对照原始记录。原始6ea61a6指纹不再用于宣称本次源端算法无变化。

## 验证边界

本地源端路由/q/metadata模型覆盖随机、本组热点、纯远端、G=1/4/16及S=1/10/28/55；非去重覆盖非对齐分片。另提供C++实际计划/Host UB检查，供有g++的环境执行。接收器文件及从RecvCnt开始的接收侧count链已与修改前逐字核对不变。

本机无C++编译器/CANN/NPU：CPU模型和Python检查不代表原生编译、设备精度、MTE进展或性能通过。最终统计写入本次交付说明。

本次CPU回归：349 passed、42项缺C++编译器跳过、9项Bash/POSIX检查未选；新增路由模型37项通过，实际C++计划/UB检查12项包含在跳过项中。未进行CANN编译或设备测试。
