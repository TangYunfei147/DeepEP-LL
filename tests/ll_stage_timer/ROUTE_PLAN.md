# 共享路由准备候选

分支 `kernel/ll-g4-route-plan`，独立工作区 `DeepEP-LL-g4-route-plan`。
从共享接收 `ef5d40c` 创建，保留双输入缓冲的一包预取、schema9 等待打点、
G4 与编译 S+R=56；combine 和静态共享接收算法保持原实现。

## 第一个特性：并行准备一次，源端消费共享元数据

56 核按连续 token 分片读取原始 IDs，生成各自产生的局部 q、目标组掩码和
原策略首跳。相同 token 组的元数据只准备一次，原 FIRST_HOP 组顺序仍按
第二次命中的位置建立，保留已有错峰调度顺序，不把它改成首次出现顺序。

准备阶段有三次本卡全核边界，所有空分片核也必须参加：

1. 发布每个 token 分片的完整专家计数列；
2. 按专家再次分片，对56列作前缀，发布每个分片的专家基址及总计数；
3. 原 token 所有者补齐 q，发布 IDs、packed-q/首跳、组掩码后进入原 S/R 发送。

q 的定义是前面 token 分片的专家计数，加本分片内该路由之前的同专家条数。
保留首 token 的 q=0 约定。packed-q 的低16位是 q，高两字节分别存首跳
top-k 下标及组的第二次出现位置（单成员取首次）。现有 Host 路由数量约束
确保 q 和本地 token 序号适合16位；不放宽已有输入约束。

Count 发布不再各核扫描原始完整 IDs，读取预先汇总的总数。
R 核仍按原专家区间筛选自己的本组任务，但直接读取共享 q；
S 核读取组掩码和首跳描述，不再做逐路由前缀比较/归约。
接收端 RecvCnt 起的链条不改，跨rank count/packet 的写入协议不改。
原寄存器频数直方图候选没有在本分支的纯计数路径继续调用，因为本次统计
同时产生稳定 q；其文件和独立接口测试仍保留作历史实验材料。

## 存储、同步与代价

共享准备区只在本rank GM中读写，位于完整source-return区之后、dispatch
packet区之前；不复用可能被较快rank的combine提前写入的source-return区。
其大小为 `E*56*32 + E*32 + 3*align512(B*K*4)`，例如E384/B256/K8为708KiB。
Host/device共用布局移动packet起点，保留原source-return基址和长度。
较快rank的跨卡return写入不会覆盖该区域；本rank进入combine后可按原流程
复用其本地stage空间（本rank dispatch已经完成），不新增跨rank阶段屏障。

每核准备UB有显式上界检查；sender仍复用原q工作区，量化不覆盖packed-q。
本卡GM计数矩阵交换和三次 SyncAll 都有开销，尤其小token可能变慢。
首包启动晚于原分支；本候选没有实现按小批次发布的流水准备。
减少重复计算不等于已证明减少完整dispatch时延。

实现身份 `g4-route-plan`，核心协议 `13000+S`，防止旧布局Host/device混装；
schema9、66阶段、每核4608B打点载体保持。准备阶段归PREPARE_INPUT，
不新增逐路由计时。源码指纹只记录经授权修改，不是设备正确性证明。

## 复现

全rank从Git取得本分支，沿用工程的SDK校验和wheel构建流程，重装完整wheel；
不因本次更改强制重建已校验SDK。保留相同S/R/G、shape、seed及Exp负载，
将第一个特性与 `ef5d40c` 比较。仍用timer OFF的正式采样判断性能，
ON诊断辅助定位。输入等待报告仍可使用，但它不判断准备阶段的GM/同步成本。

```bash
python -m pytest tests/elastic/test_ll_route_plan.py tests/ll_stage_timer -q
```

正确性需覆盖小B和空核分片、S/R极端比例、混合本组/远端、同rank多个专家、
多轮窗口复用及dispatch+combine。CPU模型检查q/总数/顺序；实际C++布局测试
在有g++环境执行。未做CANN编译和NPU验证，不声称性能收益。

第一项本地完整回归：516 passed、48 skipped、9 deselected；新增实际准备器的
56线程mock在无g++的本机跳过，相关定向检查39 passed、3 skipped。
所有CANN编译、NPU正确性及性能结论仍待目标机验收。
