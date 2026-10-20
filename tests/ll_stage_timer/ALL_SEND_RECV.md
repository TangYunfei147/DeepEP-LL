# 全核发送、全核接收候选

分支`kernel/ll-all-send-recv`，工作区`DeepEP-LL-all-send-recv`，从`b3f9004`创建。该基线位于组内初发实验a192113之前，保留已提交的参考分段打点。原分支及其未提交改动未迁移或覆盖。

## 统一dispatch阶段

1. 全部56个AIV按专家连续区间统计并发布count，所有B使用同一分工。
2. 全部56核按token（去重）或展平路由（非去重）发送。源rank本组和其他组由同一发送逻辑处理，不再交给不同核池。
3. 每核排空自己的发送流水，所有56核执行SyncAll。空发送分片同样进入同步；OFF和ON都有此业务阶段边界。
4. 重置发送UB、恢复计数归约CTRL[60]=0，全部56核分片接收count、计算并发布前缀。
5. 全部56核按原输出行分片接收，FIRST_HOP继续由接收器按原协议relay，最终清理各自槽位。

G4仍是原有去重和packet协议的逻辑组；撤掉的是组内/组间发送核所有权区分，不删除去重或改变首跳目标。接收器与combine算法保持基线；dispatch前缀生产者扩大为56，求和/ready矩阵56×56，前缀副本仍56份。控制区按E≤1024检查，仍在combine固定1MiB控制起点之前。保留72个magic容量槽，实际启动56。

明确使用全核同步形成阶段边界，不能将本分支的发送和接收耗时解释为原共享接收重叠方案。跨rank不新增集合通信；该SyncAll仅同步本rank的AIV。全部56核准备前缀会增加控制矩阵开销，最终是否获益需目标机OFF A/B验证。

## 撤回与Scalar优化

撤回组内专家所有权计划、localTaskCount、routeIndex/expertQ描述符、localSendPool、组内发送前置、非去重本组过滤及相关Scalar/UB维护。没有保留ll_group_local_send.h或ProcessGroupLocal。

按新流程进行以下通用优化：

- 源mask在Scalar寄存器中累计，每token写回一次，代替DIRECT/FIRST_HOP逐包读取和回写UB mask。
- 权重向量Copy一次，只有FIRST_HOP选中位置改成1，DIRECT不再逐项GetValue/SetValue回写权重。未选中权重仍由mask定义为无效。
- DIRECT缓存仅token/k/expert三个uint32，q在实际发送时按不可变原始routeIndex计算，省去q缓存写入/读取。q归约次数和原槽语义不变；所有首跳仍先于缓存DIRECT下发。
- Classify预计算每组专家数，以一次除法取代expert/localExperts/G两次除法。
- q及小规模专家count的Duplicate+Sub改为Subs，去掉一次向量下发及中间屏障。
- count ready字段批量初始化，向量统计路径不再逐专家Scalar写固定ready值。

不引入新的Scalar直方图扫描、跨核q表或原子任务队列；未声称这些静态操作数减少等于目标设备性能提升。

## 核数、构建和身份

`DEEPEP_LL_SEND_CORES/DEEPEP_LL_RECV_CORES`继续控制combine的S/R，必须为正整数且和56；不影响本分支dispatch固定56发送/56前缀/56接收。不要设置56/56。Native配置和ST CSV分别输出`dispatch_send_cores`、`dispatch_prefix_cores`、`dispatch_recv_cores`，均为56，避免将combine默认28/28误当作dispatch配置。

在本分支根目录、已加载实际目标CANN且已准备工程内SDK的Python环境中：

```bash
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28
export DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

新clone先走标准`git submodule update --init --recursive --depth 1`、清除外部SHMEM/CATLASS覆盖、`python scripts/build_shmem.py`和build requirements流程。既有校验通过的SDK不因本次kernel变化强制重建。全rank安装本分支匹配完整wheel；核心标识11000+combine发送核数，implementation为all-send-recv，拒绝新旧Host/device混装。

## 打点与对比

本分支stage schema8，仍为64阶段、前两次实际区间+全量metrics、每核4608B。schema8的dispatch全部核role=send_and_recv；combine继续原S/R角色。schema7及更早数据仅离线读取，不能把旧role解释成全核新数据。新同步计入KERNEL并出现在SEND_DRAIN，计时边界本身不添加新屏障；结束KERNEL后的诊断SyncAll/Flush保持原设计。

比较b3f9004（发送与前缀/接收重叠）或a192113（R核本组初发）时，固定输入、S/R（用于combine）、G、shape、dtype、seed、Exp和采样参数，使用各自完整wheel和独立run-name。正常采样仍10预热、100采集、每轮Exp+scalar all_reduce，每完整task最后50条；profiler/Event/API分开。ON仅作诊断，不替代OFF基线。

compare.py识别两轮已知身份；新旧dispatch逐核工作分配变化时不输出逐核加速比。完整kernel perf可比较。同一全核分支内改变combine S/R时，不误判dispatch角色改变。

先检查B=1/8/32/48和较大合法B、K=1/8/32、无去重/去重、热点、同组及跨组、多次调用双窗口复用。默认CPU模型使用的规模只用于验证，不是kernel特化条件。原生EP准入和设备可见性规则不变，本地不启动NPU。

## 验证边界

Python模型检查路由覆盖、延后q与原前缀扫描一致、源mask/权重语义和schema7/8离线解析。C++检查包含实际Host布局和从真实kernel提取的Process多线程模拟，用于验证空发送分片仍参与阶段同步；本机缺编译器时跳过。这些检查不模拟设备MTE或NPU内存顺序，不能替代CANN编译及目标机正确性/性能验证。

本次本地回归：334 passed、31项因缺C++编译器跳过、9项Bash/POSIX检查未选。没有执行CANN编译或NPU测试。
