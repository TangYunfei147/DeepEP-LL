# NDR combine 重复调用的 HCCL 完成协调

## B1→B2小B去重窗口复用修复（Host协议2）

combine partial暂存紧随B*K个结果包；B增大时旧小B的stage可能成为新结果槽。
H3584/K8下B1 stage起点正是B2 token1 slot0起点。M5扫描所有K槽标志，
非活跃slot旧BF16数据也参与求和，导致等待超时。CPU已用合法BF16值复现。
core29/56在B2下负责源token1，与用户case1的现象相符，设备最终原因仍需重测确认。

每次dispatch使runtime的combine窗口准备缓存失效。首次/布局变化的NDR combine
在全设备同步之后、HCCL状态交换之前，通过新native入口同步清零两个bank的
B*K结果区（包含payload/flags，不涉及dispatch数据区及计数控制区）。所有rank
准备成功后才启动combine，避免清零覆盖另一rank的新发布。同布局连续combine
继续依赖原消费清flag，不反复清零。准备失败随HCCL交换并封锁runtime。
Python Host协议升2，旧native缺入口明确拒绝；必须全rank完整重装wheel。
通信kernel ABI7、去重条件、SHMEM均不变。增加的首次准备成本属于Host/API，
不计入combine kernel task。仍需目标机B1→B2/B31→B32等变shape及多轮验证。

2026-10-22：日志在 `combine.warmup` 首次报告设备507035，随后其他rank的scalar
all_reduce超时。Profiler在WARMUP退出的警告是失败后采样不完整，不能当作根因。
日志没有设备fault kernel/PC，尚不能排除combine内部其他错误。

## 本次改动

沿用NDR初始化/销毁的现有HCCL Host协调，在`ElasticBuffer._combine_low_latency`
每次提交前、持有runtime.notify_lock时执行`combine_reuse`完成交换：

1. 等待本卡全部设备stream完成（包括上一次combine及窗口清理）。
2. 在已有HCCL组交换操作名和错误状态，确认所有rank上一轮完成。
3. native只确认Host协调，不启动设备barrier、不释放内存。
4. 创建producer event并按原路径启动combine。

固定ready标志及双窗口轮转没有跨rank的窗口复用信用确认。原正确性检查逐轮有
completion_boundary，但warmup连续提交；快rank可能再次使用慢rank尚未退休的窗口。
本次通过每次调用前的全域完成边界消除这种跨轮复用竞争，不改变当前轮的RDMA回传、
partial归约或source聚合。它是针对重复调用的修复候选，不等于已证明507035唯一根因。

协调失败封锁runtime，native launch失败同时设置notify_failed/coordination_failed。
combine_reuse与barrier/close有不同操作标签，错序明确失败。
普通MTE对照不新增协调；NDR单机EP16也遵守相同Host调用契约。
所有rank需同序调用，禁止跨线程并发提交。低层native接口不能替代公开Buffer契约。

ST capability新增已安装ElasticBuffer的combine_host_protocol=1，拒绝同一作业内
新旧Python入口混用。无需改kernel ABI或SHMEM，不重建SDK；各节点安装匹配完整wheel。

## 测量影响

关闭/开启timer、warmup、正式采样和capture都通过同一Buffer入口，因此均有协调。
10预热/100采样/最后50及Exp+scalar all_reduce规则不变；HCCL可能增加CPU/API时延，
也会改变调用间隔和重叠。Profiler只按原combine kernel task统计，不能把其耗时称为
包含Host协调的端到端时延。新结果必须保留combine_host_protocol及软件版本，不与旧
连续提交结果无条件等同比较。设备打点不包含Host协调时间。

## 更新与验证

两台机器更新代码、激活原测试环境、source tests/ll_rdma/env.sh，安装完整wheel：

```bash
git pull --ff-only
source tests/ll_rdma/env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

使用新run-name，先B1、M32、H3584、K8、E384、fp8运行正确性，再保留相同shape
运行完整warmup/perf，增加`--trace-ops`。日志应每次combine前出现
`HCCL combine_reuse completion exchange enter`和`complete`。
若仍报507035，提供最早设备plog的fault kernel、core、PC、trap/timeout，以及两机日志；
不要只扩大HCCL超时或跳过正确性。

CPU回归执行生产combine入口及Host协调方法的故障替身，覆盖提交前顺序、五次重复调用、
peer/HCCL错误、错序、失败后拒绝重入及MTE路径不新增屏障。没有CANN或NPU验收。
