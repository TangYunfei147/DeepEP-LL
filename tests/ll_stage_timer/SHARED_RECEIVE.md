> 2026-10-20本组初发更新：原R核先发送本源rank的组内包，再运行原接收流程；核心标识10000+S。当前调度/UB/验收以[GROUP_LOCAL_SEND.md](GROUP_LOCAL_SEND.md)为准。

# Dispatch发送后参与接收

分支：`kernel/ll-g4-shared-recv`。起点：`kernel/ll-g4-stagger@5c2ea11`，包含父分支已有统计更新。默认G4、MTE和错峰发送顺序保留。

## 执行过程

总AIV固定56，编译配置仍为正整数S+R=56，默认28+28。

- 前S核：发送计数 → 发送自身全部token/首跳/DIRECT → 等待自身前缀ready行 → 接收、输出、必要的relay → 清理自身槽位。
- 后R核：发送计数 → 等待计数、计算/发布前缀 → 等待自身前缀ready行 → 接收、输出、必要的relay → 清理自身槽位。
- 全部56核按输出行静态分片：核c拥有`[floor(c*T/56), floor((c+1)*T/56))`，T为本rank实际接收行数。任务、输出和flag清理唯一归属，空片直接返回。
- 各核完成自己的发送或前缀生产后就能进入接收，没有新增全核发送完成屏障。这不是动态抢任务：先进入接收的核不会接管尚在发送的核所拥有的区间。
- FIRST_HOP仍优先、接收仍可转发。源端发送等待MTE完成，不等待目标消费确认；因此软件依赖不要求所有receiver先启动。但目标机仍须验证传输进展、重复调用和relay场景。
- combine执行方案不变，仍按S/R分工。全rank必须使用新分支同一配置的完整wheel。

## 控制区及UB生命周期

前缀生产者仍为R核（实际非空生产者min(R,E)）；求和矩阵R×R。完整前缀扩展为56份，ready矩阵为56×R，每个物理核独占一行。生产核先发布所有副本中的自身片段，再发布ready；消费核只清自己的ready行，允许其他核晚到。双窗口复用依赖原有同流、同communicator调用次序，禁止同一communicator重叠调用。

原发送核读取本核副本；原接收核的副本索引也统一改为物理核号。CountPublish前先排空本核流水，再Reset TPipe回收前段UB；Reset后重新设置CTRL[60]=0以保留计数ready的FP32 subnormal归约行为，结束时恢复调用前状态。CountPublish写回完成后，Receiver才重置UB加载接收任务。

Host布局和接收UB预算同步按56参与者计算；控制区测试覆盖全部55种正整数S/R与E≤1024，保持在combine固定1MiB控制起点之前。前缀副本及ready写入比父分支增加，不能预设本候选必然更快。

## 构建与检查

在目标机该分支checkout根目录、已有匹配CANN/Python及已校验工程内SDK的环境中：

```bash
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=10
export DEEPEP_LL_RECV_CORES=46
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

这表示前段10核发送、46核前缀/接收，发送完成后56核都接收。未设置两个变量时默认28/28。仅改变运行环境变量不会改变已安装wheel。所有rank重装完整wheel；本改动不要求强制重建已校验SHMEM。新clone需按[LL_KERNEL.md](../../LL_KERNEL.md)初始化子仓与构建SDK。

原生`low_latency_core_config()`输出`send_cores=S`、`recv_cores=R`、`dispatch_recv_cores=56`。当前核心标识为`10000+S`、实现身份g4-shared-recv、阶段schema7；全rank重装匹配完整wheel。打点已迁移到参考分段逻辑，见[SEGMENT_TIMER.md](SEGMENT_TIMER.md)，旧schema6记录仅离线解析。

## 诊断与A/B

继续使用`tests/ll_stage_timer/run.py`及当前目标环境的显式设备/多机配置。不要在本地或A5双设备限制环境启动原生LL，也不伪造EP。

先验证正确性：B=1/2/4/8/16/32/48；相同M/H/K/E/dtype/seed；覆盖无去重及去重路径、同组多专家、跨组、热点、空输入和重复调用。选择合法配置，不能为了测试放宽原生EP准入或Host容量保护。验证不对称10/46与默认28/28；确认dispatch输出、路由元数据以及后续combine结果一致。

随后以父分支`5c2ea11`为基线，固定S/R/G/shape/输入/设备/Exp，使用各自匹配OFF wheel及不同run-name。沿用正式10预热、100采集、每轮Exp+scalar all_reduce、每完整task最后50条；profiler/Event/API分别报告。不要把打点ON的耗时当作OFF性能基线。

独立ON诊断中，原发送核标为send_and_recv，检查其LOCAL_WINDOW_COPY/RECV_COPY/RECV_OUTPUT记录；前缀等待为WAIT_CUMSUM。参考阶段没有独立relay字段，relay包含在RECV_OUTPUT。缺任务核可以没有消费记录。

新旧打点边界不同，跨schema6/7只允许匹配条件的OFF正式性能对照，不计算诊断耗时比。相同schema内若dispatch逐核任务划分改变，compare.py保留work_partition_changed提示，缺失阶段不补零。

## 共享接收首次交付验证（6ea61a6，打点迁移前）

阶段/专家解析、异步前缀模型、双窗口复用、全部S/R控制区边界、构建/打包及统计回归：256 passed、28因缺C++编译器跳过、9项Bash/POSIX检查未选。另Host/ST约束29 passed。新增共享接收检查17项包含在上述256项中。

没有执行CANN编译、NPU正确性或目标性能测试。Python模型验证协议分工与覆盖，不模拟设备内存顺序或MTE进展。源端转接收的最终可运行性和性能以目标环境全rank验证为准。
