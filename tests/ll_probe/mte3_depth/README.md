# 单AIV MTE3写请求排队诊断

本探针测量 **UB→本卡GM** 的MTE3写指令发射与完成等待。每次纯AIV、blockDim=1，
设备0/1逐卡顺序、单进程执行，不建立通信域。它不测跨卡/跨机写，不直接测底层OTS。

## 测量方法

- 计时前将128KiB输入从GM加载到UB，等待MTE2_S并建立MTE2_MTE3依赖。
- N=0..32编译期展开；每个copy读取独立UB区间，写入独立GM区间，stride≥chunk。
- batch连续提交后一次MTE3_S等待；serial每次提交后等待，作为串行对照。
- 分别记录issue/drain/total的GetSystemCycle原始ticks。serial的issue包含逐次等待。
- 可选Vector gate在独立32KiB UB上执行有界Duplicate，并让MTE3等待V_MTE3事件1。
  Vector自主完成，不依赖后续Scalar提交或MTE3释放，不存在循环等待。
- Host每轮更新输入pattern，检查整个GM分配：每条写的数据、未写slot、请求间隙及双侧64B guard。

默认预热10次、采集100次、按每个配置统计最后50条P50/P90/P99。
ACL Event单独记录，包含输入预加载和gate等整核开销，不能当作纯MTE3时间；不将ticks转换微秒。
此专项微架构诊断不含Exp/all_reduce，不替代正式LL的profiling协议。

## 构建与运行

先加载当前机器已有CANN环境，或显式配置CANN_ENV；不需要新建SHMEM SDK或安装wheel。

```bash
bash tests/ll_probe/mte3_depth/build.sh
bash tests/ll_probe/mte3_depth/run_a5.sh \
  --devices 0,1 --requests 0,1,4,8,12,13,14,15,16,17,18,19,20,24,32 \
  --chunk 128,4096 --stride 65536 --mode batch,serial \
  --warmup 10 --iters 100 --tail 50 --output-dir results/mte3-default-new
```

门控对照：

```bash
MTE3_DEPTH_BYPASS=1 MTE3_DEPTH_GATE_REPEATS=64 \
  bash tests/ll_probe/mte3_depth/build.sh
bash tests/ll_probe/mte3_depth/run_a5.sh \
  --requests 0,1,4,8,12,13,14,15,16,17,18,19,20,24,28,30,31,32 \
  --chunk 128 --mode batch --output-dir results/mte3-gate64-new
```

gate重复数支持0/16/32/64/128；0关闭。bypass=1仅设置目标GM的CACHE_MODE_DISABLE提示，
不是保证所有缓存均冷或所有写已落到DRAM。必须等待一次运行的两张卡全部完成后再重新构建。
runner固定四个设备可见性变量0,1；日志、缓存、临时数据和结果均放工程内。
从任意cwd调用脚本，相对路径始终从工程根解析。输出目录必须是工程内不存在的新目录。

输出包括每卡CSV/log、manifest、analysis/summary.csv/json；build/identity.json记录
方向、cache hint、gate参数、源码/二进制/反汇编指纹并复制到运行manifest。
build-evidence保存该次运行的源码和机器指令副本，后续重建不覆盖旧运行的证据。
build/kernel.dis保留生成指令，audit_asm.py核对关键batch窗口无逐请求完成等待。
原始CSV为：

```text
device,mode,requests,chunk_bytes,stride_bytes,iteration,issue_ticks,drain_ticks,total_ticks,event_us,verified
```

## 请求队列与OTS的区别

门控实验观察的是：MTE3执行暂时被阻住时，Scalar还能提交多少条写指令，以及第几条出现背压。
先确认N=0的gate剩余时间足够长，再比较不同gate长度及chunk大小的边界。
前置等待指令可能占用某一级队列项，参数准备和编译器控制流也会影响发射时间。

**有效排队容量不等于底层OTS。** 一条DataCopy可拆成多个总线事务，也可能被写合并；
总线/缓存/内存控制器的在途额度属于另一层资源。MTE3_S完成不提供这些资源的直接计数，
更不能由本卡写推导跨机UB Memory或RDMA的额度。结果只报告实际观察，不自动输出精确OTS。

统计回归：`python3 -m unittest discover -s tests/ll_probe/mte3_depth -p test_analysis.py -v`。
源码、脚本、指南通过Git交付；build、原始数据和生成报告不提交。A5观察与复现范围见VALIDATION.md。
