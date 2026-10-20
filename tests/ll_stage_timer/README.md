> 当前演进分支为`kernel/ll-g4-local-send-dynamic-recv`：保留组内初发，接收改为就绪批次动态认领，协议11000+S。以[DYNAMIC_RECEIVE.md](DYNAMIC_RECEIVE.md)为准；下文静态接收说明为父版本记录。

> 2026-10-20本组初发更新：原R核先发送本源rank的组内包，再运行原接收流程；核心标识10000+S。当前调度/UB/验收以[GROUP_LOCAL_SEND.md](GROUP_LOCAL_SEND.md)为准。

# LL参考分段打点

当前`kernel/ll-g4-shared-recv`采用graph_dc_debug_seg的采样区间+全量累计逻辑。每阶段前两次调用保留真实时间线，所有调用累计；INIT单列，KERNEL结束后全核同步再写出。详细位置、格式和限制见[SEGMENT_TIMER.md](SEGMENT_TIMER.md)。

## 构建

先加载目标机实际CANN环境；所有操作在本分支工程根目录、工程内Python环境进行：

```bash
git submodule update --init --recursive --depth 1
source scripts/ll_timer_env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
export DEEPEP_LL_SEND_CORES=10
export DEEPEP_LL_RECV_CORES=46
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

未指定S/R默认28/28，总数56。SDK符合身份时复用；不删除缓存或自动强制重建。全rank使用同提交/配置wheel。日志、缓存、临时文件由环境脚本定位到工程目录；公共脚本不默认选卡。

## 目标机采集示例

下例为用户目标机单机16rank配置，设备须为实际获分配的卡。CPU `--plan`不启动设备，也不证明设备可运行。其他主机的历史双卡限制不因此改变。

```bash
DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
RUN_NAME=segment-001
LL_ARGS=(
  --devices "$DEVICES"
  --num-nodes "$NNODES" --node-rank "$NODE_RANK"
  --master-addr "$MASTER_ADDR" --master-port 29500 --shmem-port 29501
  --num-tokens 32 --num-max-tokens-per-rank 256
  --hidden 7168 --num-topk 6 --num-experts 256 --dtype bf16 --do-cpu-sync 0
  --warmup 10 --num-iters 100
  --ll-stage-timer --ll-timer-iters 5 --ll-timer-mode observe
  --skip-perf-test --output-dir results/ll-timer --run-name "$RUN_NAME"
)
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}" --plan
python tests/ll_stage_timer/run.py "${LL_ARGS[@]}"
```

多机沿用目标环境的节点/端口/设备配置，所有节点参数和run-name一致，每机16连续rank；跨机MTE需要真实可用peer映射。不要伪造EP或把机内验证当成跨机验收。默认Exp约13.53GiB/卡，诊断输出额外保留；正确性校验仍在计时外执行。

仅测dispatch加`--dispatch-only`。新模式只接受observe，旧drain和逐专家GM打点明确拒绝。发送结束后参与接收的算法说明见[SHARED_RECEIVE.md](SHARED_RECEIVE.md)。

## 解析

```bash
python tests/ll_stage_timer/analyze.py results/ll-timer/segment-001 --trace-iteration 4
```

多机先按原目录合并同一run的各rank原始文件，不能覆盖或混入其他轮次。输出summary.json、stages.csv、metrics.csv、intervals.csv、cores.csv和trace.json；原始JSON旁另有参考格式.bin。缺rank、重复采样或格式损坏会失败。实际区间只覆盖采到的前两次，完整统计看metrics。

正式性能另装OFF wheel，不传`--ll-stage-timer`，仍使用10/100/最后50及Exp+scalar all_reduce协议。ON profiler含诊断同步/写出，不能替代OFF基线。新旧诊断schema不能直接作耗时比；旧数据仍可离线解析。
