# G4 紧凑远端任务候选

分支 `kernel/ll-g4-group-compact`，基线 `kernel/ll-g4-s3-baseline@ac876ba`。
当前实现身份 `g4-group-compact`，核心协议 `15000+S`，阶段 schema12、Stage3。
默认 G4/S28/R28、固定56 AIV；R本组初发、接收/relay、combine及七模式编译结构保留。
这是待目标机验证的优化候选，不代表性能已经提升。

## 调度方式

1. 每个S核加载完整原始路由与输入权重。原token分片仍唯一负责sourceMask和输出权重。
2. 对当前目标组用向量比较和GatherMask生成稳定、升序的uint32原始route index。
   索引使用独立UB，不能与q/量化scratch混用。
3. M>32时按token合并当前组的相邻索引：多命中生成一个FIRST_HOP，
   最小top-k位置仍是首跳，mask/目标专家/原始q/尾部权重不变；单命中生成DIRECT。
   M<=32时每条远端route都是DIRECT。M指num_max_tokens_per_rank，不是实际B。
4. 所有S核使用相同的组遍历次序，跨组和跨FIRST_HOP/DIRECT阶段持续推进owner游标。
   核只发送owner等于自身的任务。无GM任务队列、原子认领、新跨核屏障或额外kernel。
   每核先提交FIRST_HOP再提交DIRECT，阶段间仍无全核完成屏障。
5. 本组不进入该任务表，仍由R核初发。S3前移前缀、延后退休及UB不足时串行回退保留。

组遍历沿用原half-interleave排列及rank组内相位，去掉core相位：
`LlDirectGroupOrder(groups, own, rank%G, 0, step)`。这是为所有核获得相同owner流；
并不保留S3原先逐token FIRST_HOP旋转和逐core DIRECT错峰，也不加入dc_opt的组间配对。

总远端初发packet数在S核之间最多差1。FIRST_HOP总量及DIRECT总量分别最多差1；
每个目标组的每一类型也最多差1，但该组两类型相加最多差2。
FIRST_HOP的包尾成员数和q计算量不等，因此不承诺CPU工作、Scalar指令或真实时长均衡。

## 成本与边界

- 索引压紧/折叠由所有S核重复执行。去重为2*(EP/G-1)轮筛选，非去重为EP/G-1轮。
  不发包的S核也要参与本地计划计算，不能沿用“没有原token分片就返回”。
- 权重从每核原token片段扩大为整个B*K；sourceMask/输出权重仍仅保存原核片段。
  输出权重用独立、从UB零偏移开始的搬入，避免任意K时使用不对齐的UB切片。
- 远端任务缓存从最多ceil(B/S)*K个四word描述符改为B*K个一word索引。
  两者都需要按shape核算，不能只宣称“索引更紧凑所以UB更少”。
- Host的串行最低lane预算和Device overlap预算已一起修改；可能减少lane数、触发
  SERIAL_FALLBACK，或拒绝比基线更大的shape。无静默越界、无自动减核。
- q仍按原始路由计算，FIRST_HOP仍有组内relay依赖；没有降低通信包数或payload字节数。
- 增加的S_V/V_S依赖属于新索引生成的必要本核同步，不是新增跨核barrier。

## 本地证据

CPU生产调度helper检查覆盖EP16/32/64/128、S=1/10/28/55、B<S、K=32、
全本组/热点组、去重开关、原始路由覆盖和包尾成员。CPU预算/Host选择检查不能替代
AscendC编译或NPU DMA正确性。最终验证数量见docs/BUILD_DISPATCH_SPLIT.md。

固定E384/K8/M256、G4/S28/R28、normal、实际seed2的纯CPU统计如下。
“最大值均值”为每rank先取最忙S核初发包数，再对rank求均值；不是微秒或加速比。

| EP | B | S3最大值均值 | 候选最大值均值 | S3最坏rank核间差 | 候选核间差 |
|---|---|---|---|---|---|
| 16 | 8 | 3.000 | 1.000 | 3 | 1 |
| 16 | 32 | 5.875 | 3.875 | 5 | 1 |
| 16 | 48 | 6.000 | 5.000 | 5 | 1 |
| 64 | 8 | 7.328125 | 2.000 | 8 | 1 |
| 64 | 32 | 13.546875 | 7.265625 | 12 | 1 |
| 64 | 48 | 14.515625 | 10.953125 | 12 | 1 |

同批hotspot案例的最忙核包数没有改善，不能据此预计普遍收益。
全部路由、q、FIRST_HOP成员、接收行及接收核保持相同，初发包总量不变。

## 获取与构建

首次在用户选定父目录clone；已有checkout先查Git状态，避免覆盖本地修改。

```bash
git clone --single-branch --branch kernel/ll-g4-group-compact \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-g4-group-compact
cd DeepEP-LL-g4-group-compact
```

加载目标机实际CANN环境后，在工程根执行：

```bash
source scripts/ll_timer_env.sh
python3 -m venv --system-site-packages .venv-ll-timer
source .venv-ll-timer/bin/activate
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28
export DEEPEP_LL_STAGE_TIMER=OFF
python scripts/build_shmem.py
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

已校验SDK可复用，不因本候选强制重建。全rank需要同提交、同S/R的完整wheel；
ST在初始化SHMEM前检查身份、15000+S和Stage3，拒绝旧S3 wheel。

## 目标验收与A/B

沿用S3原命令、选卡、拓扑和shape，使用全新的run-name。先做完整dispatch+combine
正确性（多seed），再做timer OFF正式profiler；10预热/100采集/尾50协议不改。
优先B8/32/48、BF16/FP8，并同时记录M；测试M32/M256区分非去重/去重。
不要因理论包数均衡就跳过combine验收或复用旧输出。

诊断时改ON并全rank重装，在原命令加`--ll-stage-timer --ll-timer-mode observe`。
重点看KERNEL、SEND_TOKENS、PREPARE_INPUT、CAL_TOKEN、SEND_LANES/RECV_LANES、
SERIAL_FALLBACK及接收等待；父子阶段不可相加。schema12语义保留，但所属核变化，
不能将同编号核的局部耗时比直接解释为算法加速比。最终性能依据完整OFF kernel。

CPU核间计数可在同一checkout分别运行（不需要wheel）：

```bash
python tests/elastic/inspect_dispatch_routes.py \
  --num-processes 16 --num-nodes 1 --num-tokens 8 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 \
  --send-policy s3 --run-name s3-cpu-001 --output-dir results/group-compact
python tests/elastic/inspect_dispatch_routes.py \
  --num-processes 16 --num-nodes 1 --num-tokens 8 --num-max-tokens-per-rank 256 \
  --hidden 3584 --num-topk 8 --num-experts 384 --dtype fp8 \
  --send-policy group-compact --run-name compact-cpu-001 --output-dir results/group-compact
```

CLI默认候选；Python build_plan默认s3保留历史模型调用，比较时显式传send_policy。
manifest记录策略。events仍只是一条合法因果回放，不模拟S3异步DMA、真实延迟或lane预算。
本候选未在CANN/设备上编译运行；冷编译ON/OFF和完整ON wheel验收仍待目标机执行，
方法见docs/BUILD_DISPATCH_SPLIT.md。
