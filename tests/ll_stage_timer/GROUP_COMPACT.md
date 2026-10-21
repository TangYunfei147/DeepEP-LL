# G4 紧凑远端任务候选

分支 `kernel/ll-g4-group-compact`，基线 `kernel/ll-g4-s3-baseline@ac876ba`。
当前实现身份 `g4-group-compact`，核心协议 `16000+S`，阶段 schema12、Stage3。
默认 G4/S28/R28、固定56 AIV；R本组初发、接收/relay、combine及七模式编译结构保留。
这是待目标机验证的优化候选，不代表性能已经提升。

## 当前版本：每个token只分类一次

上一版613ae81在EP16/B256 FP8上被用户报告严重劣化。本版在同一候选分支替换重复规划，
不回写S3基线；原始版本仍可通过Git提交取得。

1. 每个S核仅对原token分片执行一次Classify，产生最多ceil(B/S)*K个
   [原始route index, mask]描述符，同时保持sourceMask/权重的唯一写者。
   非去重也按原token分片生成DIRECT描述符。本组初发仍归R。
2. 本核按“FIRST_HOP/DIRECT × 目标组”计数、前缀和、稳定排序一次，
   将连续描述符及bucket偏移发布到本卡GM。空分片也发布空表和ready。
3. 描述符及偏移的MTE3写入完成后，才发布固定ready槽的当前magic。
   所有S核等S个生产者就绪；这是新增的S-only规划依赖，R不参与，不调用SyncAll。
4. 各S核读取小型偏移表，按跨producer/group/phase连续owner确定自己的strided slice。
   用DataCopyPad只批量读取自己负责的8B描述符，每项在UB填充到32B；
   排入全部段后统一MTE2_S，随后发包。不逐packet重新搬描述符、不遍历其他核路由。
5. 保留S3前移前缀、发送末尾延后退休、原接收/relay/combine。
   当前仍先发FIRST_HOP再发DIRECT；不在两个发送阶段之间增加完成屏障。

组序仍为LlDirectGroupOrder(groups, own, rank%G, 0, step)。producer按原token分片顺序排列，
因此实际packet所有权及顺序与613ae81一致；CPU报告与原包数均衡数据仍适用。
每卡S核总远端初发包数最多差1；FIRST_HOP和DIRECT总量分别最多差1；
每个组每种类型最多差1，两类型相加最多差2。不承诺q成本、包尾成员数或真实耗时均衡。

## 共享内存生命周期

- 任务表从DispatchServerDedupCountLayout.stateBytes之后开始，Host检查结束位置不超过
  combine控制区起点1MiB；静态断言核对combine常量。没有占用第二个MiB或输出/return区域。
- ready位固定在每核原512B magic槽的+32位置，原magic word0不变。
  Host初始清零，各dispatch使用1/0交替；ready位置不随B/K/E改变，避免变shape时误认旧描述符。
- descriptor/header/ready GM视图均禁用L2缓存。先完成数据写入，再发布ready。
  同流下一次kernel必须等待本次结束，复用单份任务表；不支持同Session并发launch。
- 无原子、动态抢任务、额外kernel或跨rank计划交换。新增了本卡S-only就绪等待和GM表读写，
  不能宣称完全没有同步成本。

## UB与开销

每个S核不再做2*(EP/G-1)遍全量Compare/GatherMask，也不再全量折叠各目标组的路由。
仍保留完整路由（供原q统计）和完整输入权重（跨原token分片发包所需），本次未改变q算法。

用户配置EP16/B=M256/H3584/K8/E384、预量化FP8、S28/R28的源码规划值：

| 项目 | S3/ac876ba | 重复规划/613ae81 | 当前共享计划 |
|---|---|---|---|
| S核总UB | 106 KiB | 120.4375 KiB | 118.6875 KiB |
| R核总UB | 100.9375 KiB | 100.9375 KiB | 100.9375 KiB |
| 发送/接收lane | 6/6 | 6/6 | 6/6 |
| 回退核数 | 0 | 0 | 0 |

当前S核计划缓冲6400B：本核双描述符缓冲1280B、偏移表1792B、ready快照896B、
排序游标64B、自己负责的描述符缓存2368B。相对旧版8192B索引减少1792B。
每rank共享GM表19712B；由387KiB处开始，结束于406.25KiB，低于combine起点1MiB。
本次目标是减少重复Scalar和筛选，不是依赖UB下降恢复overlap。

规划build/publish/wait/gather统一记录在PREPARE_INPUT中；当前schema12不能从此阶段
单独提取ready等待耗时。最终判断必须看timer OFF完整kernel。新同步和GM读取仍可能抵消收益。
Host最低UB、overlap预算、GM边界检查一起维护；超限拒绝，不覆盖combine或静默截断。

## 本地证据

CPU生产slice/布局helper及producer模型检查覆盖EP16/32/64/128、S=1/10/28/55、B<S、K=32、
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
ST在初始化SHMEM前检查身份、16000+S和Stage3，拒绝旧S3 wheel。

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
