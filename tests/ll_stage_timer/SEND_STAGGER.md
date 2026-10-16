# NDR 去重 dispatch 发送错峰

本次只调整 `SendWithDedup` 的发送顺序。容量 `num_max_tokens_per_rank > 32` 时启用该路径，因此 `B=256、M=256、EP16、G4` 使用新顺序。小容量非去重路径、接收侧转发和 combine 顺序保持原样。

## 顺序

- FIRST_HOP：每个 token 的原首跳任务列表，以 `(源rank在组内的编号 + 发送核编号 + 全局token编号) % 首跳任务数` 为起点循环发送。零任务不发送，一个任务顺序不变。仍由该组 top-k 最早的专家决定首跳目标，不改成同号卡。
- DIRECT：以 `(源rank在组内的编号 + 发送核编号) % 目标组数` 旋转原来的目标组遍历。组内 token/k 稳定顺序不变；大域原有半区交替序列只做循环移位，不改成员集合。
- 当前核的 FIRST_HOP 阶段仍在 DIRECT 前；共享原 packet/lane 环，无新增 barrier、批量排空、等待或跨rank调度同步。相位是遍历起点，不是人为延迟发送。

EP16/G4、同一发送核下，组内源卡0..3从四个不同目标组开始。每卡28个发送核时，DIRECT首个遍历组各分到7核。但空组会被跳过，负载和进度也不一致，因此这不是时隙预约或“无拥塞”保证。FIRST_HOP列表由实际路由决定，其轮转也不保证不同源的首个物理目标不同。

只改变发送顺序：每核任务归属、首跳目标、q槽位、包尾成员、source mask/weight、包数、字节数和协议均不变。无需修改SDK、ABI或timer schema，也没有新增启动选项。不假设卡间直连关系；该方案针对发送相位集中，不能修复专家热点或物理链路容量不足。

## 目标机 A/B

未错峰基线提交：`bf1afabe24e5b14de4b4a1df972529bccea64f62`。候选为包含本指南的新提交。建议在独立checkout/环境分别保留基线与候选，不覆盖未提交修改。全rank使用同一版本，切换源码后重建wheel。

```bash
# 当前NDR分支、目标机已配置环境；先检查git status再快进更新。
git pull --ff-only
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

用同一输入、seed、B/M/H/K/E、实际AIV及Exp配置运行原性能命令；不要添加 `--ll-stage-timer`，按既有10预热/100采样/最后50条协议比较dispatch task时间，同时检查combine未退化。使用新的run-name，例如baseline-001与stagger-001，避免输出目录冲突。设备选择沿用已明确分配的目标机环境，不根据此文自动扩展。

诊断时，两版本都重装timer ON wheel并启用 `--ll-stage-timer`，再用analyze.py比较同一rank/轮次的三类计数字节。相同路由下计数应保持一致；计数变化不能解释为本次错峰的收益，应先排查输入/构建不一致或错误。再看最长kernel、接收等待和转发耗时的分布。不要把observe阶段下发时间解释为纯链路传输时间。

```bash
python tests/ll_stage_timer/analyze.py results/<case>/stagger-001
```

该变化未利用物理拓扑信息，也未测得目标性能收益；保留是否回退的判断，避免将模型中较均匀的起点直接当作时延改善。

## 验证边界

`test_send_stagger.py` 用均匀、热点组和重复路由覆盖G4/G16、EP16..256及空核/不均匀token分片，核对原/新调度包集合、q、首跳mask、source mask/weights、组内DIRECT稳定顺序及EP16起点分布。另提供直接编译生产order helper的C++测试。

当前本地调度专项9项通过，C++执行测试因没有本地编译器跳过。阶段/调度/dispatch选择回归合计124 passed、7 skipped（缺C++编译器）、1 deselected（既有Bash环境迁移项）；Python模型不替代生产C++或设备验证。A3 SSH编译检查被自动审批拒绝，未执行远端命令；尚无本次CANN编译、设备正确性或性能验证。其他任务的三份本地kernel修改保持原样，不纳入本任务提交。
