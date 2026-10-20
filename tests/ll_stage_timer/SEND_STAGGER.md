> 本文记录父分支的发送错峰设计。本分支另增加发送后共享接收，当前协议与验收以[SHARED_RECEIVE.md](SHARED_RECEIVE.md)为准。

> 当前编译收发配置以[CORE_SPLIT.md](CORE_SPLIT.md)为准；以下3876de4仅为错峰分支最初创建点，不可用作固定56配置的等条件性能基线。

# 泛化G4基线与NDR式发送顺序轮转

分支 `kernel/ll-g4-stagger` 从 `kernel/ll-configurable-groups@3876de4` 创建，两边默认G4。本候选仅移植NDR的发送顺序轮转，不引入RDMA、L2绕过、NDR分组tiling、冷编译拆分或schema3/4打点。

## 改动边界

- 仅M>32的去重dispatch路径。FIRST_HOP任务列表以 `(源rank组内编号 + 发送核编号 + 全局token编号) % 任务数` 为起点循环发送；首跳仍选原组内最早top-k专家。
- DIRECT按 `(源rank组内编号 + 发送核编号) % 组数` 旋转原目标组排列。组内token/k顺序、q、source mask/weights和包数保持不变。
- FIRST_HOP仍全部先于DIRECT，不实现本组DIRECT穿插；不加跨卡同步或额外lane排空。小容量非去重、接收转发及combine保持基线行为。
- 同NDR公式，globalToken与核分片存在相关性；例如B256/28发送核/两个首跳任务时，部分核会得到相同相位。它不保证物理目标错开，也不承诺timer OFF性能改善；此次按用户要求保留公式以便对照。
- MTE、编译期G配置、固定56总AIV和可配置收发分工、协议标识7000+发送核数、阶段schema5/专家schema1均保持基线。不同G和不同分支的wheel不可混在同一作业内。

## 目标机使用

两个独立checkout分别选择 `kernel/ll-configurable-groups`（无错峰）与 `kernel/ll-g4-stagger`（轮转）。使用各自匹配Python环境和完整wheel。不要覆盖已有本地修改。

```bash
git clone --branch kernel/ll-g4-stagger --single-branch \
  https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-g4-stagger
cd DeepEP-LL-g4-stagger
# 加载当前机器的CANN环境及Python环境，再执行工程环境脚本。
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
export DEEPEP_LL_STAGE_TIMER=OFF
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

已有SDK身份匹配时可按原流程复用，不因G或顺序变化自动重建SDK。修改源码后两边都需要重装wheel，清除旧的显式G16编译覆盖或核对记录中的logical_group确实为4。

正式对比保持相同B/M/H/K/E/dtype/seed/设备/Exp及10预热、100采样、最后50条协议，使用新run-name分别保留结果。OFF构建不传--ll-stage-timer，比较完整dispatch及combine task；不得以ON收益代替OFF收益。另行重装ON wheel可做schema5诊断，但不是逐次事件时间线，也没有NDR的schema3字节带宽报告。

本次本地CPU：基线83 passed/2缺编译器skip，候选96 passed/3缺编译器skip；各1项既有Bash检查未选。模型覆盖G4/G16、不同分片/路由下包集合与q/权重恒等、DIRECT稳定顺序和起点分布；未进行CANN编译、设备正确性或性能测试。实际验收由目标机完成。

当前固定56核的A/B基线为 kernel/ll-configurable-groups@02b1ef6e11d238f4b3af997057acac5e64b338e2。对比时两边使用相同S/R、G、shape和timer开关。
