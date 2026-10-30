# LL dispatch reference 打点

本分支完整替换旧 stage/expert/sparse 插桩和解析代码。参考来源固定为
`kernel/ll-ndr-metadata-cores@c3d86406733fd28fe4e081e563eaf7e51321c18f`。
当前实现覆盖 `current`、`cb59`、`cb59-stream`、`cb59-pair`、`cb59-small`；
只给 dispatch 打点，combine 保持原业务并移除旧插桩。

记录、采样和表格方法沿用参考分支：

- 主阶段每点一次 `GetSystemCycle` 和一次本核 GM Scalar 写；valid mask 区分未执行与真实零值。
- 接收每个成功 ready 批次三次 clock，Scalar 累计等待和消费；空扫描不重复 clock，首 ready 复用末点。
- S 记录第二个跨卡 MTE 包的准备、成员 q 与提交；R 在 UB 记录稳定任务的发现/消费顺序，不逐包打 clock。
- 每核 80×8 个 uint64、5120B、512B 对齐。R 日志最多128行，放在原 lane 缓冲后的剩余空间，
  不减少 lane，不新增活跃 TBufPool 项；不足则明确截断。原 Counts.Finish 完成后才做诊断 DMA 和缓存发布。
- 沿用参考的29项阶段、逐核所选轮表、单独多轮均值、原始Cycle、S包时序、R批次顺序、覆盖率和原始包记录。
  Python 先以整数作差；超15位的原始整数在 Excel 中存文本。
- 批次初始一次 scalar all-reduce，预热同步后排入100次 Exp，再连续稳定/采样；
  Exp 后不额外同步，采样循环不插 Exp/all-reduce。若 profiler 后追加 capture，另排100次 Exp且不重复初始all-reduce。

## 当前算法的必要适配

使用 **schema7**，拒绝旧 schema5/16 等结果。参考分支有独立 M 核和跨机 RDMA，当前算法没有；
当前核还可能先发送再接收，S/R 可不等分。w44..49记录实际配置，w62=0，w63为实际执行角色掩码，
不伪造 metadata 核。MTE 包明细区分 peer/self，不填写 RDMA staging、CQ 或 quiet 字段。

主 word 槽位与29阶段框架保持参考方式，边界绑定当前真实代码。`recv_prefix_load` 是 w30−w18，
`recv_buffers` 是 w32−w31，未执行的阶段留空。stream 的计数/前缀按其实际片段协议记录，pair 的
count_complete 延伸至配对交接完成。Async sender 返回只表示原有提交/依赖已执行，最终完成由原 Counts.Finish 保证。
详细 word 和公式见 [STAGE_TIMELINE.md](STAGE_TIMELINE.md)。

业务 Scalar、q、包格式、所有权、归约顺序、同步和七个 TU/O2 保持。SDK gitlink/锁文件保持本分支原值，
不复制参考分支的 SHMEM 升级、独立 M 核、1.25 接管或 RDMA 实现。
Host dispatch/combine 参数为144/88B，使用 reference7 链接符号及 Python/Host/device 身份检查；必须完整重装匹配 wheel。

## 目标机一键构建与采样

在当前 `kernel/ll-g4-small-b-prefix` 工程根目录，加载现有 CANN/Python 环境。
原环境需已装项目构建依赖及 Torch/torch_npu；脚本不安装系统依赖。
默认 EP16、B1/2/4/8/16/24/32、M32/H3584/K8/E384、FP8 INT32 scales、CPU sync0、normal、seed0，
10次预热、100次稳定/正式采样、每rank尾50，另1次 reference capture；只测 dispatch。

脚本创建独立源码clone、venv、wheel和日志。先限定拉取SHMEM，清理重编一次本组SDK，清理并构建完整reference ON wheel，
安装及Host验证成功后才进入NPU采样。**禁止拉取CATLASS**。旧组、原工程SDK及small_bs快部署不动。

```bash
run_reference() {
  git pull --ff-only || return $?
  bash tests/ll_reference_profile/one_click.sh \
    ll_reference_001 90.90.93.79 19090 29501
}
if run_reference; then
  echo '完成；仅回传输出提示中的 return_report.md'
else
  echo '已停止；保留 diagnostic-runs/ll_reference_001/logs，终端不会退出。'
fi
```

上面的地址与端口为本聊天原作业示例，使用当前分配给该作业的值。
可以在末尾加 `--build-only`，完成构建和Host检查后停止，不运行设备；随后用同一组 `--skip-build` 采样。
`--skip-build` 只复用已完成且源码/项目wheel身份匹配的构建，不清理SDK、不重编。
每次显式采样使用新 `capture-*` 子目录；失败不自动重试，旧结果保留。
附加 ST 参数放在该选项之后，例如 `--route hotspot --reference-iters 3`。

编译选择沿用原环境变量；例如选择 cb59-stream 时先设置：

```bash
export DEEPEP_LL_DISPATCH_IMPL=cb59-stream
export DEEPEP_LL_SEND_CORES=28 DEEPEP_LL_RECV_CORES=28 DEEPEP_LL_SMALL_B_PREFIX_CORES=0
```

新选择使用新组名重建；不能拿不同选择的旧wheel复用。脚本默认当前分支原选择，不自动移入参考分支的分核。
普通无打点构建用 `DEEPEP_LL_REFERENCE_PROFILE=OFF`；旧 `DEEPEP_LL_STAGE_TIMER=ON` 与 profile 参数直接拒绝。
ON 的内部时间和完整task都不是 OFF 性能基线。

## 解析与回传

一键脚本自动生成 `diagnostic-runs/<group>/analysis/<capture>/stage-times.xlsx` 和 `return_report.md`。
完整 Excel、CSV、样本和日志留目标机，只回传根目录的 **一个≤60000 UTF-8字节报告**，仍遵守累计最多10个文件。
不同case独立sheet，默认逐核末轮；多轮均值另页，不把不同shape/route混在一起。

也可独立重导出，无需重编或重采；使用新的输出目录：

```bash
python tests/ll_reference_profile/analyze.py \
  --input diagnostic-runs/ll_reference_001/runs/<capture> \
  --output-dir diagnostic-runs/ll_reference_001/export-002 --iteration last
```

只做同核差分；嵌套/重叠阶段不能求和，不跨rank对齐时钟。w21−w0不含末尾诊断导出与发布，
w57−w0含此前日志导出但不含随后header/cache写回，两者都不能替代profiler完整task。
R发现顺序不是网络到达顺序；截断记录不能当全量包统计。

历史 `no_timer_ab.py`、`cb59_original.py` 仍加载各自固定Git源码，原SDK/wheel和采样口径不变，
可以完成旧实验；当前源码没有旧设备插桩或旧原始timer解析器。旧原始timer文件需使用其原Git提交解析。
新采样协议 `initial-reduce-exp100-batch-v3` 与旧逐轮Exp/all-reduce结果不可直接当同条件A/B。

## 验证边界

本地检查包括全部22份算法文件的完整业务token和有序Scalar/DMA/同步对照、五选择×七模式Host ABI、
ON/OFF记录器执行、空轮询clock次数、包日志截断/导出、采样调用顺序、精度/损坏拒绝、Excel结构和历史入口。
这些是CPU和编译器桩检查，尚未进行当前实现的CANN编译、NPU正确性或性能验证；由用户在目标机执行。
