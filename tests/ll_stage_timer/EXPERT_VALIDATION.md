# 接收侧专家诊断：本地验证与目标机验收

基线为3262d7e；新增可选专家缓冲后核心ABI为5。阶段schema保持2，专家schema为1。
远端验证环境已回收；本次没有SSH操作，没有CANN编译、SDK构建或设备运行。
历史ABI4的A5验证记录不覆盖此变更。

本地Windows CPU检查覆盖：专家归属/转发目标、损坏与未完成记录拒绝、跨物理机转发拒绝、
跨核累计与跨度区分、缺专家数据拒绝、对齐缓冲分配/attach/延迟导出替身、
含空格目录迁移后的离线统计、旧阶段解析及隐藏重复包络时不丢累计统计。
CANN C++编译、完整LL精度和目标硬件时延均待用户验收。

## 目标机更新

所有节点更新同一提交并重建完整wheel，保持工程内原HNS_1825 SDK：

```bash
git pull --ff-only origin kernel/ll-ndr-hns1825
source tests/ll_rdma/env.sh
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_RDMA=ON
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

原完整LL启动命令增加`--ll-stage-timer --ll-expert-timer`；先用新run名的小case完成正确性检查，
例如保持原拓扑/网络配置，将B设为1、M设为256、ll-timer-iters设为1，再扩展到正常case和5轮。
observe优先用于定位下发与等待，drain用于对照排空后的成本；不能混算为未插桩基线。
不支持用EP2/4/8冒充原生LL验证，也不能用本地CPU测试声称设备正确性。

一个analyze.py命令输出原统计、trace以及experts.csv/expert_cores.csv。
专家汇总保留executing_rank与expert_rank，转发执行者与最终专家所属rank可能不同。
均值按实际执行过阶段的轮次计算；跨核累计是工作量，不是端到端延迟。
共享首跳准备、relayReadIndex写回和批量等待不分摊给单专家。

默认隐藏六类重复发送包络仅影响trace展示，stages.csv仍保留全部计数和累计值；
`--show-repeated`可恢复。旧数据可以重新导出简化trace，但无法补出新的专家测量。
