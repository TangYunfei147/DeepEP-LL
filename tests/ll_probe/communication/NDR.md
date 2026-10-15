# 通信探针：UB Memory / NDR

正式入口 `tests/ll_probe/run.py --test comm` 转发 `--transport ubmem|ndr`，默认 `ubmem`。
两种模式使用相同的域2/4/8/16/32、四种通信图、targets/sources、bytes和按字节的核间分片。
32域必须多机。这里只切换最简通信探针，不改变生产LL dispatch/combine。

## 数据路径与计时边界

| 模式 | 数据路径 | 设备task | 完成条件 |
|---|---|---|---|
| ubmem | GM→UB→SHMEM peer GM，原MTE循环 | four_comm_1..4 | 本核MTE发送流水收尾 |
| ndr | SHMEM ROCE QP put，NPU提交网卡传输 | four_ndr_1..4 | 每核、每peer的QP quiet |

模式在Host选择独立编译的库，payload循环不增加模式判断。UB原Run/启动包装保持不变。
NDR每核使用QP索引=core，对每条选中的边一次提交该核完整连续字节段，再quiet。
源和目标都在SHMEM对称内存；传入本地对称目标地址，由RDMA接口翻译到peer。
每核仅分配256B的UB描述符工作区，不把payload经UB分块搬运。`lanes/chunk`不调节NDR数据面，
仅保留公共tiling字段；`cores`同时决定分片数和每peer QP数量（当前探针1..32）。
最小32B分片可能使部分核无数据。不同peer依次提交/完成，当前不做跨peer并发队列调度。
quiet是必要的完成成本；该测量包含提交与完成，不声称是零开销网卡线速时间。
Host校验、跨rank barrier、分配和建域在目标task之外；Event/API/job_join分别保留。
empty仍是four_empty，tiling使用four_ndr_N但提前返回；二者不测NDR通信延迟。

## 每台目标机准备

先加载本机CANN 9.3.0，激活匹配的torch/torch_npu环境。以下命令在工程根执行。
本机CANN位置通过现有环境或显式CANN_ENV配置，不能复制开发机路径。
先停止使用本工程SDK的进程，再重建；所有节点使用相同Git提交与后端。

```bash
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
source scripts/project_env.sh
# 根据真实网卡/驱动选 XSCALE 或 HNS_1825；不要根据带宽名称猜测。
python3 scripts/build_shmem.py --rebuild --rdma-backend XSCALE
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python3 -m pip install -r requirements-build.txt
python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
bash tests/ll_probe/profiling/build.sh
```

首次没有SDK时可省略`--rebuild`。后续`--if-needed`会保留已记录的后端，不隐式换回RDMA关闭配置。
构建使用锁定SHMEM子仓的官方`-enable_rdma -rdma_backend`选项，不另建SDK依赖树。
当前支持的实现为XSCALE/HNS_1825；这里的NDR指NPU RDMA直通，不代表任意InfiniBand网卡兼容。
每台节点启动前，显式将`IBV_EXTEND_DRIVERS`设为实际已安装的provider库路径：
XSCALE通常为libxscale_nda.so，HNS_1825通常为libhrn5-rdmav34.so；以本机驱动为准。
本流程不安装驱动。网卡、端口、路由及SHMEM后端初始化必须在目标环境验收。

RDMA关闭时普通构建仍产出UB库并显示NDR skipped；实际选择ndr会在导入Torch前失败，
绝不回退UB。运行还会核对源码、SDK后端/版本/库hash及探针库hash，变更后必须重建。
`ndr_build.py --compile-check XSCALE|HNS_1825`仅编译链接检查，不产生可运行身份清单。

## 同机与跨机实验

先修改仓内`configs/ep2.yaml`或`ep32.yaml`的ips等部署参数。相对路径从工程根解析。
所有rank使用同一run-id、配置和参数，每次实验换新run-id。run.py根据每机卡数显式选卡。

```bash
# 同机2卡：先CPU计划，再由用户运行；将 --plan 换成 --target-run。
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep2.yaml --node-index 0 --test comm --run-id ndr-local-001 --plan -- --transport ndr --scenario 1 --bytes 15392 --cores 8

# 两机各16卡，0→16/18/31跨机；节点1运行相同命令，node-index改为1。
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0 --test comm --run-id ndr-cross-001 --target-run -- --transport ndr --scenario 3 --targets 16 18 31 --bytes 15392 --cores 8

# 汇集各节点结果后，全rank汇总
python3 tests/ll_probe/profiling/analyze_profile.py results/ndr-cross-001/node-*/data/rank_*/report.json --output results/ndr-cross-001/summary.json
```

对比UB时改为`--transport ubmem`并换run-id；按相同bytes/通信图/cores分别采样。
默认仍10预热、100采集、每次Exp+scalar all_reduce、按task取最后50条，P50/P90/P99分开报告。
配置中的UB硬件参考在ndr模式自动跳过，默认只报有效带宽；显式硬件模型必须带
`transport: ndr`，model中的容量必须来自实际NIC/拓扑，不能沿用UB芯片2016GB/s规格。
两种transport或不同SDK后端的报告不能混合汇总。

## 已验证范围

A5仅物理0/1、缩小Exp验证。XSCALE与HNS_1825设备源码及Host编译链接通过；
CPU用实际NDR函数检查分片、地址、独占QP和quiet，并检查无静默回退及报告模式隔离。
A5当前SDK的RDMA关闭，未重建SDK、未执行NDR网卡收发。编译通过不等于目标网卡可运行，
真实同机/跨机正确性与性能由用户按上述步骤执行，保存所有rank校验和原始报告。

本轮UB重建通过，双rank profiler smoke在探针调用前的HCCL初始化失败：
`RunAicpuCommInit / get kernel failed`。未获得本轮设备正确性通过结论，未重试采样；
日志保留在results/ndr-ub-validation.log及其引用的comm.log（生成日志不提交）。
