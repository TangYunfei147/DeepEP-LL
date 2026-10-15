# run.py 参数手册

本手册对应 `tests/ll_probe/run.py`，逐项列出当前统一入口的参数、默认值及实际作用。
部署步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)，带宽公式见 [BANDWIDTH.md](BANDWIDTH.md)。

## 命令结构与优先级

```bash
python3 tests/ll_probe/run.py [统一入口参数] -- [所选测试的参数]
```

`--`是分隔符：左边管理部署/执行，右边传给所选测试。没有右侧参数时可省略分隔符。
不能把`--target-run`、`--plan`等放到右侧；也不能在右侧覆盖rank、设备、endpoint或输出路径。
不接受长选项缩写。统一入口不是torchrun子进程脚本，每台机器只运行一次它，不要再套一层torchrun。

## 统一入口参数（放在 -- 左侧）

| 参数 | 默认/必填 | 含义与限制 |
|---|---|---|
| `-h` / `--help` | 可选 | 打印统一入口帮助并退出，不运行测试；底层参数详见本手册 |
| `--config PATH` | 必填 | 部署YAML；相对路径按工程根解析，不按终端cwd解析 |
| `--test TYPE` | 必填 | 选择下面测试类型表中的一个值 |
| `--node-index N` / `--node-rank N` | 0 | 本机在YAML ips中的索引，从0开始；两个选项同义 |
| `--devices-per-host N` | YAML值，缺省16 | 覆盖每机rank数，1..16；所有节点必须一致。选择本机设备0..N-1 |
| `--run-id NAME` | YAML run_id；计划时可省略 | 同一次作业各节点必须相同；实际执行必须显式给CLI或YAML值。1..96字符，首字符字母/数字，后续可含字母/数字/下划线/点/连字符 |
| `--plan` | 未指定执行模式时也默认只计划 | 打印JSON计划和命令，不启动设备。即使同时指定target-run/validation也不执行，但仍检查相应约束 |
| `--target-run` | 关闭 | 在用户目标机真正执行，可按配置扩展设备可见性；与validation互斥，禁止在A5使用 |
| `--validation` | 关闭 | A5真实单机双rank验证，四个设备可见性变量固定0,1；已有冲突值会报错。对profiler默认使用小负载，见后文 |
| `--` | 有测试参数时必须 | 后续全部按测试参数解析；统一入口仍做白名单检查 |

不写`--target-run`或`--validation`时，即使有run-id也不会执行。计划中native小域返回
`status: unsupported`、`command: null`；真正执行该组合会拒绝。计划不等于库已构建或设备/HCCL已验收。

## 测试类型

| --test | 实际world | 实际调用与测量范围 |
|---|---|---|
| `empty` | 2/4/8/16/32 | `four_empty`，完全空设备函数；框架仍有初始化、Exp、all_reduce |
| `tiling` | 2/4/8/16/32 | `four_comm_1..4`的layer1，仅读取tiling |
| `comm` | 2/4/8/16/32 | `four_comm_1..4`的layer3，最简payload搬运 |
| `metadata` | 2/4/8/16/32 | 每rank本地元数据提取，逻辑分区与world独立，不包含跨rank元数据交换 |
| `native` | 16/32 | 生产LL dispatch/combine；2/4/8明确unsupported |
| `network` | 2/4/8/16/32 | Session和指定两rank的链路诊断，不是带宽采样 |

world = ips数量 × 每机rank数；32必须多机。旧单进程IPC/双rank SHMEM不是这里的test选项。

## YAML字段（不是run.py命令行选项）

| 字段 | 缺省值 | 含义 |
|---|---|---|
| `ips` | 必填 | 唯一、非组播且非0.0.0.0的IPv4列表；顺序定义node-index，首项用于bootstrap |
| `rank_num_per_server` | 16 | 每机rank数；可被--devices-per-host覆盖 |
| `port` | 29500 | torchrun/HCCL bootstrap端口，1024..65535 |
| `shmem_port` | 29501 | SHMEM bootstrap端口，1024..65535，必须与port不同 |
| `timeout` | 600秒 | 本节点作业监督超时，10..3600秒；超时终止进程组，不能用CLI --timeout替代 |
| `run_id` | 无 | 运行身份；CLI --run-id优先 |
| `output_root` | results | 输出根，必须在工程内；最终节点目录不得通过符号链接逃逸 |
| `hardware_config` | 无 | 硬件模型YAML路径；ep*.yaml模板已设置。仅探针路径传入此参数，native/network不使用该带宽模型 |

兼容旧的纯IPv4列表YAML。未知字段和重复键会拒绝。节点输出目录为
`<output_root>/<run-id>/node-N/`，必须尚不存在；重跑请换run-id。不同节点可使用不同工程绝对路径。
当前入口没有指定任意设备ID列表的选项，只按0..N-1选择。外部CANN位置由已加载环境或CANN_ENV提供。

## empty / tiling / comm / metadata 的转发参数

以下参数放在`--`之后。不要把metadata的`--batch/--hidden`误认为comm的`--bs/--hidden-size`。

### 重复采样参数

| 参数 | 正式默认值 | 含义 |
|---|---|---|
| `--warmup N` | 10 | 预热次数，0..10000 |
| `--samples N` | 100 | 采集次数，1..10000 |
| `--tail N` | 50 | 使用最后N条，0表示全部，必须0≤N≤samples；这里不会因samples缩小而自动缩小tail |
| `--delay-rows N` | 118260 | 每次样本前Exp缓冲的行数，正整数 |
| `--delay-cols N` | 30720 | 每次样本前Exp缓冲的列数，正整数 |

默认Exp为FP32，约13.53GiB/卡，独立于payload；每次采集保留Exp+scalar all_reduce协议。
validation将默认值覆盖为warmup2/samples3/tail2/64×64；允许缩小，不允许warmup>2、samples>3或任一Exp维度>64。
validation下若把samples改为1，也必须同步设置tail≤1。正式目标模式可显式采用更多样本。

### 通信类参数（empty / tiling / comm）

| 参数 | 默认值 | 含义及实际作用 |
|---|---|---|
| `--bytes N` | 无形状时15392 | 每条有向边的payload字节数，正数且32B对齐；不隐式补齐 |
| `--bs N` | 无 | 每条边的行/token数，须与hidden-size一起给出，正整数 |
| `--hidden-size N` | 无 | 每行元素数，正整数；bytes=bs×hidden-size×element-bytes |
| `--element-bytes N` | 2 | 形状模式元素大小，可选1/2/4/8；byte-pattern搬运，不代表浮点运算 |
| `--cores N` | 8 | 每张卡启动的AIV核数，1..32；不是rank数，也不是生产LL的动态核数 |
| `--lanes N` | 4 | 每核流水槽数量，1..4；只有comm搬运路径使用 |
| `--scenario N` | 1 | 1=1→1（默认0→1）；2=多→1（默认所有非0→0）；3=0→接收集合；4=所有rank互发且无自环 |
| `--fanout N` | world-1 | 仅scenario3；接收rank为1..N，与targets互斥，仍启动完整world |
| `--targets R [R ...]` | 场景1为1；场景3为1..fanout | 场景1/3的全局目标rank列表，发送端固定0；场景1必须恰好一个目标 |
| `--sources R [R ...]` | 1..world-1 | 场景2的全局来源rank列表，接收端固定0 |
| `--hardware-config PATH` | YAML指定值，否则无 | 覆盖带宽参考配置；相对工程根解析，必须在工程内。仅comm产生带宽/理论耗时，empty/tiling不计算带宽 |

显式bytes与形状同时给出时必须相等。每边≤64MiB，world×bytes≤256MiB；EP32每边≤8MiB。
empty不读取这些数据，tiling只读取参数；它们仍由同一Host创建缓冲并检查参数，所以bytes/cores等仍须合法。
scenario控制tiling的任务名，但empty始终是four_empty。

### metadata参数

| 参数 | 默认值 | 含义与限制 |
|---|---|---|
| `--batch N` | 32 | 本地逻辑token数，1..256 |
| `--hidden N` | 4096 | 本地逻辑hidden size，1..65536；不是comm的hidden-size |
| `--top-k N` | 6 | 每token专家路由数，1..min(32,experts) |
| `--experts N` | 256 | 逻辑专家总数，1..1024且能被partitions整除 |
| `--partitions N` | 16 | 逻辑专家分区，16..256；不是启动rank数，还须满足编译分组G及Host布局/UB约束 |
| `--hardware-config PATH` | 同上 | 可记录配置快照，但metadata不计算payload带宽或利用率 |

metadata每个实际rank执行相同logical source-rank=0本地提取；实际world用于采样框架通信。
其AIV数由Host查询并使用发送半区，`--cores`不会改变metadata核数。

### 接收但对上述测试不生效的兼容参数

当前共用白名单较宽，“能解析”不等于所选kernel会使用：

- `--chunk`（默认4096）、`--mode`（默认3）：供旧peer/shmem搬运探针使用；统一入口不选择它们。
  comm实际搬运chunk在Host固定为4096；修改这两个选项不会改变本入口的comm。
- metadata之外的`--batch/--hidden/--top-k/--experts/--partitions`不改变comm。
- metadata忽略`--bytes/--element-bytes/--cores/--lanes/--scenario/--chunk/--mode`的算法作用；
  部分通用参数仍会被解析或检查。metadata明确拒绝`--bs/--hidden-size`及显式fanout。

请只设置所选测试实际使用的参数，以免配置看似生效但kernel没有变化。

## native 的转发参数

仅下表参数可通过统一入口转发；不要把原生测试脚本的全部CLI都直接塞到`--`后。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--num-tokens B [B ...]` | 16 | 每rank实际token数；支持矩阵多值，0<B≤M≤256 |
| `--num-max-tokens-per-rank M` | 当前B | 容量上限，最大256 |
| `--hidden H [H ...]` | 7168 | 隐藏维度，H≥4；实际dtype/layout/UB约束由Host检查 |
| `--num-topk K [K ...]` | 8 | top-k，1..16 |
| `--num-experts E [E ...]` | 128 | EP≤E≤1024且E能被EP整除 |
| `--dtype TYPE [TYPE ...]` | bf16 | bf16/fp16/fp8 |
| `--fp8-scale-dtype TYPE [TYPE ...]` | int32 | int32/float32；仅FP8使用 |
| `--do-cpu-sync 0/1 [0/1 ...]` | 1 | 传给生产dispatch的CPU同步选项，会影响元数据处理/校验路径 |
| `--seed N` | 0 | 测试随机种子 |
| `--warmup N` | 10 | 预热次数，≥0 |
| `--num-iters N` | 100 | 正式采集次数，≥1；不是探针的--samples |
| `--perf-tail N` | min(50,num-iters) | 最后N条；0表示全部，显式设置时须≤num-iters |
| `--delay-rows N` | 118260 | Exp行数，正整数 |
| `--delay-cols N` | 30720 | Exp列数，正整数 |
| `--dispatch-only` | 关闭 | 只执行dispatch部分，不执行combine |
| `--repeat-combine N` | 2 | combine重复正确性检查次数，至少2；不是性能采集次数 |
| `--skip-perf-test` | 关闭 | 跳过性能采样，保留正确性检查 |
| `--test-first-only` | 关闭 | 当前为兼容占位：解析器接收，但当前测试主体没有消费此字段；不能用它减少矩阵case |

多值参数形成笛卡尔积。实际启动的world、node数、每机进程数、地址/端口和输出由统一入口传入。
原生LL工作区、动态核数等协议限制仍生效；此入口不允许覆盖--num-sms/--num-bytes等。

## network 的转发参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--peer R` | 1 | 与global rank0做双向小包检查的全局rank，0<R<world；EP32两机各16卡时可选16以测试跨机 |
| `--no-huge-pages` | 关闭 | 显式关闭大页。目标模式默认8GiB工作区与大页；validation固定2MiB且无大页 |

network没有samples/bs/hidden参数，不输出带宽分位数。所有节点须使用相同peer及部署配置。

## 常用命令

以下均在已加载CANN、激活匹配Torch/torch_npu环境且探针已构建后执行。

```bash
# 仅查看EP32的节点1计划：不启动设备
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 1 --test comm --plan -- --scenario 3 --fanout 3 --bs 32 --hidden-size 7168

# 用户目标机单机16卡空核（先改YAML中的IP）
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep16.yaml --test empty --run-id empty16-001 --target-run

# A5仅双卡小负载验证
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/a5_validation.yaml --test empty --run-id empty-a5-001 --validation

# 用户目标机EP32：两台各执行一次，另一台node-index=1，共同run-id
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep32.yaml --node-index 0 --test comm --run-id fanout3-001 --target-run -- --scenario 3 --fanout 3 --bs 32 --hidden-size 7168 --element-bytes 2 --samples 100 --tail 50

# 原生LL只做正确性检查
python3 tests/ll_probe/run.py --config tests/ll_probe/configs/ep16.yaml --test native --run-id native-check-001 --target-run -- --skip-perf-test
```

构建基础wheel不会自动构建独立探针；通常还需执行`bash tests/ll_probe/profiling/build.sh`。
network另用`network_probe/build.sh`。完整路径/环境与结果汇总步骤见[DEPLOYMENT.md](DEPLOYMENT.md)。


显式rank列表使用空格分隔，不能含0、重复项或超出world的值；各机必须设置相同选择。
例如两机各16卡，global rank16/18/31对应第二台的local rank0/2/15：

```bash
# 在前面的统一部署参数后选择一种：
# 1→1：rank0发送到rank16
--scenario 1 --targets 16
# 1→多：rank0发送到rank16、18、31
--scenario 3 --targets 16 18 31
# 多→1：rank16、18、31发送到rank0
--scenario 2 --sources 16 18 31
```

上面列出分隔符`--`之后的测试参数；完整命令见DEPLOYMENT.md。
