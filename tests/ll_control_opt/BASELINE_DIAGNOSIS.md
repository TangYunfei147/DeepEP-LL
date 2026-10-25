# Baseline主瓶颈：低扰动稀疏打点

参考 `kernel/ll-g4-small-b-prefix@39eadb0` 的schema15 sparse实现，只移植打点与
实际capture profiler；不带入小B并行前缀、P1 VF、SDK或其他算法改动。
业务参考仍为 `ed1c553`，新诊断源码固定为 `9c2c333`，OFF/SPARSE均从该提交构建。
53份业务源码去探针后的token与ed1一致；这不是CANN代码生成或性能等价证明。

## 在目标机执行

从当前工程根目录执行，沿用此前成功跑性能套件的Python/CANN环境及部署变量。
固定单机EP16、S30/R26、G4、B=M256/H3584/K8/E384、FP8 dispatch/BF16 combine。
`ASCEND_RT_VISIBLE_DEVICES`为已分配的16张卡；MASTER_ADDR、MASTER_PORT、SHMEM_PORT
沿用原配置。每次设备调用前检查空闲，忙时停止，不终止其他任务。

```bash
git pull --ff-only
python tests/ll_control_opt/profile_baseline.py \
  --devices "$ASCEND_RT_VISIBLE_DEVICES" \
  --run-prefix baseline-bottleneck-001
```

加`--plan`仅查看计划。`--build-only`先完成两份完整wheel及Host导入验证；之后可以
用`--run-only`执行。通常完整命令即可，它会复用已通过身份和哈希校验的构建缓存。

入口先在独立诊断工作区构建同一提交的 **OFF与SPARSE** 两份完整wheel，均为
HT OFF/Release/Ascend950，再分别安装到独立venv。两份Host/native/kernel/SDK/source
身份检查全部成功后，才启动任何采样。既有OFF套件环境、wheel和原始结果不替换。
有效SDK沿标准流程校验复用，不强制重建，不自动联网下载依赖或安装Torch。

每个请求seed `0/17/101`（实际稳定路由seed `2/19/103`）依次运行：

| 版本 | 采样 |
|---|---|
| OFF | 10次预热、100次profiler采样、各rank尾50 |
| SPARSE | 10次预热、100次稳定调用、5轮实际挂载timer的observe capture |

两者均保持Exp+scalar all_reduce。SPARSE的5次实际调用另用profiler包住，包含其
初始化、业务及本核导出。CPU复制、正确性检查和JSON导出延后到采集结束。
dispatch和combine都采；总计6次ST、480份原始timer样本。
可显式`--timer-iters 3`缩为3轮；最多3个seed。

OFF与active SPARSE完整kernel task的比较用于观察打点扰动，不是算法加速比。
不能用未挂载timer的ON调用、内部KERNEL跨度、Event或API时间代替active任务耗时。
profiler任务缺失、顺序不明或含混时标为unavailable，不补造数值。

## 打点留下什么

每核仅14个首次时间戳，512B独占槽，退出时只发布前192B；56核分配28KiB，
实际发布10.5KiB。没有逐包累计数组、调用计数、profile扫描或额外诊断SyncAll。
仍会有首次标记检查和导出开销，不能声称零扰动。

| 点 | 含义 |
|---|---|
| ENTRY / INIT_DONE / WORK_DONE | 入口（计时器初始化前）/初始化完成/业务完成、导出前 |
| COUNTS_SENT / COUNTS_RECEIVED | 原计数发送边界/本核计数接收阶段结束 |
| PREFIX_LOCAL_DONE / PREFIX_READY | 原前缀阶段提交边界/本核前缀准备完成 |
| SEND_PREPARED / FIRST_SEND / SEND_DONE | 发送准备/首次下发前/已有发送退休完成 |
| RECV_PREPARED / FIRST_RECV_PROBE / FIRST_RECV_READY / RECV_DONE | 接收准备/首次探测/首次观察就绪/有任务接收流程收尾 |

`PREFIX_LOCAL_DONE`紧跟原V_MTE3依赖，是提交侧边界，不代表Scalar确认VF已完成；
不为统一解释而新增屏障。`SEND_DONE`可能晚于RECV_DONE，因此首发至SEND_DONE不是纯传输时间。
本分支的成对预取和串行fallback都记录首次Probe/ready，按首次真实观察记录，不从累计值推算。
combine只记录ENTRY/INIT_DONE/WORK_DONE，不声称采集了其收发细分。

第一轮可以定位初始化、计数/前缀门槛、首次发送/接收及后段收尾。
**包数、空轮询率、逐包负载均衡及各小段Scalar成本本轮不采集**；必要时在确认的疑点上
另做定点诊断。缺失不等于0，首次事件之间的跨度不是反复执行的累计时间。
不相加嵌套/重叠区间，不跨rank相减时钟，不据此宣称端口饱和或网络拥塞。

## 只回传一个文件

成功后仅发送：

```text
results/ll-baseline-profile/baseline-bottleneck-001/baseline-report.txt
```

报告先显示同seed的OFF/active SPARSE完整任务耗时，再列16rank时长代理、S/R0/其余R
的同核跨度、首次事件覆盖率，以及较慢rank的真实原始时间点。主指标保持rank-tail池化均值。
每文件按UTF-8实际字节校验不超过65535，默认输出1个，低于用户总数10个限制。
完整日志、原始样本和身份记录保留目标机，不需要上传。

已经采完可单独重新导出，使用新的输出名以保留旧报告：

```bash
python tests/ll_control_opt/report_baseline.py \
  results/ll-baseline-profile/baseline-bottleneck-001 \
  --output results/ll-baseline-profile/baseline-bottleneck-001/baseline-report-v2.txt
```

采样中断后原目录保留，换新的run prefix重跑；独立OFF/SPARSE构建缓存可以复用。
`--system-site-packages`不继承另一个外层venv内的Torch，必须使用已验证的基础Python环境。

## 验证范围

本地CPU检查覆盖实际稀疏recorder的OFF/ON模拟、7模式Host选择、旧对象拒绝、
53文件业务token、schema7～13离线解析、schema15、真实capture profiler逐样本配对、
完整磁盘样本、报告上限和缺失/混装拒绝。
定向及完整timer回归443项通过、2项跳过；完整磁盘报告另37项通过。
跳过项为Windows不支持的POSIX路径测试和历史schema13业务快照检查；当前算法使用ed1独立token证据。
按用户最新规则，本次不连接27、不编译CANN、不执行NPU；目标编译和实测由用户运行上述命令。
