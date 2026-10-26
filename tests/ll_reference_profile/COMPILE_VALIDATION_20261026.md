# 参考式 LL dispatch 打点验证记录

验证代码提交：`31d437ed780e7cb73b7fe67e5b6677e12c14adff`。
后续指南/验证记录提交不改变这次验证的C++或Python实现。

## 本地 CPU 和表格

整合回归为365项通过、2项因本机无Torch跳过，另13个子测试通过。
覆盖新reference parser/导出/capture/构建/CLI，以及NDR协议、计数、发送、接收、combine、
Host协调、旧timer采样和专家/布局合同。CPU C++模拟覆盖16/56/72核、2^60大周期、
空指针关闭、有效零、完成标记、核间记录隔离与OFF无clock/store/cache发布。

检查中更新了旧测试对诊断参数/语句的文本匹配，并修正Git Bash测试在Windows上的
路径表示；没有为通过测试改变业务传输依赖。子测试和整合测试集合重叠，不相加为独立用例。

29个阶段名称、顺序、word换算公式与用户提供的参考Excel逐项一致。
Excel的周期值采用Python整数差；超15位原始整数以文本保存，公式可引用精确差值列。
CPU模拟文件的四张表经过Artifact Tool重新计算、公式错误扫描和视觉检查。
这些模拟文件未作为设备数据交付，没有用参考表的历史测量值填充新分支结果。

本次新增单run多case回传报告：正常及大量case测试均检查实际UTF-8字节数≤60000；
过长时按完整case省略并明确省略数量，不截断字符，不删除目标机完整结果。

## 27 纯编译/Host 检查

依据本会话后续明确授权，仅在27既有隔离工程`ndr-compile-20261026`内验证。
没有初始化Runtime/Session、创建NPU张量、导入torch_npu、查询NPU或运行设备kernel。
四个设备可见性变量均固定0,1。SDK复用校验通过，源码与安装内容未重建/修改；
原`ndr-compile`及其历史wheel、日志保持。

工具链为此前同一CANN环境/Bisheng、Ascend950、并行4；完整构建14个dispatch变体、
combine及Host模块并打包，分别安装至隔离venv进行Host导入检查。

| 构建 | Reference宏 | 旧Stage宏 | 完整wheel | Host导入 |
|---|---|---|---|---|
| REF | ON | OFF | 通过 | 通过 |
| OFF | OFF | OFF | 通过 | 通过 |
| ON（兼容回归） | OFF | ON | 通过 | 通过 |

Host核验包括核心协议9、RDMA能力、reference capability与独立schema1/64word名称、
旧stage schema3、G4/G16选择及安装包构建身份。三个wheel的源码指纹均为
`056855f1e284e1738a7dc2943e103c3e3e80bcc901a0042086771e6372b569ed`。
SHMEM固定为`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`。

最终状态为`complete exit=0`；三个构建均无编译错误。现有SDK宏重定义/弃用警告保留。
完整路径相对于隔离工程：

- `.logs/reference-20261026-01/wheel-REF.log`、`wheel-OFF.log`、`wheel-ON.log`。
- 同目录`import-REF.log`、`import-OFF.log`、`import-ON.log`及`sdk.log`、`status`。
- `.artifacts/reference-20261026-01/{REF,OFF,ON}/`保存各自wheel。

wheel名均为`ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl`：

| 构建 | 文件字节数 | SHA256 |
|---|---:|---|
| REF | 3185296 | `1287ee6d50257ff2482a22770ec5ee900d2ebcb91955b951ae6ac80105765c78` |
| OFF | 3073601 | `4f7712316531db98a0d27710ce041074b7946ac1e71892339650fb0fef9e8151` |
| ON | 3617268 | `39561e971bc63f74a8db3067efa77bf060bd7ccd08ac8acb454c675a3679bba4` |

以上仅标识本机本次产物；不要求不同目标机的二进制hash相同。
目标机完整命令见[README](README.md)。设备正确性、真实逐核耗时及相对旧打点/OFF的
完整task扰动尚未测量，不能据编译或CPU通过宣称性能收益。
