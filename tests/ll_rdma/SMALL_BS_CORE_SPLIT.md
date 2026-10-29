# 小BS按token数量切分S/R的独立实验

分支 `kernel/ll-ndr-small-bs-core-split`，基于已完成包诊断的 `833667c`。通信协议25、reference schema3，所有rank完整重编wheel，禁止与父分支混装。SDK及实际总AIV查询不变，不改SDK原生56QP上限。原分支保持原1:1。

## 分工

实际总AIV数为C，实际token数为B（不是容量M）：

```text
0 < B < C-1：S=B，R=C-B
其他合法B： S=C/2，R=C/2
```

严格按“小于”实现，所以触发实验时至少保留2个R核，满足至少1个计算核的要求。B=C-1不触发实验。

| C=56时的B | 发送核 | 接收/计算核 |
|---:|---:|---:|
| 1 | 1（core0） | 55（core1～55） |
| 8 | 8 | 48 |
| 24 | 24 | 32 |
| 32 | 32 | 24 |
| 48 | 48 | 8 |
| 54 | 54 | 2 |
| 55及以上 | 28 | 28 |

S为低编号核，小B时第i个S核负责token i（且只有这个token），执行原FIRST_HOP/DIRECT发送和源元数据处理，不按peer重分包。它不是只发token的一个packet；该token所需的首跳及DIRECT均沿用原流程。剩余R核承担count统计/发送、count接收与cumsum、payload接收及组内转发。core0仍保留原配对路由SO发布和源退休，避免删除必要控制发布。

不触发小B条件时，发送/接收恢复1:1，count沿用原B<512全核、否则R核的规则。combine保持原1:1，不跟随dispatch切分。每物理核仍使用自己的QP；无8QP分工、无固定配对卡payload中转、q仍为逐次Vector。

## 跨B安全与必要布局变化

同一Buffer依次测试B1、B8、B32、B48等时，S/R数量会改变，不能继续让prefix和ready矩阵跟随当前R数量移动。

- 两个矩阵按C-1个R核固定预留；运行时仍按实际R处理，prefix只保留一份。E和总C在通信器内固定，控制区对B保持固定。
- CountFrame的body保留原source-major位置。commit独立放在E×512B body区后的E×32B区域，不再放在可能被旧frame body覆盖的位置。改变count生产者数量不会把旧body解释为新commit。
- Commit和body分别使用对应的跨source读取步长；仍是原body/commit发布，不新增控制WQE或全rank屏障。
- R核较少时，每核可能负责超过128条count记录，读取计划容量相应扩大；覆盖最多1024条逻辑记录。
- 两个矩阵的实际使用区域按原依赖清零；固定预留、单stream同序、双bank与原combine_reuse依赖保持。
- 编译期上界验证整个count控制区小于1MiB，避开combine控制区和1.25MiB的route区；Host预算用实际S/R核数。

因此与833667c直接比较时，差异包含动态核分工及其配套count布局/读取变化，不能把所有差值都称为纯发送并行度收益。未声称优化成功。

## 打点与报告

继承schema2的S第二跨机包诊断、R核UB包序列；schema3在w62/w63明确记录实际S/R数量。解析、逐核CSV及Excel按这个边界识别角色，不再用总核数/2。原主阶段和包日志大小仍保持每核5120B。

小B时每个S核只拥有一个token，EP32去重通常最多产生一个跨机packet，因此第二包时间为空是正常情况，不应补0。R数量少时可能超过128条日志容量，仍只用lane后的剩余UB并显式标记截断。

## 一键测试

两台目标机先切换相同分支并更新，所有rank完整重编；NODE为0/1，MASTER为节点0真实IP。使用新run名，不加--skip-build：

```bash
git fetch origin &&
git switch kernel/ll-ndr-small-bs-core-split &&
git pull --ff-only

bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" small-bs-split-ref-001 \
  --mode reference --tokens 1 2 4 8 16 24 32 48 54 55 56 64 96 128 192 256 \
  --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8 --reference-iters 5
```

性能测试换新run名，改为--mode perf（去掉--reference-iters亦可）；测完整dispatch+BF16 combine。建议B54/55覆盖56核阈值，并先确认目标实际AIV数。大域、大H或R很少时仍受原Host UB准入约束，不绕过预算。

本地CPU检查覆盖16～72偶数总核、严格阈值、token/专家/输出行恰好一次分配、E32/384/1024、两bank跨B控制区复用、不同count生产者数的body/commit读取、空核与实际角色导出。未执行CANN编译/NPU正确性和性能验证。
