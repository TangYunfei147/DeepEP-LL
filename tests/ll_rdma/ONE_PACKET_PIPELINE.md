# 一包深度的注册GM暂存/put流水

2026-10-29，kernel/ll-ndr-ab-per-core-qp。基于81a0ec0，通信实现身份26、reference schema4。保留原token/路由所属S核、FIRST_HOP/DIRECT顺序、q的Vector计算、1:1 S/R、路由SO与payload SO。不引入8QP映射或小BS动态切分，SDK不改。

## 流程

```text
共用SendPacket：加载/复用输入 → q与包尾 → 构造原输出lane
跨机A：DataCopy(UB_A → 原GM批次槽A) → SetFlag(MTE3_S,7)，暂不put
下一包B：用不同的原输出lane准备B（B可为机内或跨机）
提交A：WaitFlag(MTE3_S,7) → put A → 必要时按原批次quiet
提交B：机内仍是原DataCopy；跨机则暂存B并继续一包深度
尾包：循环结束显式SubmitPending，随后保留原尾批drain/quiet
```

- 只有一个staged-but-not-posted包；没有新增GM/UB缓冲、lane或事件ID。增加的是Scalar状态及一个指向原输出lane的LocalTensor句柄。
- 只有原lane数>1才延后提交；单lane回退立即等待/put，避免准备B覆盖A后SDK scratch又破坏B。
- 旧包的UB一直保留到其暂存完成和SDK使用scratch返回。提交A时SDK使用UB_A，不使用已准备好的UB_B。
- 无论下一包是机内还是跨机，先post旧A，再提交当前B，保持原提交顺序；不保证不同链路实际到达顺序。
- 原MTE3_V lane依赖保留；MTE3_S事件7从立即Set/Wait改为Set后跨一次包准备再Wait，调用SDK前已消费该事件。SDK借用的UB仍由其原退休契约保护。
- 原每核128KiB staging、最多8包批次、每peer一次quiet不变。满批只在最后一包已经post之后退休；quiet返回之前不复用GM槽。
- 最后一包在w39之前完成post，保持“路由/packet发送提交”边界；原尾批quiet仍在后续区间。
- 这是下一包准备与当前暂存的重叠，不另加一套跨机输入预取算法。共享原输入加载、缓存、量化、打包、lane轮转逻辑。

## 打点适配

延后post时，当前准备包可能已变成B，因此将“是否为第二跨机包”在暂存A时保存，post/quiet按保存的A身份计时，不能再根据当前SendBegin状态判断。

schema4仍为每核5120B，R核日志与按成功扫描批次展示方式不变；w90～w95新增：

| word | 含义 |
|---:|---|
| 90 | 第二包暂存提交及批次管理完成 |
| 91/92 | 开始等待/完成等待第二包staging |
| 93 | 实际选择延后post的跨机包数；不表示每包都获得有效重叠 |
| 94 | 准备第二包后提交前一包的区间累计 |
| 95 | 第二包是否延后post |

S表增加延后间隔、剩余等待、前包提交耗时和延后包数。

- 第二包调度 = Enqueue入口−SendPacket入口−成员q时间−提交前包的时间；主q仍在入口之前，不计入。
- 暂存到put前的整体经过包含下一包准备，不等于DMA纯耗时。延后间隔w91-w90与剩余等待w92-w91单列。
- put调用仍不含满批quiet；第二包是否触发满批依据暂存身份记录。
- 跨机与本组的时间区间可能因流水而重叠，即使packet类别顺序没有交错。两种标记分别输出，不相加。
- 保留所有原同步语义，未增加用于测量单包传输完成的quiet。put返回不等于远端消费完成。
- schema2结果用原提交导出，不能混入schema4。全rank完整重编并核验相同源码/协议。

## 对比和测试

一包深度可能隐藏部分staging等待，也可能延后首包NIC提交；没有实测前不宣称加速。对照为81a0ec0，同SDK/核数/shape/seed/采样条件分别观察完整OFF dispatch与ON诊断，不将ON内部区间当作OFF收益。

两机先更新相同分支；NODE取0/1，MASTER取节点0真实IP。新run名，不加--skip-build：

```bash
git fetch origin &&
git switch kernel/ll-ndr-ab-per-core-qp &&
git pull --ff-only

# 正确性与完整OFF性能（FP8配置的combine仍为BF16）
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" ndr-onepacket-perf-001 \
  --mode perf --tokens 1 2 4 8 16 24 32 48 64 96 128 192 256 \
  --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8

# 单独构建reference完整wheel并诊断
bash tests/ll_rdma/one_click.sh "$NODE" "$MASTER" ndr-onepacket-ref-001 \
  --mode reference --tokens 1 2 4 8 16 24 32 48 64 96 128 192 256 \
  --capacity 256 --hidden 3584 --topk 8 --experts 384 --dtype fp8 --reference-iters 5
```

模式切换不能用旧wheel跳过构建。原始数据保留，回传单个≤60000字节报告即可。

CPU验证包括216组延迟MTE3/破坏性SDK scratch/延迟NIC模型，1/2/8 lane、512B～128KiB包、满批和尾批、机内/跨机交错、事件退休，以及下一包为本组时第二跨机包身份不串包。尚未运行CANN编译、设备正确性或性能。
