# 运行身份和A/B对照

当前共享接收分支：核心标识9000+S、阶段schema7，默认G4、MTE，dispatch发送后全56核接收；combine保留S/R。[参考打点说明](SEGMENT_TIMER.md)。同一作业必须同wheel、核心分工、schema和SDK身份。

## 正式性能

使用各分支匹配的OFF wheel，固定目标设备、每机16连续rank、EP、S/R、B/M/H/K/E/dtype/路由/seed、Exp和采样协议。仍按完整kernel/task最后50条统计，不混Event/API。不同G可作为明确实验变量，不自动调整G；实际参数写入结果。

```bash
python tests/ll_stage_timer/compare.py results/baseline.json results/candidate.json \
  --kind perf --output results/perf-comparison.csv
```

路径相对工程根解析，输出不覆盖。compare.py允许已知7000/8000/9000协议族的OFF实验对照，并核验各族对应的核数/schema字段。其他设置仍必须一致。

## 诊断

同schema7数据可比较全量累计metrics；采样时间线只含前两次真实区间，不是完整逐token流水图。角色和任务分片变化的逐核行不计算比例。旧schema1/2/5/6可离线解析，但旧GM逐次累计与新延后写出、INIT/KERNEL边界不同，拒绝跨schema6/7诊断耗时比。

```bash
python tests/ll_stage_timer/compare.py results/baseline/summary.json results/candidate/summary.json \
  --kind stage --output results/stage-comparison.csv
```

阶段嵌套不能相加；轮询计数不是耗时；跨rank时钟不能相减。旧expert格式支持离线分析，新模式不再采逐专家GM记录。ON profiler包含末尾同步/写出，不能作为OFF性能基线。
