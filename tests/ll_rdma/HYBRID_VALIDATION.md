# ABI 4 混合通信验证记录

> 历史验证；当前ABI5专家诊断尚待目标机编译及设备验收，见[当前分支诊断说明](../ll_stage_timer/README.md)。

实现基于 `kernel/ll-ndr-hns1825` 的 `1f62860`，仅修改该分支；泛化+打点分支 `kernel/ll-configurable-groups` 保持 `5b76f57`。

## 已通过

- CPU 共 **367 项**：原选定回归在新策略下 364 项通过（`.logs/hybrid-cpu.log`），新增 group policy/诊断 metadata/错误物理 rank 布局三项通过（`.logs/hybrid-policy.log`）。launcher 的目录迁移等相关测试另外复核通过。
- 生产 helper 的 CPU 异步模型：跨机 peer=16 执行两阶段写；本机 peer=1 的包/计数/读取不提交 RDMA；跨机读取拒绝；保留复用、范围、QP、CQ 故障和预取事件归属检查。测试替身不提供 RDMA get 接口。
- 生产 Classify/ready 方法覆盖 EP16/32/64/128/256，分别采用 G=4/G=16，验证所有实际核数边界的通知与清理。
- key/layout 检查：dispatch key 位于 byte21、结构32B；combine key位于byte15、结构16B。EP≤16选择4，EP>16选择16；错误key拒绝；任意同逻辑组成员必在同一个16-rank物理机组内。EP2/4/8仍被原生布局准入拒绝。
- 同一 wheel 编译了 `<4>`/`<16>` 两个设备模板；RDMA ON + timer ON、RDMA ON + timer OFF 完整 wheel 编译、链接、安装和安装后检查通过。安装后实际调用 Host tiling 接口检查 EP16→4、EP32→16，核心协议4；最终保留 timer OFF 构建。
- CANN 原语程序编译/链接通过。**A5 实际设备0/1、两个真实 rank、每 rank 56 AIV、三轮机内 MTE 验证通过**：512/1536/15872B 包、compact/keep-flags marker、count、ready、本地 peer 读取及变化数据的缓冲复用。日志位于 `.logs/rdma-smoke-8zchlc5v/`，两 rank 均输出 `PASS ... local_mte_read=1`。没有创建伪造的原生 LL 域，没有使用其他设备。
- 所有 A5 运行四个可见性变量固定0,1；日志、缓存、临时文件在工程内。相对路径、带空格目录和不同 cwd 的 CPU plan/环境路径回归保留。

## 尚待目标机

A5 当前前检仍未枚举到 RDMA 设备（`.logs/hybrid-preflight.json`）；本轮实测仅为机内 MTE，**不作为跨机 RDMA 写验证**。跨机 put/完成标志可见性、真实 HNS_1825 通道资源、EP16 G4 与 EP32 G16 完整 LL 数值及性能，按 [README](README.md) 由用户在目标机执行。

正式性能协议未修改：10次预热、100次采集、Exp＋scalar all_reduce、每kernel/task最后50条；timer诊断和原语不替代性能基线。没有匹配实测基线，不宣称加速。

当时 SHMEM 为锁定基线及 ROCE-only 72-QP 补丁（该补丁于2026-10-16撤销，旧SDK不可复用；本段仅为历史事实）；未另建SDK树、未升级驱动或系统依赖。生成 SDK 修改、wheel、原始日志和数据不提交。
