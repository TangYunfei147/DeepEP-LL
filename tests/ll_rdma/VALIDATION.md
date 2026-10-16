# 历史记录：以下为 ABI 3 全 RDMA 版本的验证

当前 ABI 4 混合 MTE/RDMA-put 版本请查看 [HYBRID_VALIDATION.md](HYBRID_VALIDATION.md)。下列结果不代替新版本验证。

# 本次验证记录（2026-10-15）

> 2026-10-16：下述 ROCE 72-QP 补丁及其构建/复用特判已撤销，脚本已删除。这些记录仅保留当时事实，不代表当前构建流程或当前验证结论；旧补丁 SDK 不可继续复用。

代码在独立 `DeepEP-LL-kernel` checkout 的 `kernel/ll-configurable-groups` 上实现，未替换原实验 checkout。基线 `cb59bff99ecd39d27ef57788c5e6fd0009e4e636`；SHMEM Git revision 保持 `4dd80e2129da42bd773982c2bb3758af1ed0bba8`，生成的 ROCE-only 72-QP 三处修改由 `scripts/shmem_rdma_patch.py` 复现。

| 验证 | 结果与范围 |
|---|---|
| 统一 SHMEM HNS_1825 SDK 构建 | 通过；`-enable_rdma -rdma_backend HNS_1825`，标准仓内安装目录 |
| SDK 身份、哈希与 `--if-needed` 复用 | 通过；记录 backend/patch ID/source patch hash |
| 完整 wheel：RDMA OFF / timer OFF | 编译、链接、安装及安装后检查通过；选择 RDMA 在 Host 初始化前明确拒绝，没有 MTE 回退 |
| 完整 wheel：RDMA ON / timer OFF | 编译、链接、安装及安装后 ABI=3、RDMA 能力检查通过；最终保留此构建 |
| 完整 wheel：RDMA ON / timer ON | 编译、链接、安装及安装后能力检查通过；不作为性能基线 |
| 独立双 rank put/get/count/ready 程序 | CANN 设备编译与 Host 链接通过；包含 compact/keep-flags marker 检查；未执行网卡收发 |
| CPU 回归 | **373 passed**，包含 Runtime/API/lifecycle、LL 路由和布局、timer/profiler 协议及新 RDMA 测试 |
| 生产 transport helper 的 CPU 异步模型 | 11484 次 packet 发布，实际核数 16..72 所有偶数；检查数据先于 flags、重复复用、counts/get/self、地址越界、QP 作用域、CQ 故障不发布 flags、超时和 8 路预取事件归属 |
| 逻辑组和 ready | G=1..16 的生产方法 CPU 执行；检查各支持布局中的 ready 覆盖、消费/清理和动态核数 |
| 迁移 | 不同目录、含空格目录、不同 cwd 的 EP16/32 CPU plan 与工程日志路径通过；不表示迁移后的硬件 RDMA 验收 |
| A5 环境前检 | Hcomm 必需符号齐全；四个可见性变量均为 `0,1`；未枚举到 `/sys/class/infiniband` 设备，状态 **blocked** |
| 实际 HNS_1825、完整 LL、多机、性能 | **未验收**，按 README 由用户在目标环境执行 |

没有用 MTE 运行替代 RDMA 验证，没有启动虚假的 EP16/32，也没有宣称加速。安装后测试只检查构建能力和非法/未编译后端的拒绝路径，不创建 SHMEM 域。A5 本轮没有运行实际 RDMA kernel。

本次 SDK manifest 的 compiler/version.info 记录为 `Version=9.2.0`、`timestamp=20261009_000323778`；工具链由外部 CANN_ENV 指定。此处保留实际元数据，不根据目录名称推断版本。构建入口仍接受用户确认的 CANN 9.3.0，目标运行兼容性按实际结果记录。

主要日志位于工程内 `.logs/ndr-*`：SDK build/reuse、wheel-off-final、wheel-on-final、wheel-timer-on、installed-off-final、installed-on-final、installed-timer-on、smoke-build、cpu-final、preflight。原始日志、wheel、SDK 和模型二进制未提交。目标回传清单和判据见 [README](README.md)。
