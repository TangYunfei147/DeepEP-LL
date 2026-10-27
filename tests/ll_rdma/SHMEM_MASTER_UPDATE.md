# SHMEM 官方 master 更新（2026-10-25）

> 2026-10-28更新：当前协议11，路由表同WQE尾FLAG发布，每对跨机卡共8个QP；本文旧QP/独立commit描述作为历史记录。当前实现见 [ROCE_SO_ROUTE_8QP.md](ROCE_SO_ROUTE_8QP.md)。

本次按用户要求，将 `third-party/shmem` 从 dev 的
`4dd80e2129da42bd773982c2bb3758af1ed0bba8` 更新到官方
`https://gitcode.com/cann/shmem.git` 默认主线 master：

`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`（上游提交时间2026-10-23）。

版本来自本次实时 `ls-remote --symref` 核验；`.gitmodules`、gitlink、
`dependencies.lock.json` 的 ref/revision 和六个公开头指纹同步。
构建继续固定这个 commit，不在目标构建时追逐移动分支。
上游源码保持干净，没有恢复 QP 扩容或其他私有 SDK 补丁。

## 必要兼容

- 新初始化属性用 format-2 version 识别追加的 `runtime_config`。
  Session 保留 `attr{}` 的默认版本，UDMA shared jetty 改用该结构字段，
  不再调用上游已移除的 `aclshmemx_set_shared_jetty`。
- 官方 master 不再提供旧 sparse peer-QP mask API。旧 HT/UDMA 用户必须显式
  `DEEPEP_SHMEM_DENSE_QPS=1` 才选择密集通道，否则在初始化前明确拒绝。
  NDR 的 MTE+ROCE 路径不使用该 UDMA 设置。
- 原生 ROCE 每 peer 上限仍为56 QP，实际 AIV 每核独占一个QP的约束不变，
  不自动缩核、不共享QP、不把QP数量解释为物理端口数量。
- 新 HNS 把对称目的地址翻译移到 backend 内部。`LlTransport::Address/Put`
  继续传本地对称地址；不能再提前调用远端翻译而产生重复偏移。
- 新 HNS 根据固件选择 `ACLSHMEMI_HNS_1825_PI_ON_CHIP=0/1`。
  主仓从 SDK 成功构建后的 `aclshmem_device` Make/Ninja 实际编译规则取值，
  写入 `deepep-sdk.json` 的 `build.hns_1825_pi_on_chip`，并传给 LL 和独立探针编译。
  缺失、冲突和非法值拒绝登记；已有编译图与身份不一致时拒绝复用。
  SDK build 身份参与 native cache key，避免0/1模式共用错误的对象。

外部SDK登记也必须提供实际构建图（`record(..., build_root=...)`，CLI为
`--build-info <JSON> --build-root <build目录>`），并与申报的PI值一致。
只有头文件/库和手填模式、没有构建图时不生成新manifest；`--check`/`--verify`
仍只读校验已有身份。

SDK 自身在固件探测不可用时可能发出警告并选默认模式。本仓记录它实际编入的值，
不代表已验证该模式与目标固件匹配；目标构建应检查上游配置输出。

## 目标更新

这次依赖版本确实变化，旧 SDK 不能只靠重装 wheel 继续使用。
先停止使用该 SDK 的作业；在各 rank 对应工程加载已有 CANN 环境和工程 Python 环境，
然后在工程根执行：

```bash
git pull --ff-only
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
python scripts/build_shmem.py --rebuild
export DEEPEP_NPU_ARCH=Ascend950
export LD_LIBRARY_PATH="$PWD/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

正式性能使用 timer OFF；采集阶段诊断时全 rank 安装同一 timer ON wheel。
不改驱动、不安装系统依赖、不混用其他分支SDK。
显式重建开始前仅撤销旧生成身份 manifest；构建失败后不会留下可复用的旧身份，
成功后才登记新头文件、库、固件模式和版本。

## 验证范围

本地可验证依赖锁、身份拒绝/复用、0/1模式传播、CMake构建图及CPU协议模型。
Windows缺少的Linux生命周期用例不能标为通过。
实际 CANN/Ascend950 编译、目标HNS固件兼容、两机数据正确性和时延仍需目标环境验收。
新版官方 device barrier 的变化不作为自动撤回现有 Host HCCL 生命周期协调的理由。
