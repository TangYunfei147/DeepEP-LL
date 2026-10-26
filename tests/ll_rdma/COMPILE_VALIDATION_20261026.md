# NDR当前分支编译修复与验证

2026-10-26按本会话授权，在27的独立编译目录验证当前NDR分支，全程未启动NPU。
失败源码为`7a37ae7`，修复并完成CANN编译的源码为`4b4ba9482f8e6c1eb0df8bc3f52d04ad4b1358d2`。
后续验证记录提交仅修改文档。

## 根因和修复

新增的`combine_clear_flags.h`为取得bank大小，包含了Host专用的
`ep_memory_server_dedup_layout.h`。这把Host布局校验函数带进Bisheng的AICore编译单元。
这些Host函数调用被标记为`__aicore__`的公共helper，编译器因此报错：

```text
error: no matching function for call to 'LlCoreCountValid'
note: candidate function not viable: call to __aicore__ function from __host__ function
```

同一错误还涉及`LlGroupForWorld`、`DispatchServerDedupCountSendUbBytes`、
`DispatchServerDedupCountLayout`和`DispatchServerDedupDataLayout`，合计8个错误位置。

修复仅将该include换成定义bank常量、可用于设备端的
`ep_memory_server_dedup_count_layout.h`。未把Host校验函数改为设备函数，未删除校验，
未改变flag清理、输入输出、控制布局、数值路径或SDK源码。

## 实际验证结果

| 项目 | 结果 |
|---|---|
| 本地flag初始化、shape复用和生产时钟约束回归 | 7通过 |
| 完整timer OFF wheel | Bisheng设备编译、Host扩展编译、链接和打包通过 |
| 完整timer ON wheel | Bisheng设备编译、Host扩展编译、链接和打包通过 |
| OFF wheel隔离安装与Host导入 | protocol9、schema3、RDMA1、timer0通过 |
| ON wheel隔离安装与Host导入 | protocol9、schema3、RDMA1、timer1通过 |

Host校验只导入安装包和native扩展并读取静态能力；没有导入`torch_npu`，没有创建Runtime/Session、
tensor或调用NPU API。设备正确性和性能未测试。

目标架构保持Ascend950。SDK为锁定的`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`，
复制原产物到新隔离工程后通过现有身份/哈希校验复用，没有修改SDK或重建原工程的SDK。
该SDK登记的CANN版本为9.2.0、timestamp为`20261014_093543763`，PI模式为1。
SDK已有宏重定义/弃用等warning仍保留，没有把warning误报为本次编译失败。

新venv首次尝试另遇到PATH缺CMake，尚未进入Bisheng；将授权根中既有CMake加入本次进程PATH后解决。
该环境准备问题与上面的源码错误分开记录，未安装或修改系统工具链。

## 产物与日志

隔离工程位于原授权根下的`ndr-compile-20261026`，旧`ndr-compile`保持。
以下路径均相对新隔离工程：

- `.logs/wheel-OFF-host-device-error.log`：原始Host/AICore编译错误。
- `.logs/wheel-OFF-env-missing-cmake.log`：首轮环境准备错误。
- `.logs/wheel-OFF.log`、`.logs/wheel-ON.log`：修复后完整构建。
- `.logs/import-OFF-host.log`、`.logs/import-ON-host.log`：纯Host校验。
- `.artifacts/OFF/`、`.artifacts/ON/`：分别保存两种wheel，不相互覆盖。

wheel文件名均为`ascend_deepep-0.1.0.dev5-cp311-cp311-linux_aarch64.whl`：

| 变体 | 文件字节数 | SHA256 |
|---|---:|---|
| OFF | 3047343 | `34b5a46a6d1a8b3fe40bb7b23ae703c65b0279ac8b3ba1f62becfc92a7242168` |
| ON | 3632416 | `49932692675e894bbcbc8370cff956714c594fad00fd31b816a826076dad298c` |

这些是本次工具链下的构建产物摘要，不要求其他机器生成相同二进制哈希。

## 仅重编命令

加载目标机已有CANN环境和构建venv，确认工程内SDK已准备后，在当前分支工程根执行：

```bash
ndr_compile_only() {
    git pull --ff-only || return
    local root mode
    root=$(pwd -P) || return
    export DEEPEP_NPU_ARCH=Ascend950
    unset SHMEM_ROOT CATLASS_ROOT
    export TMPDIR="$root/.tmp" TMP="$root/.tmp" TEMP="$root/.tmp"
    export XDG_CACHE_HOME="$root/.cache" PIP_CACHE_DIR="$root/.cache/pip"
    export PYTHONDONTWRITEBYTECODE=1
    export CUDA_CACHE_PATH="$root/.cache/cuda" TORCH_EXTENSIONS_DIR="$root/.cache/torch_extensions"
    export TRITON_CACHE_DIR="$root/.cache/triton"
    export SHMEM_LOG_PATH="$root/.logs/shmem"
    export ASCEND_WORK_PATH="$root/.logs/ascend" ASCEND_PROCESS_LOG_PATH="$root/.logs/ascend"
    export LD_LIBRARY_PATH="$root/third-party/shmem/install/shmem/lib:${LD_LIBRARY_PATH:-}"
    mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SHMEM_LOG_PATH" "$ASCEND_WORK_PATH" || return
    for mode in OFF ON; do
        mkdir -p ".artifacts/compile-check-$mode" || return
        DEEPEP_LL_STAGE_TIMER="$mode" python -m pip wheel -v \
            --no-build-isolation --no-deps \
            --wheel-dir ".artifacts/compile-check-$mode" . || return
    done
}
if ndr_compile_only; then
    printf '%s\n' 'OFF/ON wheel build completed'
else
    printf '%s\n' 'Build failed; current shell remains available'
fi
```

此命令只构建wheel，不启动dispatch/combine或设备测试。本次include修复无需变更SDK或协议标识。
