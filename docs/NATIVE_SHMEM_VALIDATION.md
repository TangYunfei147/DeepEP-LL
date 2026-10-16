# 原生 QP 限制恢复：本地验证（2026-10-16）

两分支删除自加扩容及自动应用、识别、复用豁免、指纹字段和专用测试。当前代码中旧补丁标识和脚本引用为零；历史事实及精确逆向迁移说明保留。锁定SDK源码证据与目标更新步骤见[NATIVE_SHMEM_MIGRATION.md](NATIVE_SHMEM_MIGRATION.md)。

## CPU结果

- NDR stage/host专项：95 passed，1 skipped（没有C++编译器），2 deselected（torch、Bash依赖）。
- 泛化stage专项：82 passed，1 deselected（Bash依赖）。
- NDR runtime/build/identity专项：82 passed，1 skipped（同一C++边界测试），8 deselected（POSIX编译器fixture）。
- 泛化runtime/build/identity专项：81 passed，8 deselected（POSIX编译器fixture）。
- 各组有重叠，不相加为独立测试总数。迁移到含空格目录及不同cwd的Python计划/分析检查通过；Linux shell检查不在本机验收范围。
- 早期完整选择因缺torch/Bash报错；最终明确排除环境依赖后通过。CMake修改已审阅，未执行CANN编译；test_device_incremental仅更新身份fixture，本轮未运行。

在相应分支工程根、安装pytest等测试依赖后复现：

```bash
python -m pytest tests/ll_stage_timer -q -k 'not moved_checkout_with_spaces_and_other_cwd'
# NDR再包含tests/ll_rdma/test_native_qp_limit.py及test_host_contract.py，并排除no_ht_fallback（本地无torch）。
python -m pytest tests/runtime/test_build.py tests/runtime/test_ep_gmm_fused_build.py tests/runtime/test_sdk_identity.py -q -k 'not compiler_symlink and not auto_host_compiler and not host_compiler_fallback and not host_compiler_cann'
# NDR额外：python -m pytest tests/ll_rdma/test_native_qp_limit.py -q
```

SDK完整身份在构建、直接CMake及独立check入口统一检查。未知build字段与缺失身份报错；已有明确本地修改选项保留，取消任何隐式豁免。旧SDK不删除、不自动重建，也不能靠重新登记或修改manifest绕过拒绝。

三个本地checkout的SHMEM子仓均未初始化，无三处源码修改需要撤销。其他任务的三份dirty kernel内容哈希与开始前一致，未纳入本任务提交。无SSH、CANN或设备验证；跨机56-AIV上限来自锁定源码，不代表HNS实机验收通过。