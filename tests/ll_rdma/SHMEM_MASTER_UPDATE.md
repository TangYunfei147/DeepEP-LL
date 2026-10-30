# SHMEM官方master修复更新（2026-10-30）

本次仅更新用户指定的两个分支：`kernel/ll-ndr-metadata-cores`、`kernel/ll-ndr-ab-per-core-qp`。
二者统一从`8e3fc610f91ae9e5e8cf13748cfb74826b272c2a`更新到实时核实的官方master：

`4c671684e1698aec199619850a3120bfdf85c95e`（上游提交时间2026-10-30 10:59:03 +08:00）。

`third-party/shmem` gitlink与`dependencies.lock.json`同步锁定这个提交；构建时不追逐移动master。
六个锁定公开头的LF规范化SHA256已重新核对，本次均未变化。SHMEM工作树不加私有补丁。
其他历史A/B、小B、hns1825和独立LL工作区不更新；旧性能结果也不改写。

## 主要上游变更

- [e658bbce：HNS_1825 WQE ULP修复](https://gitcode.com/cann/shmem/commit/e658bbcec97ec3e653da8bc3d9f0d602a958b690)：任务段`ulp`使用`lkey & 0xffffff00U`，替代旧`lkey & 0xffffU`；低8位为ULP flags。
- [9c7770c3：UDMA队列状态发布修复](https://gitcode.com/cann/shmem/commit/9c7770c34c55da9f88ae524dd77bc1dd37ef9d4e)：doorbell前发布队列生产索引/CQE目标，并调整完成索引刷新及聚合提交顺序。
- 同批包含性能示例FFTS地址获取时序修复、新增SIMT UDMA接口及对应测试/构建功能检测。它们不等同于本项目已经启用新SIMT路径。

项目使用的`aclshmemx_roce_qp_put_nbi/quiet`公开接口及RDMA封装未变化，不需要改LL调用。
kernel、wire、QP分工、采样协议及分支原实现身份保持各自原样；SDK revision和完整库/头指纹单独拒绝旧SDK或混装。
HNS_1825与严格整数PI-on-chip 0/1校验保留，不手改manifest或恢复私有SDK补丁。
新增`tests/runtime/test_shmem_pin.py`核对gitlink、锁定revision、已初始化源码及公开头指纹。

## 目标机更新与重编

本地只做源码/API检查和CPU回归，未执行CANN编译或NPU验证。
此次SDK源码确实变化，不能只重装wheel或继续使用旧SDK。各主机先结束使用该工程SDK的作业，加载已有CANN环境，然后在各自分支工程根执行。

使用现有NDR一键入口（例如2机，每机NODE_RANK分别为0和1、MASTER_ADDR相同；测试名为全新名称）：

```bash
if git pull --ff-only; then
    if bash tests/ll_rdma/one_click.sh "$NODE_RANK" "$MASTER_ADDR" shmem4c67168-perf-001 \
        --num-nodes 2 --mode perf --rebuild-shmem --tokens 8 --dtype bf16 fp8; then
        echo "PASS: rebuilt SDK/wheel and completed OFF dispatch/combine tests"
    else
        echo "FAILED: keep this terminal; inspect .logs/ndr-one-click"
    fi
else
    echo "Git update failed; keep the terminal and inspect the error"
fi
```

metadata分支沿用已选的`DEEPEP_LL_METADATA_CORES`（默认4）；每核QP基线保留原S/R分工。
一键入口显式使用`git submodule update --init --recursive --depth 1 -- third-party/shmem`，只拉取SHMEM及其必要依赖，禁止拉CATLASS。
显式`--rebuild-shmem`更新本工程SDK，随后重建完整wheel，所有rank一起更新。
已有`--skip-build`仅适用于本次新SDK和wheel已经成功构建并通过身份检查的后续测试，不能用于首次升级。
SDK重建失败不登记新身份；不要手动修改旧manifest绕过检查。不删除历史结果、日志或其他工程SDK。
