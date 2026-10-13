# 脚本路径与实验约束

- 所有脚本的工程文件、默认输入输出、日志、缓存和临时目录，以工程根目录为基准，通过脚本自身位置定位，不依赖检出目录名称或调用时工作目录。
- 禁止硬编码开发机绝对路径。相对CLI参数和环境变量路径从工程根解析；外部CANN工具链由用户显式配置，SHMEM等子仓依赖遵守下述仓内流程。运行时可以解析为绝对路径，但不能把当前机器路径写死进生成脚本。
- 修改路径后验证异名目录、包含空格的目录、非工程工作目录及多机结果汇总。生成产物和原始数据不提交Git。
- 审计或修改测试kernel前先查看 tests/ll_probe/KERNEL_MAP.md。four_layers/snapshot 是生成目录，修改提取规则后重新生成。
- 性能采样保留既定10次预热、100次采集、每次Exp和scalar all_reduce、各kernel/task最后50条的协议；Event/API/task时间不可互相替代。
- A5_241只用于设备0/1的正确性验证；实际目标环境的通信域与性能由用户运行。不能将两rank验证当作原生LL EP16+验收。
- SDK依赖统一使用仓内子模块：`git submodule update --init --recursive --depth 1`，`unset SHMEM_ROOT CATLASS_ROOT`，`python scripts/build_shmem.py`，设置`DEEPEP_NPU_ARCH=Ascend950`，将`third-party/shmem/install/shmem/lib`加入`LD_LIBRARY_PATH`，再依次安装`requirements-build.txt`并执行`python -m pip install -v --force-reinstall --no-build-isolation --no-deps .`。子仓及构建产物不得放在工程外或另建探针SDK树；不通过外部SHMEM/CATLASS覆盖该流程。

操作说明见 [tests/ll_probe/PATHS.md](tests/ll_probe/PATHS.md)。

- 本分支生产LL固定每物理主机16个连续rank，单机EP16是一个组，跨机按每机16卡扩展。共享HT的8路UDMA QP归属不是LL分组，不可机械替换。修改LL分组须同时核对dispatch分类、Host UB预算、combine ready通知/等待、metadata参考及启动合约，运行步骤见tests/ll_probe/LL16.md。

- 公共project_env.sh禁止默认设置或覆盖设备可见性；A5设备0/1边界仅在SSH、a5_validation_env.sh及专用验证器执行。目标机入口显式选卡，原Python入口继承调用者ASCEND环境。

- LL核数使用PlatformAscendCManager及当前设备AIV查询的较小值，运行时1:1收发；禁止恢复固定64核或将预留容量72当参与核数。当前支持16..72偶数，Session内固定，调试CORE=-2随实际接收起点变化；细节见tests/ll_probe/CORE_CONFIG.md。
