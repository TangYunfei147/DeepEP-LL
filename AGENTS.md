# 脚本路径与实验约束

- 所有脚本的工程文件、默认输入输出、日志、缓存和临时目录，以工程根目录为基准，通过脚本自身位置定位，不依赖检出目录名称或调用时工作目录。
- 禁止硬编码开发机绝对路径。相对CLI参数和环境变量路径从工程根解析；外部CANN工具链由用户显式配置，SHMEM等子仓依赖遵守下述仓内流程。运行时可以解析为绝对路径，但不能把当前机器路径写死进生成脚本。
- 修改路径后验证异名目录、包含空格的目录、非工程工作目录及多机结果汇总。生成产物和原始数据不提交Git。
- 审计或修改测试kernel前先查看 tests/ll_probe/KERNEL_MAP.md。four_layers/snapshot 是生成目录，修改提取规则后重新生成。
- 性能采样保留既定10次预热、100次采集、每次Exp和scalar all_reduce、各kernel/task最后50条的协议；Event/API/task时间不可互相替代。
- A5_241只用于设备0/1的正确性验证；实际目标环境的通信域与性能由用户运行。不能将两rank验证当作原生LL EP16+验收。
- SDK依赖统一使用仓内子模块：`git submodule update --init --recursive --depth 1`，`unset SHMEM_ROOT CATLASS_ROOT`，`python scripts/build_shmem.py`，设置`DEEPEP_NPU_ARCH=Ascend950`，将`third-party/shmem/install/shmem/lib`加入`LD_LIBRARY_PATH`，再依次安装`requirements-build.txt`并执行`python -m pip install -v --force-reinstall --no-build-isolation --no-deps .`。子仓及构建产物不得放在工程外或另建探针SDK树；不通过外部SHMEM/CATLASS覆盖该流程。

操作说明见 [tests/ll_probe/PATHS.md](tests/ll_probe/PATHS.md)。
