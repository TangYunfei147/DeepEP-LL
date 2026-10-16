# NDR wheel / kernel 增量构建

适用于kernel/ll-ndr-hns1825。继续使用原pip安装命令，无需新增缓存开关。
本次仅修改构建规则，不改变kernel算法、ABI或运行时采样协议。
上述对象共享针对默认NDR核心wheel；MegaMoE等可选组件继续保留原有构建目录隔离。

## 改进与边界

- 相同配置重复构建：复用配置，不再无条件运行Host和Device的configure。仍进入CMake build检查源文件、编译器depfile和CMake规则；不能按源码摘要直接跳过build。
- LL timer不再拆分整套native缓存。五个非LL设备编译单元与Host对象复用，两个LL单元分别保留OFF/ON对象，只构建选中版本。首次切换仅补编两个LL对象；已构建两种版本后切换只重链接。仅当前选中版本进入最终libdeepep_kernels.so。
- 单个LL源或其头文件变化由依赖图处理，不能复用过期对象。尚未选中的timer版本下次选中时也检查依赖。
- 首次构建仍编译完整七个设备单元和Host代码。本次没有减少模板实例，不保证冷编译显著提速。
- Python、pybind、Host/Bisheng编译器、SDK身份、编译flags或构建类型变更仍保守隔离native缓存；本次不跨不兼容工具链复用。
- 删除或修改Host/Device cache、缺失构建图、参数变化会重新configure；失败不保存可复用的新配置签名。独立执行CMake仍支持自动规则再生成。
- pip仍会打包和安装wheel；--force-reinstall不等于清空工程内CMake对象。不要为普通源码修改删除build目录。首次使用新缓存布局会创建新目录，不自动删除旧缓存。

## 目标机复现

先在目标环境加载CANN，并在本仓库根目录执行。外部SDK工具链仍来自当前机器环境。

```bash
source scripts/ll_timer_env.sh
git submodule update --init --recursive --depth 1
unset SHMEM_ROOT CATLASS_ROOT
export DEEPEP_NPU_ARCH=Ascend950
python scripts/build_shmem.py
python -m pip install -r requirements-build.txt
```

保持同一Python环境、工具链、SDK、flags和并行度，连续运行：

```bash
mkdir -p .logs/native-build
set -o pipefail
index=0
for timer in OFF OFF ON OFF ON; do
  index=$((index + 1))
  export DEEPEP_LL_STAGE_TIMER="$timer"
  python -m pip install -v --force-reinstall --no-build-isolation --no-deps . \
    2>&1 | tee ".logs/native-build/${index}-${timer}.log" || break
done
```

第一轮用于准备缓存，并非必然冷构建。预期第二轮没有C++对象编译；首次ON仅新增两个LL编译；
之后OFF/ON只重新链接相关库，没有再次编译未改变的对象。每次wheel都需全rank安装一致版本，测试结束时注意最后安装的是timer ON。
若需要冷构建对比，使用新的独立clone及其工程内依赖，不删除当前工作区或其他任务的缓存/改动。

每次成功构建末尾打印报告位置：native构建目录的build-timing.json，以及保留历次记录的build-timings.jsonl。
报告包含preflight/configure/build/install/total秒数、timer、jobs、源码摘要和cache key。
configure_skipped只表示跳过显式configure；若规则变化，CMake仍可在build阶段自动再生成，其时间计入build。
这些是native构建阶段耗时，total不包括整个pip解析/打包/安装过程，也不是kernel执行性能。
失败轮次查看对应log，不拿前一轮成功报告当成本轮结果。

## 本地验证

CPU测试使用真实CMake/Ninja与输出对象/depfile的模拟编译器，覆盖no-op、规则变化、SDK头文件变化、
单个LL源变化、timer双向切换、配置失败以及含空格路径。构建调度测试还覆盖Python源码身份更新、
缓存删除/外部修改、jobs变化及失败恢复。它们验证依赖与调度，不证明Bisheng代码生成或设备正确性。
目标CANN编译耗时及wheel安装后的设备正确性由用户验证；不报告未实测的提速百分比。

2026-10-16隔离快照回归：201 passed、10 deselected（8个Linux shell检查及2个Linux shell/Torch依赖检查未运行）。真实CMake/Ninja+模拟编译器覆盖6项；新增父构建调度覆盖15项。首次深层临时目录触发Windows路径长度限制，缩短临时路径后全部已选测试通过；含空格路径仍覆盖。未执行CANN或设备测试。
