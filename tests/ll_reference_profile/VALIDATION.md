# 本地验证记录

验证日期：2026-10-30。当前分支为 `kernel/ll-g4-small-b-prefix`，参考打点来自
`kernel/ll-ndr-metadata-cores@c3d86406733fd28fe4e081e563eaf7e51321c18f`。
当前MTE格式为schema7。没有进行SSH、CANN编译、NPU正确性或性能测试。

## 结果

- 项目CPU回归：**2032通过**，243环境跳过，59排除。运行排除device/framework的50项、8个依赖Linux可执行夹具的用例，以及下面列明的1项既有API合同偏差。
- 独立本地C++记录器/Host检查：**13通过**。覆盖ON/OFF、无效/空指针不读时钟、10000次空轮询只读一次等待起点、每个成功批次三次clock、私有记录/缓存范围、剩余UB包日志，以及5个selector×7个入口的分派与144/88B布局。
- 真实stream/pair头文件的多线程CPU模型：**2通过**。仅同步仿真，不是设备验证。
- 补充count VF/实际轮询方法、Host身份与拒绝旧链接符号检查：**14通过**。其中存在与其他集合重叠的检查，不相加为总数。
- cb59/small生成器检查通过；22份算法文件完整业务token与有序Scalar/DMA/同步序列对照通过；原SDK锁、gitlink、wire及core协议未改变。
- 新capture/采样/解析/Excel/交付检查通过；包内旧模块残留会拒绝加载，构建暂存仅清理指定旧模块。Bash入口通过本地 `bash -n`。
- 原 `no_timer_ab.py` 的固定源码投影、三wheel编排和显式续跑回归保留并通过。该历史实验不使用当前schema7或Exp100采样。

全量运行日志留本地 `.tmp/reference-final-regression.log`，不提交测试产物。
新旧测试集合有重叠；本记录不将CPU数量解释为NPU验收或性能收益。

## 已知非本次失败与环境边界

首次未排除的全量检查中，`tests/api/test_contracts.py::test_signature_matches_project_contract[ElasticBuffer.__init__]`
仍按旧合同期待 `use_1g_huge_pages=True` 且没有 `transport`，而当前代码为 `None` 和 `transport='udma'`。
已对照e4d114c，构造函数参数AST与本次之前完全相同；本次没有修改公开构造函数来迎合旧合同。

8个Host编译器选择夹具依赖Linux脚本执行/符号链接行为，在Windows不能作为本次代码失败或编译验收。
其余跳过项包括本环境没有Torch、默认PATH无CMake/g++、Linux专用生命周期和符号链接权限；
本地C++验证使用已有Zig编译器单独执行。后续退役了5个仍引用已删除旧记录器的专属C++用例；
其职责由当前reference ON/OFF、角色解析和链接检查覆盖。

## 目标机仍需完成

按 [README.md](README.md) 的新入口重编完整wheel、执行正确性及实际capture，并检查报告中的真实
源码/算法/分核/路由身份。只回传一个≤60000字节的 `return_report.md`。
新协议为 `initial-reduce-exp100-batch-v3`，不能将其与旧逐轮Exp/all-reduce数据视作同条件A/B。
