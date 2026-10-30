# LL dispatch 分编译单元与reference打点

当前分支只维护reference dispatch打点，参考源为
`kernel/ll-ndr-metadata-cores@c3d8640`，运行/导出见
[reference指南](../tests/ll_reference_profile/README.md)。

Host `csrc/kernels/dispatch/low_latency.cpp` 只选择入口，7个输入/量化模式分别在
`low_latency/variants/*.cpp` 实例化，公共设备入口在 `dispatch_variant_kernel.h`。
必须保持七TU、禁用unity/jumbo、目标Ascend950和生产O2；不裁模式、不降优化以掩盖编译问题。

`DEEPEP_LL_REFERENCE_PROFILE=ON` 编入当前schema6记录器；默认OFF。
参考开关进入构建指纹和wheel元数据，工具链/SDK/flags变化隔离缓存；每次仍运行实际构建系统，
检查编译depfile，不能仅凭source hash跳过编译。SDK构建身份、完整产物hash与PI合法变体保持。
不改变锁定SHMEM commit，也不给当前MTE添加RDMA/PI编译定义。

旧stage/expert参数和所有计时宏已删除。dispatch Host参数144B，仅一个referenceProfile指针；
combine Host参数88B，无诊断参数。内部符号 `_ref6` 与Host布局检查拒绝旧Host/device对象混用。
新打点不改变业务wire19000+S、UB预算、lane、q、权重、发送顺序和同步。

每核5120B、512B对齐。主点一次clock+GM store；接收ready批次累计两段，日志只占剩余UB，
无空间则显式截断。日志导出所用事件/搬运位于本核业务w21后；最终身份/cache发布位于w57后。
这些属于诊断开销，不能将w21-w0替代未打点完整kernel task。

CPU验证使用实际OFF/ON recorder mock、七入口Host selector、ABI/字典校验、
原53文件业务范围对照、包日志截断/顺序/空核模型、Python解析和输出工作簿检查。
Linux执行/符号链接与CMake环境缺失的用例须明确标记，不作为设备通过。

本地完整相关回归461通过、21跳过；最终字段/导出/部署/清理回归115通过。
旧模块暂存清理与真实setuptools打包检查88通过、8项Linux专属用例未运行。
以上批次有交叉，不能将数量相加。合成11页工作簿重算/错误扫描/逐页渲染通过。

本任务没有SSH、CANN编译或NPU执行。目标机通过 `one_click.sh --prepare-only`
构建并Host检查；去掉该参数后进行真实正确性/采样。需要独立检查OFF时使用新run名和
`--mode perf --prepare-only`，与reference编译身份隔离；不覆盖已有结果。
