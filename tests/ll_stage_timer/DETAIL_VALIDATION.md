# dispatch细分打点验证（2026-10-16）

分支：`kernel/ll-ndr-hns1825`，基于`91b083b`。核心通信ABI保持4，timer schema升为2。
新增18个阶段，原阶段ID保留。张量仍为`[实际AIV,32,8]`，每核2048B、512B对齐；
阶段采用4-word紧凑记录，解析器兼容schema 1旧文件。

## 已通过

- CPU回归77项，覆盖格式/损坏/旧格式兼容、阶段映射、trace轨道唯一性、采样协议、
  路径迁移、group policy及Host/ST契约。日志：`.logs/timer-detail-cpu-final.log`。
- 生产Packet helper的CPU异步模型分别在timer OFF/ON执行；ON版本覆盖11,484次
  packet发布，检查6个RDMA子阶段每包各记录一次、无MTE阶段误记，并维持body完成后
  才发布flags、两次quiet、QP归属、事件复用和故障路径的既有检查。
  单独ON模型日志：`.logs/timer-detail-transport-on.log`；其中timeout打印来自预期故障注入。
- A5设备0独立timer原语：ON/OFF、observe/drain、16个真实AIV、全部新旧阶段、
  重复累计及尾部guard通过。日志：`.logs/timer-detail-device.log`。
  不创建原生LL通信域，不伪造rank。
- RDMA ON + timer ON完整wheel构建、安装及安装后schema/分配检查通过。
  当前设备查询为56 AIV，对齐与全零缓冲验证通过；没有启动完整LL。
  日志：`.logs/timer-detail-wheel.log`、`.logs/timer-detail-installed.log`。
- timer OFF生产dispatch/combine在RDMA OFF和ON两种编译配置均通过。
  首次扩展独立检查遗漏HNS后端宏，已按正式CMake定义修正后通过；生产构建无此遗漏。
  最终日志：`.logs/timer-detail-off-fixed.log`。

所有A5操作固定四个可见性变量为0,1，文件位于授权工程目录。
SDK复用工程内已识别HNS_1825版本，未升级SDK、驱动或新增依赖树。

## 目标机仍需验证

完整dispatch/combine数值正确性、真实HNS_1825跨机收发及阶段耗时由用户运行。
A5未提供该RDMA硬件，CPU模型和编译结果不代表实际跨机验收。
各rank须重建并安装匹配的timer ON wheel，入口检查schema 2；SDK配置未变时无需重建SDK。
使用新run目录重新采集后，原analyze.py命令即可生成包含细分阶段的CSV和trace。
旧数据仍可解析，但不能补出未采集的子阶段。

observe记录代码边界/下发成本，drain增加排空并改变重叠，详见README中的阶段表。
父子阶段不可直接相加，重复阶段的trace宽度为包络，累计值见total_us。
