> 2026-10-20本组初发更新：原R核先发送本源rank的组内包，再运行原接收流程；核心标识10000+S。当前调度/UB/验收以[GROUP_LOCAL_SEND.md](GROUP_LOCAL_SEND.md)为准。

# 编译配置56个AIV的收发分工

本分支为 `kernel/ll-g4-shared-recv`。默认前段28发送+28计数/接收；dispatch发送核完成发送后也加入接收，最终56核消费包。可用两个编译环境变量调整，要求正整数且相加等于56。不要求1:1、不要求各自为偶数、不增加8核倍数等比例限制。

```bash
# 仓库根目录，已激活该分支匹配的Python/CANN环境
source scripts/ll_timer_env.sh
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_SEND_CORES=20
export DEEPEP_LL_RECV_CORES=36
export DEEPEP_LL_STAGE_TIMER=OFF   # 独立诊断时改ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

只改一项导致总数不是56时会报错，不自动计算另一项。未设置时两项分别默认28。整数/正数检查用于避免除零和负索引；既有SDK、EP、shape及UB容量检查继续保留。配置不是运行时开关：运行前设置环境变量不能改变已安装wheel。

直接使用CMake时，传 `-DDEEPEP_LL_SEND_CORES=20 -DDEEPEP_LL_RECV_CORES=36`。根Host与子Device构建会获得相同定义；直接Device入口也校验总数。Python构建指纹包含两项，切换比例不会复用不匹配的构建目录。不需因为调整比例重建已校验的SHMEM。

## 执行语义

LL启动总数固定56：dispatch前S核发送，后R核生产计数前缀；各核完成前段工作后等待自己的前缀ready行，再按物理core编号在全部56核间分配接收输出行。`recv_cores=R`表示初始接收/前缀核数，`dispatch_recv_cores=56`表示完整dispatch接收参与者。没有全核发送完成屏障，也不动态抢其他核任务。

combine继续使用前S核direct、后R核partial/aggregate，局部编号仍为core-S。G4、MTE、去重首跳与错峰发送顺序保持。前缀求和矩阵仍R×R，前缀副本56份，ready矩阵56×R；按56核计算dispatch接收UB预算。详见[SHARED_RECEIVE.md](SHARED_RECEIVE.md)。

## 安装一致性与结果

全rank必须使用同一分支、同一S/R的完整wheel。Host/device协议标识改为`10000+S`，防止混装不同分工的库；测试入口跨rank同时检查send_cores/recv_cores。不同S/R试验可通过compare.py比较，结果附baseline/candidate收发核数；同一作业仍禁止混配。逐核阶段对照若该核从发送变为接收或相反，输出role_changed，不计算该行加速比。

当前打点为参考分段schema7，每核4608B；每阶段前两次保留实际区间，全量累计所有调用。发送后接收的核标为send_and_recv，前缀核标为cumsum；角色指初始职责，不限制后续接收。旧schema1/2/5/6仅离线解析。新模式拒绝drain和逐专家GM打点，详见[SEGMENT_TIMER.md](SEGMENT_TIMER.md)。

运行后查看summary.csv/ranks.csv中的send_cores、recv_cores及active_aiv，或在已绑定目标设备的环境调用：

```python
from deep_ep._native import require_native
print(require_native("LL split inspection").low_latency_core_config())
```

性能对比使用两边timer OFF、相同输入/seed/Exp和原10预热、100采集、尾50协议。不同分工可能改变任务量及UB占用，不保证更多接收核更快。

## 本地验证边界

CPU模型遍历1+55到55+1的全部正整数分配，检查核角色、任务覆盖、通知目标覆盖及控制区1MiB边界；另覆盖schema5/6与旧数据、错误配置、构建指纹、跨比例对照。真实C++布局/平台查询测试已改为固定56并覆盖不对称比例，但本地无C++编译器，执行项跳过。本次未执行CANN编译或设备测试；原始NDR、事件分支及其他任务未提交kernel均不修改。

本次实现与验证边界见[SHARED_RECEIVE.md](SHARED_RECEIVE.md)；不得沿用基础分支的设备验证结论。
