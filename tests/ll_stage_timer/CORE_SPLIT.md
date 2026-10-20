# 编译配置56个AIV的收发分工

适用于 `kernel/ll-configurable-groups` 和 `kernel/ll-g4-stagger`。默认28发+28收；可用两个编译环境变量调整，要求正整数且相加等于56。不要求1:1、不要求各自为偶数、不增加8核倍数等比例限制。

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

LL启动总数固定56：前S个核负责发送/直接回传，后R个核负责接收/partial及聚合；dispatch和combine共同使用该配置。接收局部编号为`core-S`，不能写成`core-R`。平台/设备查询值仅保留为诊断信息，不据此缩减或替换56；目标设备是否能运行该配置需在目标环境验证。

计数前缀矩阵、接收任务分片、Host UB预算和combine参与者使用实际S/R。控制容量仍保留72个magic槽；接收通知行容量56。通知工作由最后`min(8,R)`个接收核覆盖所有逻辑组成员，R<8也不新增准入限制；准备区仍保留原8行容量。G4逻辑分组和MTE传输保持原样，错峰分支继续保留原轮转算法。

## 安装一致性与结果

全rank必须使用同一分支、同一S/R的完整wheel。Host/device协议标识改为`7000+S`，防止混装不同分工的库；测试入口跨rank同时检查send_cores/recv_cores。不同S/R试验可通过compare.py比较，结果附baseline/candidate收发核数；同一作业仍禁止混配。逐核阶段对照若该核从发送变为接收或相反，输出role_changed，不计算该行加速比。

打点仍是累计阶段格式，每核2048B，但schema升为5，header的mode字段高位保存发送核数。解析器按记录分界标注send/receive，旧schema1/2仍可离线读取。新打点需匹配ON wheel并重采，不能把旧1:1数据重新解释成新比例。专家schema1数据结构不变，采样元数据额外保存send_cores，并检查与阶段记录一致。

运行后查看summary.csv/ranks.csv中的send_cores、recv_cores及active_aiv，或在已绑定目标设备的环境调用：

```python
from deep_ep._native import require_native
print(require_native("LL split inspection").low_latency_core_config())
```

性能对比使用两边timer OFF、相同输入/seed/Exp和原10预热、100采集、尾50协议。不同分工可能改变任务量及UB占用，不保证更多接收核更快。

## 本地验证边界

CPU模型遍历1+55到55+1的全部正整数分配，检查核角色、任务覆盖、通知目标覆盖及控制区1MiB边界；另覆盖schema5与旧数据、错误配置、构建指纹、跨比例对照。真实C++布局/平台查询测试已改为固定56并覆盖不对称比例，但本地无C++编译器，执行项跳过。本次未执行CANN编译或设备测试；原始NDR、事件分支及其他任务未提交kernel均不修改。

验证记录（2026-10-20）：泛化分支合计220 passed、27项缺C++编译器跳过、9项Bash/POSIX检查未选；错峰分支236 passed、28项跳过、9项未选。包含阶段/配置及构建/打包CPU检查；未执行CANN编译或NPU测试。旧58核协议6000+S被拒绝，全rank需重装56核完整wheel，SDK无需因本次核数修正重建。
