# 独立元数据核、固定转发与均衡搬出

分支 `kernel/ll-ndr-metadata-cores` 从 `3e4ab86` 分出，原 `kernel/ll-ndr-ab-per-core-qp` 保持原实现。

编译变量 `DEEPEP_LL_METADATA_CORES=M`，默认4，要求2～70内的偶数，且实际AIV核数C至少为M+2。
核0..M-1执行原count交换、接收、cumsum和任务生成；随后 `(C-M)/2` 个S核直接发送；最后同样数量的R核负责转发和搬出。56核默认4M+26S+26R。
combine仍使用原C/2分工，SDK、物理核QP编号、动态AIV查询和公开张量接口保持。
协议身份为 `40000+2*M+histogram`，默认40008，直方图40009；全rank必须更新完整wheel，不能混装旧协议或不同M。

## 两套所有权

固定候选槽编号为 `(((source+destination)%EP)*localExperts+expert)*B+q`，总计E×B；这是实际B，capacity不参与槽编号。
转发owner为 `slot%R`，在prefix就绪前可扫描并转发FIRST_HOP；prefix就绪后剔除无效q，但仍处理晚到包。
每核UB位图记录已接管槽，防止重复转发。DIRECT、TERMINAL同样由固定owner检查并接管。

元数据核将实际count按原expert/source顺序生成“输出行→物理槽”表，复用原relay预留区；每核生产互不重叠片段。
元数据0等待所有片段提交完成，再为每个R核发布32B `[begin,end)` 描述符和独立32B就绪槽。
区间边界为 `floor(total*core/R)`，每核搬出任务数最多差1。R读取并清理自己的通知，空任务核也照常完成转发扫描。
这些控制写入均为本机MTE，不增加跨机put；原count消息数量和payload去重、q、权重保持。

## 一个新状态：1.25

原1.0/1.5发布标记保留。只新增首flag首float的1.25，含义是固定转发owner已完成输入读取，并已提交必要转发。
就绪判定仍归约全部32B flag，接受总和 `8*blocks`、`8*blocks+0.5`、`8*blocks+0.25`，不会只读首flag。

搬出核可在接管前读payload并写输出GM，但只能在观察到完整1.25状态后清全部flag。
已经搬出的任务在本核pending索引中标记，后续只检查清理条件，不重复写输出。
循环交替转发、查询prefix、搬出/清理，不在单个未接管包上阻塞，从而避免两个搬出核互相妨碍转发。
接管标记使用独立的每lane元数据暂存，不覆盖仍被转发DMA读取的wire。原输入header不改；转发副本改为TERMINAL和1.0。
lane完成事件仍保护UB复用；1.25不代表远端已消费，也不替代发送侧CQ退休。

## 复用与combine临时stage

prefix前扫描所有候选槽，要求每个512B周期的末32B不会含旧raw payload。
经用户确认，combine本机临时stage改为480B数据+32B零间隔，读写和Host容量同步；末尾不满480B也清对应32B。
stage stride变为 `ceil(2H/480)*512`；接近旧容量边界的shape可能不再准入。没有增加跨机消息、改变归约/Cast顺序或新增UB。
零值来源复用本阶段尚未使用的result UB，全部stage写完成后才发布原prepare通知。

dispatch有效包及combine回包的flag仍在原消费后清理；task表只存有界整数槽号，不会形成浮点ready值。
已消费的普通count记录整32B清零，元数据前缀完成矩阵统一使用正常float 1.0发布/归约，避免专家数变化时旧整数count/prefix落入新控制标志地址。没有增加控制状态、DMA字节、跨机put或同步。
同通信器、固定M/C、单stream同序和双bank仍是前提：本rank结束n+1需要收齐各peer的n+1 count，证明各peer已结束n，才可能开始同bank的n+2发送。
combine交替继续沿用 `combine_reuse` 协调。没有将SO扩展为跨QP/独立WQE全局顺序，也没有删除源暂存/CQ生命周期保护。

## 打点

reference schema5仍为每核640个uint64。w62记录M，w63为send=0、receive=1、metadata=2。
只有元数据核记录count阶段；R先准备缓冲，再交替扫描与等待prefix，不能沿用旧S/R二等分解释。
w33/w34为等待/处理累计：一个等待起点跨交替扫描与prefix查询，空扫描不重复读clock；有可接管/搬出/清理的批次才读末点。w41统计此类就绪批次，w108单独统计发现新搬出任务的批次。
R包明细记录搬出owner的发现/处理顺序，包中FIRST_HOP成员数不是“该搬出核执行转发的次数”。
日志只用既有lane分配后的剩余UB，最多128行，w21后导出。OFF无GetSystemCycle；ON结果不能作为OFF性能。
阶段定义见 [STAGE_TIMELINE.md](STAGE_TIMELINE.md)。

## 目标机测试

所有主机使用同分支、同M和同run_name；以下以2机EP32为例，NODE_RANK分别为0、1，MASTER_ADDR替换为首机地址。
新目录获取源码时不递归clone：

```bash
git clone --single-branch --branch kernel/ll-ndr-metadata-cores https://github.com/TangYunfei147/DeepEP-LL.git DeepEP-LL-ndr-metadata-cores
cd DeepEP-LL-ndr-metadata-cores
export DEEPEP_LL_METADATA_CORES=4
export NODE_RANK=0
export MASTER_ADDR=192.0.2.1
if bash tests/ll_rdma/one_click.sh "$NODE_RANK" "$MASTER_ADDR" meta4-perf-001 \
    --num-nodes 2 --mode perf --tokens 1 8 32 --dtype bf16 fp8; then
    echo "PASS: correctness and OFF dispatch/combine"
else
    echo "FAILED: keep this terminal and inspect .logs/ndr-one-click"
fi
```

入口默认保留10预热/100采集/尾50、initial-reduce-exp100-batch-v3；正确性和重复combine仍检查。
依赖命令严格限定 `git submodule update --init --recursive --depth 1 -- third-party/shmem`，不拉CATLASS。
SDK沿用本项目正常构建与身份检查，不修改其实现。

独立reference重编测试（所有主机执行，使用新名字）：

```bash
if bash tests/ll_rdma/one_click.sh "$NODE_RANK" "$MASTER_ADDR" meta4-reference-001 \
    --num-nodes 2 --mode reference --tokens 8 --dtype fp8 --reference-iters 1; then
    echo "PASS: inspect results/ll-ndr/meta4-reference-001-reference"
else
    echo "FAILED: inspect .logs/ndr-one-click"
fi
```

完整Excel、日志与raw留目标机；只回传run目录导出的单个`table-analysis/return_report.md`（入口控制在60000 UTF-8字节以内）。
本地CPU模型覆盖真实元数据/接收器、零计数、独立owner、延迟prefix/包、转发去重、接管前搬出但不清flag、连续case及combine零间隔；不代表CANN编译、NPU数值或性能已通过。

本地验证记录：核心CPU用例分批共586项通过、2项因缺少Torch跳过；最后139项相关回归（含13个subtest）复测通过，与前述用例重叠，不相加。覆盖M=2/4/18的实际count流程、M=2/4/6/18/70的编译分工、16/72核ON/OFF接收、BF16/预量化FP8/非去重、E/B变化和combine stage交替。另49项构建流程检查通过；8项已有Linux编译器探测/符号链接测试在Windows环境失败，不属于CANN编译结果。
