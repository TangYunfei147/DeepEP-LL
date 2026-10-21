# S核去重路径私有q表

基线3d400d7，只优化S核的SendWithDedup和FIRST_HOP FillTail。
非去重发送仍按展平路由分片使用原CalTokenSendExpertCnt；R核仍读取计数阶段生成的expertQ。
发送顺序、q/包尾编码、权重、mask、包数、接收/转发算法不变。

每个非空S核先清零E项uint32直方图，从完整routeIds的0扫描到start*K播种，
再按原始顺序扫描本核count*K条：先读histogram[expert]写qTable，再递增histogram。
保留原index<K时q=0的特判，首token重复专家也逐项匹配原实现（不扩展合法路由契约）。
FIRST_HOP首项、DIRECT任务q和FillTail成员q改查qTable[index-routeQBegin]。
DIRECT继续缓存q；后续轮转发送及最后待处理包不依赖当前直方图状态。

## UB与依赖

- qWorkBuf：持久q表，原分配容量为对齐后的B*K*4，足够容纳本核count*K；
  建表后只读，保持到EndInputPrefetch处理完最后一个包。没有把它复用为量化工作区。
- qSubBuf：E项直方图，容量max(旧scratchBytes,align(E*4,32))；建表结束后不再访问。
- qDstBuf：仍为量化工作区；QuantMxfp8不读写qWork/qSub。
- 非去重路径仍保留原qDst/qSub/qWork工作区用途。布局预算按两路径上界保守规划，
  没有在本次删除旧工作区或宣称一定降低UB。
- 清零用Duplicate并等待一次本核V_S，之后播种/填表均scalar。路由读取沿用原搬入依赖。
  没有跨核同步、原子、额外GM数据流量；新增一次局部完成依赖，去掉去重路径反复q归约的V_S。

Host PacketUbFits和Device LlMakeOverlapLayout同步扩大直方图预算。
E>B*K时可能增加占用、改变lane/fallback；常用B256/K8/E384时原qSub容量足够。
协议parallel18000+S/r0 19000+S拒绝旧Host/Device预算混装；prefix编译选择保持不变。

## 打点与解读

schema15新增ID91 BUILD_Q_TABLE，全量记录一次清零、V_S、播种及填表，位于
PREPARE_INPUT结束后、SEND_TOKENS开始前；不适用于空S核、非去重路径或R核。
该V_S也可能等待先前V初始化的余量，因此建表阶段不是纯scalar播种时间。
每核载体仍4608B，192条trace+92项metrics恰好用满容量，新增阶段不能继续直接追加。
解析保留schema14及更旧格式；旧数据不补建表耗时。dispatch_packets.md/CSV新增q建表列，
慢核线索包含该阶段；去重S核不再出现原CAL_TOKEN归约不是漏打点。

固定S、N=B*K时，播种总量近似(S-1)*N/2，本分片扫描合计N；还要计入每核E项清零。
小B、稀疏远端路由可能负收益；后编号核播种更长，首包可能推迟。
比较BUILD_Q_TABLE、SEND_TOKENS、首次接收和FINAL_DRAIN，最后看OFF整体耗时，
不能只看发送循环缩短或拿schema14/15诊断值直接当加速比。

## 目标机

全rank更新同提交并重装完整wheel，不强制重建已校验SDK：

```bash
git pull --ff-only origin experiment/ll-g4-overlap-ab
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_LL_PREFIX_MODE=parallel  # r0也需单独构建/对照
export DEEPEP_LL_STAGE_TIMER=ON
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

沿用原ST参数、seed及S/R，ON capture需重采；原analyze.py命令不变。
以3d400d7为基线，分别重装OFF做正式性能对照。覆盖M>32的去重路径与M<=32的非去重回归、
空/不均匀S分片、K1/K8、E16/384/1024、BF16/FP8及热点路由，记录lane与fallback。
CPU模型对实际私有q helper逐项比对原前缀扫描，并检查反向查表及表尾哨兵。

## 验证记录

按用户要求先验证未提交补丁，再创建提交。基线3d400d7，验证补丁SHA256：
`8ba7f9e783eb31f10181cfb0386c4f258999baffa7d680202579b60f13b4a480`。
最终源码与该补丁一致，仅追加文档记录。

- 本地352通过、33环境跳过；A3实际CPU/C++385通过，补齐跳过项目。
- A3_docker/CANN9.3.0/py311_tyf，保持Ascend950、O2、S28/R28、jobs4、7个独立模式TU。
- parallel冷dispatch编译ON77.650秒/OFF49.011秒；r0 ON74.242秒/OFF47.325秒。
- 两模式完整ON wheel编译链接通过；未安装wheel、未运行NPU、未重建SDK。
- 编译结束只反向撤销本任务补丁，已恢复远端原分支。

远端工程`.logs/private-route-q-001/`保存cpu.log、patch.sha256及各模式
`{parallel,r0}/{on,off}/timing.json`、wheel.log、revision.txt；wheel在
`.artifacts/private-route-q-001/{parallel,r0}/`。timing.json的revision是补丁基线，
必须连同patch.sha256识别实际编译内容，不能将这些数据当成未修改3d400d7的编译结果。
这些时间是编译耗时；设备正确性、首包时延和OFF性能收益仍待目标机验证。
