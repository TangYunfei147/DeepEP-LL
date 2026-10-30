# LL reference 打点分支约束

## 当前要求（2026-10-30）

- 用户要求完全参考 `kernel/ll-ndr-metadata-cores` 的打点及相关解析，替换当前打点且不保留旧代码。参考固定 `c3d8640`，当前修改仅在 `kernel/ll-g4-systemcycle-profile` / `DeepEP-LL-systemcycle-profile`。
- 当前只保留 `DEEPEP_LL_REFERENCE_PROFILE` 的 LL dispatch 打点。旧 stage/expert、SystemCycle40点、sparse、MTE3、bandwidth 记录器、宏、兼容解析和原采集/报表入口均删除；combine完全无打点。旧数据如需读取使用Git历史对应版本，当前不加兼容层。
- 沿用参考的29项主阶段、一次clock+GM主点写入、ready批次累计、128条稳定任务包日志、单轮Excel与多轮统计分开。新schema6/80×8word/每核5120B，w22/w23真实S/R、w62=0，拒绝NDR schema5与旧数据混读。
- 当前业务仍是9027275恢复基线的MTE/G4/56核，S完成源发送后共享接收，R计数/本组发送后共享接收；不得移植M核、NDR transport/wire/QP/SDK升级。无实际RDMA的put/quiet/第二跨机包细分留为不适用，不能将外G4首跳标成RDMA。
- 主时点必须对应真实业务边界，提交不能称完成。仅ready批次读钟，空轮询/逐包日志不读钟；日志只用实际剩余UB，不减lane，截断显式。日志DMA和所需诊断事件仅发生在业务w21后；w57包含日志导出，末尾身份发布不计入。
- 保持原业务Scalar/DMA/同步、7个独立TU、O2、原UB预算/lane/event及wire19000+S。dispatch仅保留referenceProfile指针、144B Host参数；combine无诊断参数、88B；内部_ref6符号拒旧对象混装。OFF不读钟、不挂日志。

## 采样与导出

- 采样与参考统一 `initial-reduce-exp100-batch-v3`：每批只初始all_reduce一次，预热同步后、active step前100次Exp，连续稳定/采集，循环无Exp/all_reduce，不在Exp后额外同步。恢复请求seed及matching handle，记录实际profile_seed和完整route哈希。
- 唯一目标入口 `tests/ll_reference_profile/one_click.sh`，默认reference、1轮、EP16/B=M256/H3584/K8/E384/G4/S30R26、预量化FP8、HT OFF。可显式多机及shape；正常perf也使用新采样协议，不能与旧逐轮Exp+reduce当作同条件A/B。
- 仅解析当前格式。按每rank每轮最长单核w21-w0统计，不能称OFF完整kernel；单轮逐核表默认last，全部轮次均值另表，不拼造平均单轮。原始uint64保真、未测留空、有效零保留、父子不相加、不跨rank时钟相减。
- 完整XLSX/CSV/JSON留目标机，只回传一个≤60000 UTF-8字节 `reference-analysis/return_report.md`；多case省略项必须显式说明，总文件不超过10。

## 构建与维护

- 修改kernel前读 `tests/ll_probe/KERNEL_MAP.md` 和 `docs/BUILD_DISPATCH_SPLIT.md`。Host只做模式选择，不把7种实现合进单global/TU，不降低生产优化级别。
- 保持当前SDK锁8e3fc610及PI0/1合法身份兼容、完整SDK哈希和同作业wheel/source/ABI保护。用户指定另两NDR分支的SDK升级不适用于此工作区。
- 只准备SHMEM及必要子模块：`git submodule update --init --recursive --depth 1 -- third-party/shmem`，不拉CATLASS。不擅自修改SDK或manifest，不自动重建不兼容SDK。
- 工程、日志、缓存、临时文件按脚本位置定位工程根；显式外部CANN配置可继承，不能写死开发机路径。结果/日志不覆盖，不停止无关进程。
- 先查Git状态，其他工作区不改。仅本地CPU检查，不SSH、不连接27/其他编译主机、不执行CANN/NPU；完成后提交推送，给用户目标机命令。CPU/mock不是CANN或性能验收。
- 用户交互命令使用函数return和外层if，失败保留shell/日志，不用exit结束当前交互shell。
- 相关本地检查：`tests/ll_reference_profile`、`tests/ll_kernel`、`tests/elastic/test_ll_st_contract.py`、构建身份回归。旧53文件业务ledger保留为源码语义核对证据，不更新hash伪造等价。
