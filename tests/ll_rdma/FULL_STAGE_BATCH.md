# 按GM stage容量批处理（dispatch默认32/直方图33）

第一项改动的对照为f5e1891。每核128KiB GM stage不再固定最多8包，容量为
`floor(131072 / packetBytes)`，可用完整包槽全部占用后才批量quiet；不足一包的尾部不使用。
算子末尾未满的尾批仍收尾。UB lane数、GM分配量、count独立源区、wire和SDK不变。

原firstPeers_按包序号使用32位掩码，直接扩大容量会在超过32包时移位溢出。
现在使用按peer编号索引的256位位图（8个uint32，与旧peer数组同为32B），每个peer只quiet一次。
512B包可容纳256包，4096B包32包，7680B包17包，128KiB包1包。
SDK自己的SQ/CQ近满检查仍可能poll，不承诺所有put在队列压力下均不阻塞。

只改变dispatch批处理容量；dispatch其他依赖候选仅审计，见DISPATCH_DEPENDENCY_AUDIT.md。
combine后续复用此helper的独立提交及身份见COMBINE_BATCH_RETURN.md。

包数不变、每包一次put，改变的是源暂存利用率、在途度和每批每peer的quiet。
若单核总包数本来就少于8，扩容本身未必减少quiet；必须比较完整OFF task，不能预先宣称收益。
全rank完整wheel更新，NDR原采样协议与测试入口保持；未做CANN/NPU验证。
