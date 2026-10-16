# 事件分支的比较边界

本分支使用schema4逐次事件，原NDR/泛化分支的阶段格式不能作为本分支输入。没有共享文件字节一致性承诺或旧格式适配。

正式性能可用compare.py比较双方timer OFF且无stage/expert capture的报告；允许G4/G16作为显式比较变量，保留其他shape、核数、路由、seed、采样协议的一致性检查。timer ON时schema或模式不同拒绝比较。

PIPE_ALL与事件写入开销会改变重叠，不能把mega模式阶段时间与原observe累计时间直接相减解释成优化收益。使用相同版本、输入及模式，单变量对照并记录完整Git提交。
