# 结节分类

## 模型训练流程
1. 模型初始化
2. 加载数据
    1. 获取batch数据
    2. 进入加载器
    3. 传入模型
    4. 计算损失
    5. 记录模型效果
    6. 反向传播
    7. 参数更新
    8. 获取验证数据
    9. 加载验证数据
    10. 预测结果
    11. 计算损失
    12. 记录模型效果
    13. 输出周期记录
## 模型核心 卷积模型
* channel: 1-8-16-32-64  datasize: 32*48*48- 2*3*3
* block分块 conv->relu->conv->relu->maxpooling
* last_block->view->linear->softmax->output