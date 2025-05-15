# PyTorch 张量基本操作
- [一个简单的张量](ch04/01.ipynb)
- [从Numpy创建张量,存储与获取](ch04/02.ipynb)
- [张量拼接、切分、索引、变换](ch04/04.ipynb)
- [张量传递到GPU](ch04/07.ipynb)
- [张量偏移量和步长](ch04/09.ipynb)
- [二维图像加载](ch05/01.ipynb)
- [DICOM图像加载](ch05/03.ipynb)
- [CSV加载](ch05/04.ipynb)
- [词嵌入加载、One-Hot编码](ch05/07.ipynb)

# 神经网络
- [模型常规训练过程](ch06/01.md)
- [温度计示转换与损失、广播](ch06/02.ipynb)
- [梯度、学习率、归一化、超参数](ch06/05.ipynb)
- [PyTorch自动计算梯度、与优化器](ch06/09.ipynb)
- [常用激活函数](ch06/11.ipynb)
- [nn模块搭建神经网络](ch06/12.ipynb)
- [神经网络区分飞机与鸟](ch07/01.ipynb)
- [PyTorch搭建卷积网络、提取特征、分类、正则化、深度、宽度等](ch07/08.ipynb)

# 肺部结节检测
- [肺癌检测介绍](ch08-project/01.ipynb)
- [CT数据、标注数据加载、分割训练集、验证集](ch08-project/06.ipynb)
- [CT数据可视化](8-11.ipynb)

# 模型训练与优化
- [结节分类介绍](ch09/01.ipynb)
- [定义模型训练框架、初始化、数据加载器、模型核心、定义损失、训练环节、绘制曲线](ch09/02.ipynb)
- [数据优化、训练](9-13.ipynb)
- [数据增强、旋转、翻转、放大缩小、噪声添加、平移](9-15.ipynb)
- [U-Net分割模型介绍](9-16.ipynb)
- [分割模型数据预处理、构建DataSet类、Adam优化器、Dice损失、模型存储](9-19.ipynb)  
- [分割模型训练、TensorBoard查看](9-25.ipynb) 

# 端到端模型链接与部署
- [链接分类和分割模型](ch10/01.ipynb)
- [AUC-ROC评分](10-2.ipynb)
- [finetune微调模型](10-3.ipynb) 
- [完整端到端实现](10-4.ipynb) 
- [使用flask部署模型](flask_http.py) 
    - 运行 
    ```bash
        # 启动分类模型http服务

        python -m flask_http data/model/cls_2025-05-14_18.32.02_dlwpt.best.state

        # 客服端测试
        python client_test.py
        ```
