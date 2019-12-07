# 实验过程

## 第一次实验
    实验目的：验证模型拟合能力是否满足需求
    实验参数：batch_size = 64
             gamma = 0
             epoch = 100
    实验结果：在第20epoch之前，模型开始出现过拟合，最后的训练集正确率在75%以上，由于数据集质量较差，训练集和测试集样本的特征分布并不十分一致，所以训练集正确率达到75%已经够了，再增大模型规模对测试集正确率提升并不会有什么帮助。
             最佳测试集正确率：57.50%（出现在过拟合之后，可信度不高）
![first result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_3%20Multi-classification/train%20and%20val%20loss%20vs%20epoches%201.jpg)
## 第二次实验
    实验目的：提高测试集精度，加快训练速度。
    实验参数：batch_size = 64
             gamma = 0.1
             epoch = 30
    实验结果：模型训练的过程基本平稳，最佳模型参数为第16个epoch，在过拟合之前，比较可信。
             最佳测试集正确率：52.50%
![second result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_3%20Multi-classification/train%20and%20val%20loss%20vs%20epoches%202.jpg)