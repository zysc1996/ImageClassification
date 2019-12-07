# 实验过程

## 第一次实验
    实验目的：验证模型拟合能力是否满足需求
    实验参数：batch_size = 64
             gamma = 0
             epoch = 100
    实验结果：在第20epoch之前，模型开始出现过拟合，最后的训练集正确率在95%以上，证明模型拟合能力满足需求。
             最佳测试集正确率：56.25%
![first result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_2%20Species_classification/train%20and%20val%20loss%20vs%20epoches%201.jpg)
## 第二次实验
    实验目的：提高测试集精度，加快训练速度。
    实验参数：batch_size = 16
             gamma = 0.1
             epoch = 30
    实验结果：模型训练的过程基本平稳，最佳模型参数为第17个epoch，比较可信。
             最佳测试集正确率：56.25%
![second result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_2%20Species_classification/train%20and%20val%20loss%20vs%20epoches%202.jpg)