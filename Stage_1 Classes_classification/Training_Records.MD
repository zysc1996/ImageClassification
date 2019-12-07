# 实验过程

## 第一次实验
    实验目的：验证模型的拟合能力是否满足需求。
    实验参数：batch_size = 64  
             gamma = 0 
             epoch = 100
             其余参数为默认值
    实验结果：在第20个epoch左右，模型开始出现过拟合，最后的训练集正确率在95%以上，证明模型拟合能力满足需求。  
             最佳正确率为76.67%
![first result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_1%20Classes_classification/train%20and%20val%20loss%20vs%20epoches%20experiment%201.jpg)
## 第二次实验
    实验目的：提高测试集精度，加快训练速度。
    实验参数：batch_size = 16
             gamma = 0.1
             epoch = 30
             其余为默认值
    实验结果：模型在训练刚开始的时候就陷入了局部最优，训练集和测试集上的loss值始终在50%上下波动，需要对参数进行调整。
             最佳正确率为70.00%（偶然情况，不可信）
![second result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_1%20Classes_classification/train%20and%20val%20loss%20vs%20epoches%20experiment%202.jpg)
## 第三次实验
    实验目的：提高测试集精度，加快训练速度，防止模型陷入局部最优。
    实验参数：batch_size = 16
             gamma = 0.05
             epoch = 30
    实验结果：模型训练的过程基本平稳，最佳模型参数为第19个epoch，比较可信。
             最佳正确率为81.67%
![third result](https://github.com/zysc1996/ImageClassification/blob/master/Stage_1%20Classes_classification/train%20and%20val%20loss%20vs%20epoches%20experiment%203.jpg)


