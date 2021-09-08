---
description: 文本分类调研综述
---

# 1、文本分类

### 三步骤

#### 特征工程

1. 常用的特征主要是词袋特征，复杂的特征词性标签、名词短语和树核

#### 特征选择

1. 特征选择旨在删除噪音特征，常用的就是移除法
2. 信息增益、L1正则

#### 分类器

1. 机器学习方面主要有逻辑回归、朴素贝叶斯、支持向量机 ----- 数据稀疏性问题
2. 深度学习和表征学习为解决数据稀疏问题

### RCNN

1.  在我们的模型中，当学习单词表示时，我们应用递归结构来尽可能多地捕获上下文信息，与传统的基于窗口的神经网络相比，这可以引入相当少的噪声。
2.  还使用了一个最大池层，它自动判断哪些单词在文本分类中起关键作用，以捕获文本中的关键成分。
3. 双向递归神经网络来捕捉上下文。

![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=MmY1ZWI5Njc2MTNkNGY2MmE1ZTU0OTAyYTJlMTZhN2RfNGhFZlNiOUVhUHRZUjYzdm1SajNqNzJDSENwazZZMmFfVG9rZW46Ym94Y25tTUhTaEVPeVZNWEs2N24zQWtPc0hoXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=MjQyMjcyMGQ5ZmFhODQzZmRmNzliMjUyMWRlMmM1ZjJfenVqYUJPdG5FZ2l1dGVQb1BxWXJsVmhRbTRQc0d0RHFfVG9rZW46Ym94Y24wTkZqcW5pd3lWSEJPczZTRThib1BnXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)

* $$c\_l\(w\_i\)$$用公式一计算，其中$$c\_l\(w\_{i-1}\)$$是当前 word 的左侧的 context，$$e\(w\_{i-1}\)$$是单词的嵌入，稠密向量

1. TextCNN比较类似，都是把文本表示为一个嵌入矩阵，再进行卷积操作。不同的是TextCNN中的文本嵌入矩阵每一行只是文本中一个词的向量表示，而在RCNN中，文本嵌入矩阵的每一行是当前词的词向量以及上下文嵌入表示的拼接

### HAN

[论文解读](https://zhuanlan.zhihu.com/p/54165155)![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=NmU0Y2Q5ZDA0NTFjMWFmOTA3ZDgwYWJmMmYyNTM3YzhfU1NuTHBMRHhiUGVMdXZSNkF3ZTB6YlU4SEVYalJVUDdfVG9rZW46Ym94Y25uWno2TlJjR0wyUjM2Q1RieTA0RlFjXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)

### GCN

[论文解读](https://lsvih.com/2019/06/27/Graph%20Convolutional%20Networks%20for%20Text%20Classification/)![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=YmEzYjVlODAyMmMzYzFkYTY3ZjdjYzFmYmZiM2JkYWJfaVRyZ3luWEMyY3JQZVNEcTFVV1VhODJsR3JFc1JVVTFfVG9rZW46Ym94Y241MnJjRFBZeXRnUVNoZ01MbVpEbVZkXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)

1. 提出了一种新的文本分类图形神经网络方法。据我们所知，这是第一次将整个语料库建模为异构图，并利用图神经网络联合学习单词和文档嵌入的研究。
2. 在几个基准数据集上的结果表明，我们的方法优于现有的文本分类方法，不使用预先训练的单词嵌入或外部知识。我们的方法还自动学习预测单词和文档嵌入。

```text
输入特征矩阵X=I：one-hot
图边计算方式
    word-document：tf-idf
    word-word：PMI-为了利用全局词共现信息，我们对语料库中的所有文档使用固定大小的滑动窗口来收集共现统计。
A_ij计算方式如下：
    PMI(i, j)； i, j are words, PMI(i, j) > 0
    TF-IDF_ij；  i is document, j is word
    1； i = j
    0； otherwise
```

![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=NDcyM2Y0NmY1NmQxNGIwMzc4NDI1NzI5MmIwMDJmMjVfenpISklyVHA3eTQ3ZUVOY1l4TzF6WjJ5ZExza1czWDBfVG9rZW46Ym94Y25BaUJlM0xmQ1BZbDc0MlF6WnJrQlBiXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=MTRjM2M1ODZiZWYyYTliOGQ3Njg4MWQ0NDYyM2M4ZDlfNUl3MXZzcnI4Z0M0Zzc4SEM0dXplVHhNcEdYS2JwdGdfVG9rZW46Ym94Y25qRkxta3dGMUtUNHpwbkJoMmhZUUxiXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)不直接用$$D^{-1}A$$而选用$$D^{-1/2}AD^{-1/2}$$，是因为$$D^{-1}A$$的结果不是对称矩阵，这个大家动手算一下就知道了。虽然两者结果不相同，但是$$D^{-1/2}AD^{-1/2}$$已经做到了近似的归一化，而且保持了矩阵的对称性，我想这就是选用对称归一化的拉普拉斯矩阵的原因[链接](https://blog.csdn.net/qq_35516657/article/details/108225441)，属于拉普拉斯对称归一化[链接](https://zhuanlan.zhihu.com/p/362416124)。

### 损失函数

#### 交叉熵损失

KL距离常用来度量两个分布之间的距离，其具有如下形式![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=NjEzMjAyZjVjNmM1OWUyZDczOWY2OThjOTY2YTc1NDJfbG02Q0RHaklvbDRTRkdYQlJuSTBwTlhLUTN1aUh4SEtfVG9rZW46Ym94Y25TQ2xXU3E5Rm8xNXJMWEd0WTZ3Y0NSXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA)其中p是真实分布，q是拟合分布，H\(p\)是p的熵，为常数。因此![](https://g22h5luj8j.feishu.cn/space/api/box/stream/download/asynccode/?code=YjE4MjVjZGM0MTJkMjczYzNlMjU0MTRjMmMzOGY1YTJfbGZiTHRDMkNBRzNYUWlEQTU1V3lXOWpHZmV6VTBIZW1fVG9rZW46Ym94Y24xUzY3UFJHWjJvT3NCSVo1bmhOVlplXzE2MzEwODc0Mzc6MTYzMTA5MTAzN19WNA) 度量了p和q之间的距离，叫做交叉熵损失。

### 优化方法

#### 数据优化

1. 训练集合和测试集合部分特征抽取方式不一致
2. 最后的结果过于依赖某一特征

* 优化方法
  * 在全连接层 增加dropout层，设置神经元随机失活的比例为0.3，即keep\_rate= 0.7
* 在数据预处理的时候，随机去掉10%的A特征

1. 泛化能力较差

* 优化方法
  * 增加槽位抽取：针对部分query, 增加槽位抽取的处理，比如将英文统一用&lt;ENG&gt;表示，模型见到的都是一样的，不存在缺乏泛化能力的问题. 瓶颈在于槽位抽取的准确率。

1. 缺乏先验知识

#### 模型优化

1. embedding\_dim长度
2. 在全连接层增加dropout层

