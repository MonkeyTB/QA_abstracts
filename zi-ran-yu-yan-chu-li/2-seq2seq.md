---
description: 通用seq2seq模型分析理解
---

# 2、seq2seq

## seq2seq 综述

seq2seq是一种循环神经网络的变种，包括编码器和解码器两部分，可用于机器翻译、对话系统、自动摘要

## 常见seq2seq结构

seq2seq是一种重要的RNN模型，也称为Encoder-Decoder模型，可以理解为 $$N*M$$ 的模型，模型包含两部分，Encoder部分编码序列的信息，将任意长度的信息编码到一个向量 $$c$$ 里，而Decoder部分通过向量 $$c$$ 将信息解码，并输出序列，常见的结构如下几种：

![&#x7B2C;&#x4E00;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%286%29.png)

![&#x7B2C;&#x4E8C;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%284%29.png)

![&#x7B2C;&#x4E09;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%285%29.png)

### Encoder

· 这三种模型结构的主要区别在于Decoder上，Encoder是一样的，Encoder部分的RNN通过接受输入 $$X$$ ，通过RNN编码所有信息到 $$c$$ （所有神经元最后的输出，不要中间神经元的输出）。

### Decoder

· 第一种Decoder，结构简单，将Encoder编码信息 $$c$$ 作为Decoder解码隐层的初始输入，后续神经元接受上一神经元的隐层输出状态 $$h`$$ ；

· 第二种Decoder，将Encoder编码信息 $$c$$ 作为Decoder解码隐层每个神经元的输入，而不在作为隐层的初始输入；

·  第三种Decoder，再第二种的基础上，每个神经元的输入有了两部分，一部分是Encoder编码信息 $$c$$，另一部分是Decoder上个神经元的输出，两部分通过不同的权重矩阵来作为当前神经元的输入。

