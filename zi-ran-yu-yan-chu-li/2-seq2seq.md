---
description: 通用seq2seq模型分析理解
---

# 2、seq2seq

## seq2seq 综述

seq2seq是一种循环神经网络的变种，包括编码器和解码器两部分，可用于机器翻译、对话系统、自动摘要

## 常见seq2seq结构

seq2seq是一种重要的RNN模型，也称为Encoder-Decoder模型，可以理解为 $$N*M$$ 的模型，模型包含两部分，Encoder部分编码序列的信息，将任意长度的信息编码到一个向量 $$c$$ 里，而Decoder部分通过向量 $$c$$ 将信息解码，并输出序列，常见的结构如下几种：

![&#x7B2C;&#x4E00;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%2811%29.png)

![&#x7B2C;&#x4E8C;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%287%29.png)

![&#x7B2C;&#x4E09;&#x79CD; seq2seq&#x7ED3;&#x6784;](../.gitbook/assets/image%20%286%29.png)

### Encoder

· 这三种模型结构的主要区别在于Decoder上，Encoder是一样的，Encoder部分的RNN通过接受输入 $$X$$ ，通过RNN编码所有信息到 $$c$$ （所有神经元最后的输出，不要中间神经元的输出）。

### Decoder

· 第一种Decoder，结构简单，将Encoder编码信息 $$c$$ 作为Decoder解码隐层的初始输入，后续神经元接受上一神经元的隐层输出状态 $$h`$$ ；

· 第二种Decoder，将Encoder编码信息 $$c$$ 作为Decoder解码隐层每个神经元的输入，而不在作为隐层的初始输入；

·  第三种Decoder，再第二种的基础上，每个神经元的输入有了两部分，一部分是Encoder编码信息 $$c$$，另一部分是Decoder上个神经元的输出，两部分通过不同的权重矩阵来作为当前神经元的输入。

## 使用trick

### Attention

seq2seq模型中，Encoder总是将源句子的信息编码到一个固定长度的上下文 $$c$$ 中，然后Decoder的过程中 $$c$$ 总是不变的，这样就存在几个问题：

1. 当源句子较长时，很难用一个定长的向量 $$c$$ 表示完所有的信息；
2. RNN存在梯度小时的问题，只使用最后一个神经元得到的向量 $$c$$ 效果不理想；
3. 与人类阅读时的注意力方式不同，人类总是把注意力放在当前句子上

Attention（注意力机制），是一种将模型注意力放在当前翻译单词上的一种机制。例如翻译“I have a cat"，翻译我时注意力在”I“上，翻译猫时注意力在”cat“上。

使用Attention后，Decoder的输入就不是固定的上下文向量 $$c$$ 了，而是根据当前翻译的信息，计算当前 $$c$$ 。

![Attention](../.gitbook/assets/image%20%2810%29.png)

Attention需要保留Encoder每一个神经元的隐藏层向量 $$h$$ ，然后Decoder的第 $$t$$ 个神经元要根据上一个神经元的隐藏层向量 $$h`_{t-1}$$ 计算当前状态与Encoder每一个神经元的相关行 $$e_{t}$$ 。 $$e_{t}$$ 是一个N维的向量（Encoder神经元个数维N），若$$e_{t}$$第 $$i$$ 维越大，则说明当前节点与Encoder第 $$i$$ 个神经元的相关性越大。$$e_{t}$$的计算方式有很多，即相关性系数的计算函数 $$a$$ 有很多种：

![Attention &#x76F8;&#x5173;&#x6027;](../.gitbook/assets/image%20%285%29.png)

 上面得到相关性向量 $$e_{t}$$  后，需要进行归一化，使用 softmax 归一化。然后用归一化后的系数融合 Encoder 的多个隐藏层向量得到 Decoder 当前神经元的上下文向量 $$c_{t}$$ ：

![Attention &#x8BA1;&#x7B97;&#x4E0A;&#x4E0B;&#x6587;c](../.gitbook/assets/image%20%282%29.png)

总体流程图

![&#x603B;&#x4F53;&#x6D41;&#x7A0B;](../.gitbook/assets/image%20%283%29.png)

### Teacher Forcing

### Beam Search

## 参考

[https://www.jianshu.com/p/80436483b13b](https://www.jianshu.com/p/80436483b13b)

[https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)



