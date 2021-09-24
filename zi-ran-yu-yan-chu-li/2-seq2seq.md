---
description: 通用seq2seq模型分析理解
---

# 2、seq2seq

## seq2seq 综述

seq2seq是一种循环神经网络的变种，包括编码器和解码器两部分，可用于机器翻译、对话系统、自动摘要

## 常见seq2seq结构

seq2seq是一种重要的RNN模型，也称为Encoder-Decoder模型，可以理解为 $$N*M$$ 的模型，模型包含两部分，Encoder部分编码序列的信息，将任意长度的信息编码到一个向量 $$c$$ 里，而Decoder部分通过向量 $$c$$ 将信息解码，并输出序列

