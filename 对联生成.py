#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
print('NUMs GPUs Avaliable:',len(tf.config.experimental.list_physical_devices('GPU')))
# 1. 加载数据
# 1.1 加载数据集
root =  'data/couplet' # 数据根目录

with open(root + '/train/in.txt','r',encoding = 'utf-8') as f:
    data_in = f.read() # 全读取
with open(root + '/train/out.txt','r',encoding = 'utf-8') as f:
    data_out = f.read()
# 按换行符切分为一句
train_x = data_in.split('\n')
train_y = data_out.split('\n')
# 按空格切分为char字符
train_x = [data.split() for data in train_x]
train_y = [data.split() for data in train_y]
## 1.2 构造字典
import itertools
# 获取所有字
words = list(itertools.chain.from_iterable(train_x)) + list(itertools.chain.from_iterable(train_y))
'''
chain.from_iterable(iterables):
一个备用链构造函数，其中的iterables是一个迭代变量，生成迭代序列，此操作的结果与以下生成器代码片段生成的结果相同：
'''
# 列表去重
words = set(words)
# 构建vocab,index+1是为了unk为0，其他char从1开始
vocab = {word:index+1 for index,word in enumerate(words)}
# 添加unk标签
vocab['unk'] = 0
## 1.3 数据预处理
def get_max_length(data,max = -1):
    for i in range(len(data)):
        if len(train_x[i]) > max:
            max = len(train_x[i])
    return max
max = get_max_length(train_x)
max = get_max_length(train_y)

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 转换成索引
train_x_ids = [[vocab.get(word,0) for word in sen] for sen in train_x]
train_y_ids = [[vocab.get(word,0) for word in sen] for sen in train_y]

# 填充长度
train_x_ids = pad_sequences(train_x_ids,maxlen = max,padding = 'post')
train_y_ids = pad_sequences(train_y_ids,maxlen = max,padding = 'post')
print(train_y_ids.shape,train_x_ids.shape)
# 扩展维度
train_y_ids = train_y_ids.reshape(*train_y_ids.shape,1)
print(train_y_ids.shape)
# 2. 模型构建
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import GRU,LSTM,Input,Dense,TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

def seq2seq_model(input_length,output_sequence_length,vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size,output_dim=128,input_length=input_length))
    model.add(Bidirectional(LSTM(128,return_sequences=False)))
    model.add(Dense(128,activation='relu'))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size,activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,optimizers=Adam(1e-3))
    model.summary()
    return model
model = seq2seq_model(train_x_ids.shape[1],train_y_ids.shape[1],len(vocab))

# 2.1 模型训练
model.fit(train_x_ids,train_y_ids,batch_size=64,epochs = 1)
## 2.2 模型保存
model.save('data/epochs_10_batch_64_model.h5')
## 2.3 加载模型
# 加载模型包括权重和优化器
model = tf.keras.models.load_model('data/epochs_1_batch_64_model.h5')

## 2.4 模型预测
import numpy as np
input_sen = '飞流直下三千尺'
char2id = [vocab.get(i,0) for i in input_sen]
input_data = pad_sequences([char2id],max)
result = model.predict(input_data)[0][-len(input_sen):]
result_label = [np.argmax(i) for i in result]
dict_res = {i:j for j,i in vocab.items()}
print([dict_res.get(i) for i in result_label])

