#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('utils')

import pandas as pd
import numpy as np
from utils.data_loader import build_dataset
from utils.config import *
from gensim.models.word2vec import LineSentence,Word2Vec

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import GRU,Input,Dense,TimeDistributed,Activation,RepeatVector,Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy



def get_vocab(model):
	# 更新vocab
	vocab = {word:index for index,word in enumerate(model.wv.index2word)}
	reverse_vocab = {index:word for index,word in enumerate(model.wv.index2word)}
	#更新词向量矩阵
	embedding_matrix = model.wv.vectors
	return embedding_matrix,vocab,reverse_vocab
def transform_data(sentence,vocab):
	unk_index = vocab['<UNK>']
	# 字符切分成词
	words = sentence.split('	')
	# 按照vocab的index进行转换
	ids = [vocab[word] if word in vocab else unk_index for word in words]
	return ids
# 搭建简易模型
def seq2seq(input_length, output_sequence_length, embedding_matrix, vocab_size):
    model = Sequential() # 模型结构（序列式）
    model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], trainable=False,input_length=input_length))
	# 编码
    model.add(Bidirectional(GRU(300, return_sequences=False))) # 双向GRU
    model.add(Dense(300, activation="relu")) # 线性层

    model.add(RepeatVector(output_sequence_length)) # 输入2D，输出3D,线性层出来的维度2D，重复output_sequence_length次，就变成3D了，为了解码的输入
	#解码
    model.add(Bidirectional(GRU(300, return_sequences=True))) # 双向GRU
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy,optimizer=Adam(1e-3))

    model.summary() # 打印参数
    return model


if __name__ == '__main__':
	# 第二节内容，直接用了老师封装好的，自己之前封装的没有老师的好
	train_x,train_y,test_x,wv_model = build_dataset(train_data_path,test_data_path)
	embedding_matrix,vocab,reverse_vocab = get_vocab(wv_model)
	# 第一步：将词转换成索引
	train_ids_x = train_x.apply(lambda x:transform_data(x,vocab))
	train_ids_y = train_y.apply(lambda x:transform_data(x,vocab))
	test_ids_x = test_x.apply(lambda x:transform_data(x,vocab))

	# 索引转矩阵->为了丢进模型
	train_data_x = np.array(train_ids_x.tolist())
	train_data_y = np.array(train_ids_y.tolist())
	test_data_x = np.array(test_ids_x.tolist())

	# 第二步：seq2seq
	# 输入长度
	input_length = train_data_x.shape[1]
	# 输出长度
	output_sequence_length = train_data_y.shape[1]
	# 词表大小
	vocab_size = len(vocab)

	# 词向量矩阵
	embedding_matrix = wv_model.wv.vectors

	print(input_length,output_sequence_length,embedding_matrix,vocab_size)
	# 模型构建
	model = seq2seq(input_length,output_sequence_length,embedding_matrix,vocab_size)

	# 模型训练
	model.fit(train_data_x,train_data_y,batch_size=32,epoch=1,validation_split=0.2)

	# 模型保存
	model.save('data/seq2seq_model.h')

	print('词表大小：',vocab_size)
	# 模型预测
	test_data_y = model.predict(test_data_x) # 这个模型的输出是怎么确定的？

	print('词表大小：',vocab_size)
