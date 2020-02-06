#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # gpu报错，使用cpu
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('utils')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.wv_loader import get_embedding_matrix,get_vocab
from utils.config import *
from utils.data_loader import build_dataset,load_dataset

from gensim.models.word2vec import LineSentence,Word2Vec
import tensorflow as tf
from model_layers import seq2seq_model
import time
build_dataset(train_data_path,test_data_path)

# 加载数据集
train_x,train_y,test_x,wv_model = load_dataset()
# 加载vocab
vocab,reverse_vocab = get_vocab(save_wv_model_path)
# 加载预训练权重
embedding_matrix,embedding_dim = get_embedding_matrix(wv_model)

def transform_data(sentence,vocab):
	unk_index = vocab['<UNK>']
	# 字符切分成词
	words = sentence.split(' ')
	# 按照vocab的index进行转换
	ids = [vocab[word] if word in vocab else unk_index for word in words]
	return ids
# 第一步：将词转换成索引
train_ids_x = train_x.apply(lambda x:transform_data(x,vocab))
train_ids_y = train_y.apply(lambda x:transform_data(x,vocab))
test_ids_x = test_x.apply(lambda x:transform_data(x,vocab))

# 索引转矩阵->为了丢进模型
train_data_x = np.array(train_ids_x.tolist())
train_data_y = np.array(train_ids_y.tolist())
test_data_x = np.array(test_ids_x.tolist())

# 输入长度 train_data_x.shape -> ((82871, 260))
input_length = train_data_x.shape[1]
# 输出长度 train_data_y.shape -> (82871, 33)
output_sequence_length = train_data_y.shape[1]
# 词表大小
vocab_size = len(vocab)

## 取部分数据进行训练,正式训练不要这一部分
sample_num = 640
train_x = train_data_x[:sample_num]
train_y = train_data_y[:sample_num]
# 训练集的长度
BUFFER_SIZE = len(train_x) # 640
# 输入的长度
max_length_inp = train_x.shape[1] # 260
# 输出的长度
max_length_targ = train_y.shape[1] # 33

BATCH_SIZE = 64

# 训练一轮需要迭代多少步
steps_per_epoch = len(train_x) // BATCH_SIZE # 10

# 词向量的维度
embedding_dim = 300

# 隐藏层单元数
units = 1024

# 词表大小
vocab_size = len(vocab)

# 构建训练集
dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)


class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, embedding_matrix, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
												   weights=[embedding_matrix],
												   trainable=False)
		self.gru = tf.keras.layers.GRU(self.enc_units,
									   return_sequences=True,
									   return_state=True,
									   recurrent_initializer='glorot_uniform')

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden)
		return output, state

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, query, values):
		'''
		query:上次GRU隐藏层
		values:编码器的编码结果enc_output
		在seq2seq模型中，s_t是后面query向量，而编码过程中的隐藏状态h_i是values
		'''
		hidden_with_time_axis = tf.expand_dims(query, 1)  # 增维
		# 计算注意力权重值
		score = self.V(tf.nn.tanh(
			self.W1(values) + self.W2(hidden_with_time_axis)))
		# attention_weight shape == (batch_size,max_length,1)
		attention_weights = tf.nn.softmax(score, axis=1)

		# 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
		# context_vector shape after sum == (batchz_size,hidden_size)
		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights


class Decoder(tf.keras.Model):
	def __init__(self, vocb_size, embedding_dim, embedding_matrix, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
												   trainable=False)
		self.gru = tf.keras.layers.GRU(self.dec_units,
									   return_sequences=True,
									   return_state=True,
									   recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		self.attention = BahdanauAttention(self.dec_units)

	def call(self, x, hidden, enc_output):
		# 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
		# enc_output shape == (batch_size,max_length,hidden_size)
		context_vector, attention_weights = self.attention(hidden, enc_output)

		# x shape after passing through embedding == (batch_size,1,embedding_dim)
		x = self.embedding(x)

		# 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
		# x shape after concatenation == (batch_size,1,embdding_dim+hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x)

		# output shape == (batch_size,vocab)
		x = self.fc(output)

		return x, state, attention_weights

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction='none')

# loss函数定义
pad_index = vocab['<PAD>']
def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,pad_index))
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask,dtype = loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
# 保存点设置
checkpoint_dir = 'data/checkpoints/train_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(optimizer = optimizer,
                                encoder = encoder,
                                decoder = decoder)



# 训练
@tf.function
def train_step(inp, targ, enc_hidden):
	loss = 0

	with tf.GradientTape() as tape:
		# 1.构建encoder
		enc_output, enc_hidden = encoder(inp, enc_hidden)
		# 2. 复制
		dec_hidden = enc_hidden
		# 3. <START> * BATCH_SIZE
		dec_input = tf.expand_dims([vocab['<START>']] * BATCH_SIZE, 1)
		# Techer forcing - feeding the target as the next input
		for t in range(1, targ.shape[1]):
			# decoder(x,hidden,enc_output)
			predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

			loss += loss_function(targ[:, t], predictions)
			dec_input = tf.expand_dims(targ[:, t], 1)
		batch_loss = (loss / int(targ.shape[1]))
		variables = encoder.trainable_variables + decoder.trainable_variables
		gradients = tape.gradient(loss, variables)
		optimizer.apply_gradients(zip(gradients, variables))
		return batch_loss

EPOCHS = 10

for epoch in range(EPOCHS):
	start = time.time()

	# 初始化隐藏层
	enc_hidden = encoder.initialize_hidden_state()
	total_loss = 0
	for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
		batch_loss = train_step(inp, targ, enc_hidden)
		total_loss += batch_loss

		if batch % 1 == 0:
			print('Epoch {} Batch {} Loss{:.4f}'.format(epoch + 1,
														batch,
														batch_loss.numpy()))
	# saving (checkpoint) the model every 2 epochs
	if (epoch + 1) % 2 == 0:
		chekpoint.save(file_prefix=checkpoint_prefix)

	print('Epoch {} Loss{:.4f}'.format(epoch + 1,
									   total_loss / steps_per_epoch))
	print('Time taken for 1 epoch{} sec\n'.format(time.time() - start))


from utils.data_loader import preprocess_sentence
import matplotlib
from matplotlib import font_manager
# 解决中文乱码
font = font_manager.FontProperties(fname='data/Truetype/simhei.ttf')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def evaluate(sentence):
	attention_plot = np.zeros((max_length_targ, max_length_inp + 2))
	inputs = preprocess_sentence(sentence, max_length_inp, vocab)
	inputs = tf.convert_to_tensor(inputs)
	result = ''
	hidden = [tf.zeros((1, units))]
	enc_out, enc_hidden = encoder(inputs, hidden)
	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([vocab['<START>']], 0)
	for t in range(max_length_targ):
		predictions, dec_hidden, attention_weights = decoder(dec_input,
															 dec_hidden,
															 enc_out)
		#         print(predictions)# (1, 1, 32799)
		#         print(predictions[0]) # (1, 32799)
		#         print(tf.argmax(predictions[0]).numpy()) # (32799,)

		# storing the attention weights to plot later on
		attention_weights = tf.reshape(attention_weights, (-1,))
		print('1')
		attention_plot[t] = attention_weights.numpy()
		predicted_id = tf.argmax(predictions[0][0]).numpy()

		print('2', predicted_id, type(predicted_id))

		result += reverse_vocab[predicted_id] + ' '
		#         result = [''.join(reverse_vocab[i]) for i in range(len(predicted_id))]
		print('3')
		if reverse_vocab[predicted_id] == '<STOP>':
			return result, sentence, attention_plot

		# the predicted ID is fed back into the model
		dec_input = tf.expand_dims([predicted_id], 0)
	return result, sentence, attention_plot


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.matshow(attention, cmap='viridis')

	fontdict = {'fontsize': 14, 'fontproperties': font}

	ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
	ax.set_yticklabels([''] + predict_sentence, fontdict=fontdict)

	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	plt.show()


def translate(sentence):
	result, sentence, attention_plot = evaluate(sentence)

	print('Input: %s' % (sentence))
	print('Predicted translation:{}'.format(result))

	attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
	plot_attention(attention_plot, sentence.split(''), result.split(' '))

sentence='漏机油 具体 部位 发动机 变速器 正中间 位置 拍 中间 上面 上 已经 看见'
translate(sentence)