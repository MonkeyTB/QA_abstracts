#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:unicode_escape -*-
# 为了解决read_csv乱码的情况，加上面这句，读取的时候加encoding='gbk'

import time
import numpy as np
import tensorflow.compat.v1 as tf
import random
import pandas as pd
from collections import Counter

cut_text_path = '../data/cut_text.csv'

# load data set
def loader_data(path):
	df = pd.read_csv(path,encoding='gbk',header = None).rename(columns = {0:'text'})
	## cat sentence
	text = '	'.join(df['text'])
	return text

text = loader_data(cut_text_path)
print(len(text.split('	')))
#　data pre-processing
def processing(text,freq = 5):
	'''
	:param text: 文本数据
	:param freq: 词频阈值
	:return:
	'''
	text = text.lower().replace('。', ' <PERIOD> ').replace(',', ' <COMMA> ').replace('，', ' <COMMA> ')\
		.replace('"', ' <QUOTATION_MARK> ').replace(';', ' <SEMICOLON> ').replace('!', ' <EXCLAMATION_MARK> ')\
		.replace('?', ' <QUESTION_MARK> ').replace('(', ' <LEFT_PAREN> ').replace(')', ' <RIGHT_PAREN> ')\
		.replace('--', ' <HYPHENS> ').replace('?', ' <QUESTION_MARK> ').replace('：', ' <COLON> ')
	words = text.split('	')
	words_count = Counter(words)
	print(words_count)
	trimmed_words = [word for word in words if words_count[word] > freq]
	return trimmed_words
# 问题：为啥不全训练要去掉低频词？
words = processing(text)
print(words[:20])
print(len(words))

# 构建隐射表
vocab = set(words)
print(len(vocab))

vocab2int = {word:index for index,word in enumerate(vocab)}
int2vocab = {index:word for index,word in enumerate(vocab)}
print('total words:{}'.format(len(words)))
print('unique words:{}'.format(len(set(words))))

# 原始文本进行vocab到int的转换
int_words = [vocab2int[w] for w in words]

int_word_counts = Counter(int_words)

# $P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}$
t = 1e-3 # t值
threshold = 0.7 #剔除阈值概率

#统计单词出现的频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
#计算单词频率
word_freqs = {w:c/total_count for w,c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w:1-np.sqrt(t/word_freqs[w]) for w in int_word_counts}
#对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]
drop_words = [int2vocab[w] for w in int_words if prob_drop[w] > threshold]
# print(set(drop_words)) # 车主说、技师说也被剔除？原因
# print(int_words)
# print(train_words)
# print(int_words)

def get_targets(words,idx,window_size = 5):
	'''
	:param words:单词列表
	:param idx: input word的索引号
	:param window_size: 窗口大小
	:return:
	'''
	target_window = np.random.randint(1,window_size+1)
	#这里考虑input word前面单词不够的情况
	start_point = idx - target_window if (idx-target_window)>0 else 0
	end_point = idx + target_window
	#output words（窗口上下文单词）
	targets = set(words[start_point:idx] + words[idx+1:end_point+1])
	return list(targets)


def get_batches(words, batch_size, window_size=5):
	'''
	构造一个获取batch的生成器
	'''
	n_batches = len(words) // batch_size

	# 仅取full batches
	words = words[:n_batches * batch_size]

	for idx in range(0, len(words), batch_size):
		x, y = [], []
		batch = words[idx: idx + batch_size]
		for i in range(len(batch)):
			batch_x = batch[i]
			batch_y = get_targets(batch, i, window_size)
			# 由于一个input word会对应多个output word，因此需要长度统一
			x.extend([batch_x] * len(batch_y))
			y.extend(batch_y)
		yield x, y

# 构建网络
train_graph = tf.Graph()
with train_graph.as_default():
	inputs = tf.placeholder(tf.int32,shape=[None],name = 'inputs')
	labels = tf.placeholder(tf.int32,shape=[None,None],name = 'labels')
vocab_size = len(int2vocab)
embedding_size = 100
print(vocab_size)
with train_graph.as_default():
	# 嵌入层权重矩阵
	embedding = tf.Variable(tf.random.uniform([vocab_size,embedding_size],-1,1) )
	# 实现lookup
	embed = tf.nn.embedding_lookup(embedding,inputs)
	print(embed)

n_sampled = 1000
with train_graph.as_default():
	softmax_w = tf.Variable(tf.truncated_normal([vocab_size,embedding_size],stddev=0.1))
	softmax_b = tf.Variable(tf.zeros(vocab_size))

	# 计算negative sampling下的损失nec
	loss = tf.nn.sampled_softmax_loss(weights=softmax_w,biases=softmax_b,inputs=embed,labels=labels,num_sampled=n_sampled,num_classes=vocab_size)

	cost = tf.reduce_mean(loss)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

#验证
with train_graph.as_default():
# # 随机挑选一些单词
#     valid_size = 16
#     valid_window = 10
#     # 从不同位置各选8个单词
#     valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
#     valid_examples = np.append(valid_examples,
#                                random.sample(range(1000,1000+valid_window), valid_size//2))
	valid_examples = [vocab2int['丰田'],
					  vocab2int['发动机'],
					  vocab2int['助力'],
					  vocab2int['方向机'],
					  vocab2int['雨刮器']]
	valid_size = len(valid_examples)
	# 验证单词集
	valid_dataset = tf.constant(valid_examples,dtype=tf.int32)
	#计算每个词向量的摸并进行单位化
	norm = tf.sqrt(tf.reduce_mean(tf.square(embedding),1,keepdims=True))
	normalized_embedding = embedding / norm
	# 查找验证单词的词向量
	valid_embedding = tf.nn.embedding_lookup(normalized_embedding,valid_dataset)
	# 计算余弦相似度
	similarity = tf.matmul(valid_embedding,tf.transpose(normalized_embedding))
print(len(train_words))

# 训练模型
epochs = 2
batch_size = 2000
window_size = 3
with train_graph.as_default():
	saver = tf.train.Saver()
with tf.Session(graph = train_graph) as sess:
	iteration = 1
	loss = 0
	#添加节点用于初始化所有变量
	sess.run(tf.global_variables_initializer())
	for e in range(1,epochs+1):
		# 获得batch数据
		batches = get_batches(train_words,batch_size,window_size)
		start = time.time()
		for x,y in batches:
			feed = {inputs: x,
					labels: np.array(y)[:, None]}
			train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
			loss += train_loss

			if iteration % 1000 == 0:
				end = time.time()
				print("Epoch {}/{}".format(e, epochs),
					  "Iteration: {}".format(iteration),
					  "Avg. Training loss: {:.4f}".format(loss / 1000),
					  "{:.4f} sec/batch".format((end - start) / 1000))
				loss = 0
				start = time.time()

			# 计算相似的词
			if iteration % 1000 == 0:
				print('*' * 100)
				# 计算similarity
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = int2vocab[valid_examples[i]]
					top_k = 8  # 取最相似单词的前8个
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log = 'Nearest to [%s]:' % valid_word
					for k in range(top_k):
						close_word = int2vocab[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
				print('*' * 100)

			iteration += 1

		save_path = saver.save(sess, "checkpoints/text8.ckpt")
		embed_mat = sess.run(normalized_embedding)
