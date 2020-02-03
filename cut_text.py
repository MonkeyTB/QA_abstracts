#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import re
from multiprocessing import cpu_count, Pool

jieba.load_userdict('../data/user_dict.txt')
train_path = '../data/AutoMaster_TrainSet.csv'
test_path = '../data/AutoMaster_TestSet.csv'
stopwords_path = '../data/哈工大停用词.txt'
merged_cut_text_path = '../data/merged_cut_text.csv'
test_cut_text_path = '../data/test_cut_text.csv'
train_cut_text_path = '../data/train_cut_text.csv'
test_cut_text_path = '../data/test_cut_text.csv'
# 切词操作
def text_cut(texts):
	# 创建停用词
	def stopwordslist(filepath):
		stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
		return stopwords
	stopwords = stopwordslist(stopwords_path)

	# texts = re.sub(r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|技师说|车主说','',texts)
	texts = re.sub(r'[!\[\]]+|[:：()（）\|…]+|技师说|车主说','',texts) # 正则掉  暂时认为最优
	# 句子中多余的空格去掉
	texts = texts.replace(' ','')
	cut = list(jieba.cut(texts,cut_all=False))
	sentences = '	'.join(str(m) for m in cut if m not in stopwords)
	return sentences

def data_frame_proc(df):
	'''
	:param df:
	:return:
	'''
	# 批量预处理 训练集和测试集
	for col_name in ['Brand','Model','Question','Dialogue']:
		df[col_name] = df[col_name].apply(text_cut)
	# 对训练集中的Report单独处理
	if 'Report' in df.columns:
		df['Report'] = df['Report'].apply(text_cut)
	return df

# 并行计算
def  parallelize(df,func):
	'''
	:param df:DataFram data
	:param func: 预处理函数
	:return:
	'''
	# cpu 数量
	cores = cpu_count()
	# 分块个数
	partitaions = cores
	data_split = np.array_split(df,partitaions)
	# 线程池
	pool = Pool(cores)
	# 数据分发 和并
	data = pd.concat(pool.map(func,data_split))
	# 关闭线程池
	pool.close()
	# 执行完关闭后不会有新的进程加入到pool，join函数等待所有子进程结束
	pool.join()
	return data

# 数据处理
def data_handle(train_path,test_path):
	df_train = pd.read_csv(train_path,encoding='utf-8')
	df_test = pd.read_csv(test_path,encoding='utf-8')
	# 对缺失空值填充
	df_train.dropna(subset=['Brand','Model','Question','Dialogue','Report'],how='any',inplace=True)
	df_test.dropna(subset=['Brand','Model','Question','Dialogue'],how='any',inplace=True)
	# 并行处理
	df_train = parallelize(df_train,data_frame_proc)
	df_test = parallelize(df_test,data_frame_proc)

	# 保存
	df_train.to_csv(train_cut_text_path,index = None,header = True)
	df_test.to_csv(test_cut_text_path,index = None,header = True)

	# 合并训练数据
	# 全保留
	'''
	df_merged = pd.concat([df_train['Brand'],df_train['Model'],df_train['Question'],
						   df_train['Dialogue'],df_train['Report'],df_test['Brand'],
						   df_test['Model'], df_test['Question'], df_test['Dialogue']],
						  axis=0)'''
	# 去除车和车型
	df_merged = pd.concat([df_train['Question'],
						   df_train['Dialogue'],df_train['Report'],
						   df_test['Question'], df_test['Dialogue']],
						  axis=0)

	df_merged.to_csv(merged_cut_text_path,index = None,header = False)
	return df_train,df_test,df_merged


if __name__  == '__main__':
	train_df,test_df,merged_df = data_handle(train_path,test_path)





