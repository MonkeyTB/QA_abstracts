#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import re
jieba.load_userdict('../data/user_dict.txt')
train_path = '../data/AutoMaster_TrainSet.csv'
# train_path = '../data/test.csv'
test_path = '../data/AutoMaster_TestSet.csv'

df = pd.read_csv(train_path,encoding='utf-8')
# print(df.info())
# print(df.head())
# print(df.iloc[1])

# 对缺失数据的行删除---处理方式一
df = df.dropna(how='any')
# 创建停用词
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist('../data/哈工大停用词.txt')
# 切词操作
def text_cut(texts):
	sentences = ''
	texts = re.sub(r'\[|\]',' ',texts) # 正则掉[]
	texts = texts.replace(' ','').replace('--','').replace('(','（').replace(')','）').split('|')
	for i in range(len(texts)):
		cut = list(jieba.cut(texts[i],cut_all=False))
		sentences += '	'.join(str(m) for m in cut if m not in stopwords)
		sentences += '	'

	return sentences

text_list = []
for i in range(len(df)):
	sentences = ''
	for j in range(1,5):
		sentences += text_cut(df.iloc[i][j])
	text_list.append(sentences)

# 先创建并打开一个文本文件
file = open('..\data\cut_text.txt', 'w')
# 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
for k in text_list:
	try:
		file.write(str(k) + '\n')
	except:
		print(k)

# 关闭文件
file.close()