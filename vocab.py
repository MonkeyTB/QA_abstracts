#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import re
import  time
start_time = time.time()
jieba.load_userdict('../data/user_dict.txt')
train_path = '../data/AutoMaster_TrainSet.csv'
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
	texts = re.sub(r'\[|\]',' ',texts) # 正则掉[]
	texts = texts.replace(' ','').replace('--','').replace('(','（').replace(')','）').split('|')
	for i in range(len(texts)):
		sentences = list(jieba.cut(str(texts[i]),cut_all=False))
		for k in range(len(sentences)):
			if sentences[k] not in stopwords:
				if sentences[k] in vocab_dict:
					vocab_dict[sentences[k]] += 1
				else:
					vocab_dict[sentences[k]] = 1
vocab_dict = {}
for i in range(len(df)):
	for j in range(1,5):
		text_cut(df.iloc[i][j])
# print(vocab_dict)
# print(type(vocab_dict))

# 先创建并打开一个文本文件
file = open('..\data\Vocab.txt', 'w')
# 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
i = 0
for k, v in vocab_dict.items():
	if v > 5:
		try:#  1	出来个这么个词
			file.write(str(k) + ' ' + str(i) + '\n')
			i += 1
		except:
			print(k,v)
# 关闭文件
file.close()
end_time = time.time()
print(end_time-start_time)