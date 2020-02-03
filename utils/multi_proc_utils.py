# -*- coding:utf-8 -*-
# Created by LuoJie at 11/17/19

import pandas as pd
import numpy as np
from multiprocessing import cpu_count, Pool

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