---
description: 分享会一些问题的回答
---

# 2、机器学习过拟合欠拟合问题

1、当遇到过拟合问题，当优化开发集和训练集的差异时，面临整体误差上升，有什么好的解决办法

* 过拟合，数据简单，模型复杂，开发训练分布不一致，优化数据，整体误差上升，欠拟合，模型不能拟合数据，第一：模型不适用这个数据，第二：数据量不够或者数据特征不够

2、拟牛顿法比牛顿法的优点是什么

*   **牛顿法是迭代算法，每一步都需求解目标函数的海塞矩阵（Hessian Matrix），计算比较复杂。拟牛顿法通过正定矩阵近似海塞矩阵的逆矩阵或海塞矩阵，简化了这一计算过程。**

    ![\[公式\]](https://www.zhihu.com/equation?tex=x\_{k%2B1}%3Dx\_{k}-\lambda\_k+H^{-1}\_{k}g\_k\\\\)
* ![img](https://pic1.zhimg.com/v2-dac11d82ecb2566f54ce8b518d51293c\_b.webp)
* 牛顿法虽然收敛速度快，但是需要计算海塞矩阵的逆矩阵 ![\[公式\]](https://www.zhihu.com/equation?tex=H%5E%7B-1%7D) ，而且有时目标函数的海塞矩阵无法保持正定，从而使得牛顿法失效。为了克服这两个问题，人们提出了拟牛顿法。这个方法的基本思想是：不用二阶偏导数而构造出可以近似海塞矩阵（或海塞矩阵的逆）的正定对称阵。不同的构造方法就产生了不同的拟牛顿法（**DFP**/**BFGS算法**）。
*   牛顿法是二阶收敛，梯度下降法是一阶收敛，所以牛顿法更快，下图形象化地显示了这一点：

    ![img](https://pic4.zhimg.com/80/v2-4980eec3aa27524f9bc70e57dafe56ab\_720w.jpg)

    其中，红色路径代表牛顿法，绿色路径代表梯度下降法。

    * 牛顿法和深度学习

    深度学习中，往往采用梯度下降法作为优化算子，而很少采用牛顿法，主要原因有以下几点：

    1. 神经网络通常是非凸的，这种情况下，牛顿法的收敛性难以保证；
    2. 即使是凸优化，只有在迭代点离全局最优很近时，牛顿法才会体现出收敛快的优势；
    3. 可能被鞍点吸引。(鞍点：不是局部极值点的驻点)

3、推荐中如何模拟人的期望与偏好，是基于学术理论或研究建立一个用户偏好模型吗？标签怎么弄？

* 是的
* 假设对一个交友活动网站来说，对于一个用户，有基础特征，行为特征等，其基础特征会包含性别、年龄、身高、体重等基础信息，除此之外我们假设一个用户老是搜索电影，那么当前用户的偏好就可以打上爱好：电影，而另一个用户喜欢搜索文学，那么可以打上爱好：读书，或者利用一些组合特征打上其他标签，比如性格（读书，女）安静，（篮球，男）活泼，那么我们再推荐一些线下读书交流会给性格安静的，而推荐一些团体运动会给活泼的

4、某项任务上机器已经超越人类，那么当前任务上人类存在的价值是？ 5、短文本分类的方法？（材料无关）

* 短文本分类首要问题需要标签，
  * 可以先通过聚类（LDA），相同簇的分类比较接近，标注起来比较方便；其次对应较少的类，也可以抽取到给人工标注
  * 基于关键词标注，假设要给新闻标题分类，那么对于“体育”类，“足球”、“篮球”、“NBA”等就可以作为这个类的关键词
* 朴素贝叶斯、逻辑回归、支持向量机、GBDT、随机森林
* 深度学习（FastText，text-cnn，bert，图模型）

6、近似搜索算法如何避免陷入局部最优？

* beam search
* 遗传算法
  * 模拟物竞天择的生物进化过程，通过维护一个潜在解的群体执行了多方向的搜索，并支持这些方向上的信息构成和交换。是以面为单位的搜索，比以点为单位的搜索，更能发现全局最优解。（在遗传算法中，有很多袋鼠，它们降落到喜玛拉雅山脉的任意地方。这些袋鼠并不知道它们的任务是寻找珠穆朗玛峰。但每过几年，就在一些海拔高度较低的地方射杀一些袋鼠，并希望存活下来的袋鼠是多产的，在它们所处的地方生儿育女。）（或者换个说法。从前，有一大群袋鼠，它们被莫名其妙的零散地遗弃于喜马拉雅山脉。于是只好在那里艰苦的生活。海拔低的地方弥漫着一种无色无味的毒气，海拔越高毒气越稀薄。可是可怜的袋鼠们对此全然不觉，还是习惯于活蹦乱跳。于是，不断有袋鼠死于海拔较低的地方，而越是在海拔高的袋鼠越是能活得更久，也越有机会生儿育女。就这样经过许多年，这些袋鼠们竟然都不自觉地聚拢到了一个个的山峰上，可是在所有的袋鼠中，只有聚拢到珠穆朗玛峰的袋鼠被带回了美丽的澳洲。）
* 退火算法
  * 这个方法来自金属热加工过程的启发。在金属热加工过程中，当金属的温度超过它的熔点（Melting Point）时，原子就会激烈地随机运动。与所有的其它的物理系统相类似，原子的这种运动趋向于寻找其能量的极小状态。在这个能量的变迁过程中，开始时，温度非常高， 使得原子具有很高的能量。随着温度不断降低，金属逐渐冷却，金属中的原子的能量就越来越小，最后达到所有可能的最低点。利用模拟退火的时候，让算法从较大的跳跃开始，使到它有足够的“能量”逃离可能“路过”的局部最优解而不至于限制在其中，当它停在全局最优解附近的时候，逐渐的减小跳跃量，以便使其“落脚 ”到全局最优解上。（在模拟退火中，袋鼠喝醉了，而且随机地大跳跃了很长时间。运气好的话，它从一个山峰跳过山谷，到了另外一个更高的山峰上。但最后，它渐渐清醒了并朝着它所在的峰顶跳去。）
* 进化策略

7、降低模型过拟合和欠拟台风险的方法?

* 过拟合直观解释，再对训练数据上进行拟合时，需要照顾每个点，从而导致拟合函数波动大，即方差大。
* 误差图判断过拟合还是欠拟合：

> 模型再训练集与测试集上误差均很大，则说明模型bias很大，欠拟合
>
> 训练集和测试集误差之间有很大的Gap，则说明Variance很大，过拟合

* 欠拟合解决方法

> * 增加新的特征，考虑加入组合特征、高次特征，来增大假设空间
> * 尝试非线性模型，比如SVM、决策树、DNN等模型
> * 有正则项则尝试降低正则项参数\lambda

* 过拟合解决方法

> * 交叉验证，通过交叉验证得到最优的模型
> * 特征选择，减少特征数或者使用较少的特征组合，对于区间化离散特征，增大划分的空间
> * 正则化，常用的有L\_1,L\_2正则，而且L\_1正则还可以自动进行特征选择
> * 如果有正则项考虑增大正则项参数\lambda
> * 增加训练数据可以有效的避免过拟合
> * Bagging，将多个弱学习器Bagging一下效果会好很多，比如随机森林
> * DNN中常用的方法
>   * 早停。本质还是交叉验证策略，选择合适的训练次数，避免训练的网络过度拟合训练数据
>   * 集成学习策略。利用Bagging思路来正则化，首先对原始的m个训练样本进行又放回的随机采样，构建N组m个样本的数据集，然后分别用N组数据训练DNN，即得到N个DNN模型的ｗ，ｂ参数组合，最后对N个DNN模型的输出用加权平均法或者投票法决定最终输出。缺点就是参数增加了N倍，导致训练花费更多的时间和空间，因此N一般不能太多，例如5-10个。
>   *   Dropout策略。
>
>       > 训练的时候一一定的概率p让神经元失活，类似于Bagging策略，测试时所有神经元都参与，但是在其输出上乘以1-p.
>   *   simpler model structure（简化模型）
>
>       > 简单模型拟合复杂数据时，导致模型很难拟合数据的真实分布，这边是模型欠拟合了，同样的，复杂模型拟合简单数据，会导致数据的噪声也被拟合了，导致模型再训练集上效果非常好，但泛化性能很差。
>   *   regularization（正则化）
>
>       > L\_1 正则项会产生稀疏解
>       >
>       > L\_2正则项会产生比较小的解
>       >
>       > 假设均方差误差损失函数
>       >
>       > &#x20;J(w,b)=\frac{1}{2m}\sum\_{i=1}^{m}||a^L-y||\_2^2
>       >
>       > L\_2通常只针对系数矩阵w,而不针对偏置系数b,则损失函数为
>       >
>       > J(w,b)=\frac{1}{2m}\sum\_{i=1}^{m}||a^L-y||\_2^2+\frac{\lambda}{2m}\sum\_{L=2}^L||w||\_2^2
>       >
>       > 反向传播从:
>       >
>       > w^l=w^l-\alpha\sum\_{i=1}^{m}\sigma^{i,l}(a^{x,l-1})^T
>       >
>       > 变为:
>       >
>       > w^l=w^l-\alpha\sum\_{i=1}^{m}\sigma^{i,l}(a^{x,l-1})^T-\alpha\lambda w^l
> *   data augmentation(数据集扩增)
>
>     > 有效前提是训练集与将来的数据是**独立同分布**的,或者近似独立同分布
>     >
>     > * 从数据源头采集更多的数据
>     > * 复制原有数据并加上随机噪声
>     > * 重采样
>     > * 根据当前数据估计数据分布参数,使用该分布产生更多数据
> * Bootstrap/Bagging（封装）
> * ensemble（集成）
> *   early stopping（提前终止训练）
>
>     > 迭代次数截断的方法来防止过拟合,再模型对训练数据集迭代收敛之前停止迭代防止过拟合.
> * utilize invariance（利用不变性）
> * Bayesian（贝叶斯方法）
