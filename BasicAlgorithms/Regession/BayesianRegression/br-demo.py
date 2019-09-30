#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
author ： duanxxnj@163.com
time : 2016-06-21-09-21

贝叶斯脊回归
这里在一个自己生成的数据集合上测试贝叶斯脊回归

贝叶斯脊回归和最小二乘法(OLS)得到的线性模型的参数是有一定的差别的
相对于最小二乘法(OLS)二样，贝叶斯脊回归得到的参数比较接近于0

贝叶斯脊回归的先验分布是参数向量的高斯分布

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

from sklearn.linear_model import BayesianRidge, LinearRegression

###############################################################################
# 随机函数的种子
np.random.seed(int(time.time()) % 100)
# 样本数目为100，特征数目也是100
n_samples, n_features = 100, 100
# 生成高斯分布
X = np.random.randn(n_samples, n_features)
# 首先使用alpha为4的先验分布.
alpha_ = 4.
w = np.zeros(n_features)
# 随机提取10个特征出来作为样本特征
relevant_features = np.random.randint(0, n_features, 10)
# 基于先验分布，产生特征对应的初始权值
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_))

# 产生alpha为50的噪声
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# 产生目标数据
y = np.dot(X, w) + noise

###############################################################################
# 使用贝叶斯脊回归拟合数据
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)

# 使用最小二乘法拟合数据
ols = LinearRegression()
ols.fit(X, y)

###############################################################################
# 作图比较两个方法的结果
plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, 'b-', label="Bayesian Ridge estimate")
plt.plot(w, 'g-', label="Ground truth")
plt.plot(ols.coef_, 'r--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.show()
