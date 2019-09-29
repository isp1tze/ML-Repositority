# from sklearn.datasets import load_iris
#
# #########################################
# #导入IRIS数据集
# iris = load_iris()
# #特征矩阵
# iris.data
# #目标向量
# iris.target
# print(iris.data.shape)
# print(iris.target.shape)
#
# #########################################
# # 均值方差
# from sklearn.preprocessing import StandardScaler
# StandardScaler().fit_transform(iris.data)
#
# #########################################
# from sklearn.preprocessing import MinMaxScaler
# #区间缩放，返回值为缩放到[0, 1]区间的数据
# MinMaxScaler().fit_transform(iris.data)
#
# #########################################
# # norm：可以为l1、l2或max，默认为l2
# # 若为l1时，样本各个特征值除以各个特征值的绝对值之和
# # 若为l2时，样本各个特征值除以各个特征值的平方之和
# # 若为max时，样本各个特征值除以样本中特征值最大的值
# from sklearn.preprocessing import Normalizer
# # 归一化，返回值为归一化后的数据
# Normalizer().fit_transform(iris.data)
#
# #########################################
# from sklearn.preprocessing import Binarizer
# # 二值化，阈值设置为3，返回值为二值化后的数据
# Binarizer(threshold=3).fit_transform(iris.data)
#
# #########################################
# from sklearn.preprocessing import OneHotEncoder
# # 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
# OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))
#
# #########################################
# # 补缺省值
# import numpy as np
# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 使用特征的均值进行填充，其余还有使用众数填充等,只需要把mean改成median即可
# data = np.array([np.nan, 2, 6, np.nan, 7, 6]).reshape(3,2)
# print(data)
# print(imp.fit_transform(data))
#
# #########################################
# # 多项式转换
# # 参数degree为度，默认值为2
# from sklearn.preprocessing import PolynomialFeatures
# PolynomialFeatures().fit_transform(iris.data)
#
# #########################################
# #自定义转换函数为对数函数的数据变换
# #第一个参数是单变元函数
# from numpy import log1p
# from sklearn.preprocessing import FunctionTransformer
# FunctionTransformer(log1p).fit_transform(iris.data)
#
# #########################################
# # 去掉取值变化小的特征
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(0.5))
# sel.fit_transform(iris.data)
#
# # 单变量特征选择（Univariate feature selection）
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# iris = load_iris()
# X, y = iris.data, iris.target
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
#
# # 递归特征消除Recursive feature elimination （RFE）
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
#
# #递归特征消除法，返回特征选择后的数据
# #参数estimator为基模型
# #参数n_features_to_select为选择的特征个数
# RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
#
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
#
# #GBDT作为基模型的特征选择
# SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
#
#
# # Feature selection using SelectFromModel
# import matplotlib.pyplot as plt
# import numpy as np
#
# from sklearn.datasets import load_boston
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LassoCV
#
#
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LogisticRegression
#
# #带L1惩罚项的逻辑回归作为基模型的特征选择
# SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
#
#
#
# # Load the boston dataset.
# boston = load_boston()
# X, y = boston['data'], boston['target']
#
# # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
# clf = LassoCV()
#
# # Set a minimum threshold of 0.25
# sfm = SelectFromModel(clf, threshold=0.25)
# sfm.fit(X, y)
# n_features = sfm.transform(X).shape[1]
#
# # Reset the threshold till the number of features equals two.
# # Note that the attribute can be set directly instead of repeatedly
# # fitting the metatransformer.
# while n_features > 2:
#     sfm.threshold += 0.1
#     X_transform = sfm.transform(X)
#     n_features = X_transform.shape[1]
#
# # Plot the selected two features from X.
# plt.title(
#     "Features selected from Boston using SelectFromModel with "
#     "threshold %0.3f." % sfm.threshold)
# feature1 = X_transform[:, 0]
# feature2 = X_transform[:, 1]
# plt.plot(feature1, feature2, 'r.')
# plt.xlabel("Feature number 1")
# plt.ylabel("Feature number 2")
# plt.ylim([np.min(feature2), np.max(feature2)])
# plt.show()
#
# # L1-based feature selection
# from sklearn.svm import LinearSVC
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# iris = load_iris()
# X, y = iris.data, iris.target
# X.shape
# #(150, 4)
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
# model = SelectFromModel(lsvc, prefit=True)
# X_new = model.transform(X)
# X_new.shape
#
# # Tree-based feature selection
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectFromModel
# iris = load_iris()
# X, y = iris.data, iris.target
# X.shape
# #(150, 4)
# clf = ExtraTreesClassifier()
# clf = clf.fit(X, y)
# clf.feature_importances_
# model = SelectFromModel(clf, prefit=True)
# X_new = model.transform(X)
# X_new.shape
# #(150, 2)
#
#
# # #########################################
# # from sklearn.feature_selection import VarianceThreshold
# #
# # #方差选择法，返回值为特征选择后的数据
# # #参数threshold为方差的阈值
# # VarianceThreshold(threshold=3).fit_transform(iris.data)
# #
# #
# # #########################################
# # from sklearn.feature_selection import SelectKBest
# # from scipy.stats import pearsonr
# #
# # SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
# # #########################################
# # from sklearn.feature_selection import SelectKBest
# # from sklearn.feature_selection import chi2
# #
# # #选择K个最好的特征，返回选择特征后的数据
# # SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
# #
# #
# # #########################################
# # from sklearn.feature_selection import SelectKBest
# # from minepy import MINE
# #
# # #由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
# # def mic(x, y):
# #     m = MINE()
# #     m.compute_score(x, y)
# #     return (m.mic(), 0.5)
# #
# # #选择K个最好的特征，返回特征选择后的数据
# # SelectKBest(lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
# #
# #
# # #########################################
# #########################################
#
# #########################################
#
# #########################################
#
#
#
# #########################################
# from sklearn.decomposition import PCA
#
# # 主成分分析法，返回降维后的数据
# # 参数n_components为主成分数目
# PCA(n_components=2).fit_transform(iris.data)

# --- 例子 ---
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import time

time1 = time.time()

iris_data = load_iris()
category = pd.Categorical(iris_data.target)  # 将标签进行量化，就是说本来都是字符串啊，但是最后计算的时候都需要量化成1，2，3类等

pca_2c = PCA(n_components=2)  # 使用PCA降到2维
# pca_2c = KernelPCA(n_components=2)

x_pca_2c = pca_2c.fit_transform(iris_data.data)
x_pca_2c.shape
plt.scatter(x_pca_2c[:, 0], x_pca_2c[:, 1], c=category.codes)
plt.show()

print(time.time()-time1)
time1 = time.time()
#########################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import manifold
# 线性判别分析法，返回降维后的数据
# 参数n_components为降维后的维数
iris = load_iris()
LDA(n_components=2).fit_transform(iris.data, iris.target)

# --- 例子 ---
# LDA相较于pca是有监督的，但不能用于回归分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris_data = load_iris()
category = pd.Categorical(iris_data.target)  # 将标签进行量化，就是说本来都是字符串啊，但是最后计算的时候都需要量化成1，2，3类等

lda_2c = LDA(n_components=2)
x_pca_2c = lda_2c.fit_transform(iris_data.data, iris_data.target)
x_pca_2c.shape
plt.scatter(x_pca_2c[:, 0], x_pca_2c[:, 1], c=category.codes)
plt.show()

print(time.time()-time1)
time1 = time.time()

iris_data = load_iris()
category = pd.Categorical(iris_data.target)  # 将标签进行量化，就是说本来都是字符串啊，但是最后计算的时候都需要量化成1，2，3类等

tsne = manifold.TSNE(n_components=2,perplexity=10.0)
x_tsne_3c = tsne.fit_transform(iris_data.data, iris_data.target)
plt.scatter(x_tsne_3c[:, 0], x_tsne_3c[:, 1], c=category.codes)
plt.show()

print(time.time()-time1)
time1 = time.time()
#########################################


#########################################


#########################################


#########################################


#########################################


#########################################


#########################################


#########################################


#########################################


#########################################