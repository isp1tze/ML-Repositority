# coding:utf-8
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# 注意导入包的位置
from sklearn.datasets.samples_generator import make_blobs

# 生成样本点
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# 通过下列代码可自动检测bandwidth值
# 用于估计加权核的带宽，n_samples参数指定用于估计的样本数，quantile指定至少
# 被使用的指定数量样本数的分位数。（取值与[0, 1]）
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# bin_seeding用来设定初始核的位置参数的生成方式，default False,默认采用所有点的
# 位置平均，当改为True时使用离散后的点的平均，前者比后者慢。

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
new_X = np.column_stack((X, labels))

print("number of estimated clusters : %d" % n_clusters_)
print("Top 10 samples:", new_X[:10])

# 图像输出
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()