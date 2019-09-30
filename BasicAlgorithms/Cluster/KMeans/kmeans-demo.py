#导入对应的模块
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


#设置全部列显示和浮点数格式
pd.options.display.max_columns=None
pd.set_option('display.float_format',lambda x:'%.6f'%x)

#数据读取
df=pd.read_csv(open(r".\data\customer_data.csv"),encoding='utf-8',index_col=0)
#为了方便分析数据，需要重新定义列名
df=df.rename(columns={'Gender':'gender','Age':'age','Annual Income (k$)':'annual_income','Spending Score (1-100)':'spending_score'})
df.gender.replace(['Male','Female'],[1,0],inplace=True)#用数字表示性别
# print(df.head(2)) # 打印表格的前两行

#计算出数据的均值和标准差
dfms=pd.concat([df.mean().to_frame(),df.std().to_frame()],axis=1).transpose()
dfms.index=['mean','std']
#数据标准化
df_scaled=pd.DataFrame()
for i in df.columns:
    if (i=='gender'): df_scaled[i]=df[i]
    else:
        df_scaled[i]=(df[i] - dfms.loc['mean', i]) / dfms.loc['std', i]
df_scaled.head()


#按照男女划分
dff=df_scaled.loc[df_scaled.gender==0].iloc[:,1:]
dfm=df_scaled.loc[df_scaled.gender==1].iloc[:,1:]

#选择最优的质心点数
def numbers_of_clusters(df):
    wcss=[]
    for i in range(1,20):
        km=KMeans(n_clusters=i,random_state=0,init='k-means++')
        km.fit(df)
        wcss.append(km.inertia_)# inertia簇内误差平方和 用来评估簇的个数是否合适
    df_elbow=pd.DataFrame(wcss)
    df_elbow=df_elbow.reset_index()
    df_elbow.columns=['n_clusters','within_cluster_sum_of_square']
    return df_elbow

#生成质心点
dff_elbow=numbers_of_clusters(dff)
dfm_elbow=numbers_of_clusters(dfm)

plt.subplot(1,2,1)
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['figure.figsize']=(16,10)
matplotlib.rcParams['font.size']=12
plt.plot(dff_elbow.n_clusters,dff_elbow.within_cluster_sum_of_square)
plt.xticks(range(1,19,1))
plt.title('Female')
plt.scatter(x=dff_elbow.n_clusters[5:6],y=dff_elbow.within_cluster_sum_of_square[5:6],color='black',marker='*')

plt.subplot(1,2,2)
matplotlib.rcParams['font.family']='SimHei'
matplotlib.rcParams['figure.figsize']=(16,6)
matplotlib.rcParams['font.size']=12
plt.plot(dfm_elbow.n_clusters,dff_elbow.within_cluster_sum_of_square)
plt.xticks(range(1,19,1))
plt.title('Male')
plt.scatter(x=dfm_elbow.n_clusters[5:6],y=dff_elbow.within_cluster_sum_of_square[5:6],color='black',marker='*')
plt.show()

# n_clusters：整形，缺省值=8 【生成的聚类数，即产生的质心（centroids）数。】
# max_iter：整形，缺省值=300
# 执行一次k-means算法所进行的最大迭代数。
# n_init：整形，缺省值=10
# 用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果。
# init：有三个可选值：’k-means++’， ‘random’，或者传递一个ndarray向量。
# 此参数指定初始化方法，默认值为 ‘k-means++’。
# （１）‘k-means++’ 用一种特殊的方法选定初始质心从而能加速迭代过程的收敛（即上文中的k-means++介绍）
# （２）‘random’ 随机从训练数据中选取初始质心。
# （３）如果传递的是一个ndarray，则应该形如 (n_clusters, n_features) 并给出初始质心。
# precompute_distances：三个可选值，‘auto’，True 或者 False。
# 预计算距离，计算速度更快但占用更多内存。
# （１）‘auto’：如果 样本数乘以聚类数大于 12million 的话则不预计算距离。This corresponds to about 100MB overhead per job using double precision.
# （２）True：总是预先计算距离。
# （３）False：永远不预先计算距离。
# tol：float形，默认值= 1e-4　与inertia结合来确定收敛条件。
# n_jobs：整形数。　指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。
# （１）若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。
# （２）若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。
# random_state：整形或 numpy.RandomState 类型，可选
# 用于初始化质心的生成器（generator）。如果值为一个整数，则确定一个seed。此参数默认值为numpy的随机数生成器。
# copy_x：布尔型，默认值=True
# 当我们precomputing distances时，将数据中心化会得到更准确的结果。如果把此参数值设为True，则原始数据不会被改变。如果是False，则会直接在原始数据
# 上做修改并在函数返回值时将其还原。但是在计算过程中由于有对数据均值的加减运算，所以数据返回后，原始数据和计算前可能会有细小差别。

def k_means(n_clusters,df,gender):
    kmf=KMeans(n_clusters=n_clusters,random_state=0,init='k-means++')
    kmf.fit(df)
    centroids=kmf.cluster_centers_
    cdf=pd.DataFrame(centroids,columns=df.columns)
    cdf['gender']=gender
    cdf['count']=pd.Series(kmf.labels_).value_counts()
    return cdf

df1=k_means(5,dfm,'Male')
df2=k_means(5,dff,'Female')
dfc_scaled=pd.concat([df1,df2],axis=0)

#数据非标准化
dfc=pd.DataFrame()
for i in dfc_scaled.columns:
    if (i=='gender'):dfc[i]=dfc_scaled[i]
    elif (i=='count'):dfc[i]=dfc_scaled[i]
    else:
        dfc[i]=(dfc_scaled[i]*dfms.loc['std',i]+dfms.loc['mean',i])
        dfc[i]=dfc[i].astype(int)

#分类
dfc['type']=1
a_i=dfms.loc['mean']['annual_income']
s_s=dfms.loc['mean']['spending_score']
dfcm=dfc[dfc['gender']=='Male']
dfcf=dfc[dfc['gender']=='Female']
remark=['年长/有孩子的收入一般的潜在男性客户','中年/有孩子的收入较高的优质男客户','年轻的收入一般的潜力男客户','年长/有孩子的收入较低的男客户','中年/有孩子的收入较高的潜在男客户']
dfcm['type']=pd.Series(remark)
remark=['年长/有孩子的收入一般的潜在女性客户','年轻的收入一般的潜力女客户','中年/有孩子的收入较高的优质女客户','年轻的收入较低的可发展女客户','中年/有孩子的收入较高的一般女客户']
dfcf['type']=pd.Series(remark)
print(dfcf)
print(dfcm)
