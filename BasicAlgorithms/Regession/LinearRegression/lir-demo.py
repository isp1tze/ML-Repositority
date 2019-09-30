import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

####################################################
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(np.linspace(axis[0], axis[1], int((axis[1] - axis[0])*100)).reshape(1, -1),
                         np.linspace(axis[2], axis[3], int((axis[3] - axis[2])*100)).reshape(1, -1),)
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

####################################################
# np.random.seed(666)
# x = np.random.normal(0, 1, size=(200, 2))
# y = np.array(x[:,0] ** 2 + x[:,1] ** 2 < 1.5, dtype='int')
np.random.seed(666)
x = np.random.normal(0, 1, size=(200, 2))
y = np.array(x[:,0] ** 2 + x[:,1] < 1.5, dtype='int')
# 添加一些噪音。
for i in range(20):
    y[np.random.randint(200)] = 1

plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()


####################################################
log_reg = LogisticRegression()
log_reg.fit(x, y)
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomiaLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scale', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])

poly_log_reg = PolynomiaLogisticRegression(degree=2)
poly_log_reg.fit(x, y)
poly_log_reg.score(x, y)
# 0.94999999999999996
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

#################################################### 增大degree
poly_log_reg2 = PolynomiaLogisticRegression(degree=20)
poly_log_reg2.fit(x, y)

plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(x, y)
log_reg = LogisticRegression()
log_reg.fit(x, y)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='warn',
#         n_jobs=None, penalty='l2', random_state=None, solver='warn',
#          tol=0.0001, verbose=0, warm_start=False)
# 通过输出结果我们可以发现默认C=1.0，这个C就是最开始提到的逻辑回归中正则中的C，penalty='l2'说明sklearn默认使用L2正则来进行模型正则化。


####################################################
log_reg.score(x_train, y_train)
# 0.7933333333333333
log_reg.score(x_test, y_test)
# 0.7933333333333333
plot_decision_boundary(log_reg, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################
# 使用多项式Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomiaLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scale', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


poly_log_reg = PolynomiaLogisticRegression(degree=2)
poly_log_reg.fit(x_train, y_train)
poly_log_reg.score(x_train, y_train)
# 0.9133333333333333
poly_log_reg.score(x_test, y_test)
# 0.94
plot_decision_boundary(poly_log_reg, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################
poly_log_reg2 = PolynomiaLogisticRegression(degree=20)
poly_log_reg2.fit(x_train, y_train)

poly_log_reg2.score(x_train, y_train)
# 0.94
poly_log_reg2.score(x_test, y_test)
# 0.92
plot_decision_boundary(poly_log_reg2, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()
####################################################
# 使用Logistic Regression L2正则
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# 传入一个新的参数C
def PolynomiaLogisticRegression(degree, C):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scale', StandardScaler()),
        ('log_reg', LogisticRegression(C=C))
    ])

poly_log_reg3 = PolynomiaLogisticRegression(degree=20, C=0.1)
poly_log_reg3.fit(x, y)

poly_log_reg3.score(x_train, y_train)
# 0.8533333333333334
poly_log_reg3.score(x_test, y_test)
# 0.92
plot_decision_boundary(poly_log_reg3, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################
# 使用Logistic Regression L1正则
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def PolynomiaLogisticRegression(degree, C, penalty='l2'):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scale', StandardScaler()),
        ('log_reg', LogisticRegression(C=C, penalty=penalty))
    ])

poly_log_reg4 = PolynomiaLogisticRegression(degree=20, C=0.1, penalty='l1')
poly_log_reg4.fit(x_train, y_train)

poly_log_reg4.score(x_train, y_train)
# 0.8266666666666667
poly_log_reg4.score(x_test, y_test)
# 0.9
plot_decision_boundary(poly_log_reg4, axis=[-4, 4, -4, 4])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

####################################################

