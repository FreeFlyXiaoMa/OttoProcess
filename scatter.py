import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
y_train=np.array([368,340,376,954,331,556])
y_train=np.reshape(y_train,(-1,1))
X_train=np.array([1.7,1.5,1.3,5,1.3,2.2])
X_train=np.reshape(X_train,(-1,1))
lr.fit(X_train,y_train)
print('截距项',lr.intercept_)
print('权重：',lr.coef_[0])
#plt.scatter(X_train,y_train,label='scatter')
#plt.show()DDDDD

###
