import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


#1 读取数据
train=pd.read_csv('FE_pima-indians-diabetes.csv')
#print(train.shape)

#2 准备数据
y_trian=train['Target']
X_trian=train.drop(['Target'],axis=1)

feat_names=X_trian.columns
#print(feat_names)

from scipy.sparse import csr_matrix
X_trian=csr_matrix(X_trian)

#3 分割数据
from sklearn.model_selection import train_test_split
X_trian_part,X_val,y_trian_part,y_val=train_test_split(X_trian,y_trian,
                                                       train_size=0.8,
                                                       random_state=0)
#print(X_trian_part.shape)

#4 SVM默认参数调优
from sklearn.svm import LinearSVC
#def fit_grid_point_Linear(C,X_trian,y_trian,X_val,y_val):
#在训练集上训练SVR
#SVC1=LinearSVC()
#模型训练
#SVC1.fit(X_trian_part,y_train_part)

#在校验集上测试，估计模型性能
#y_predict=SVC1.predict(X_val)
#accuracy_scores=accuracy_score(y_val,y_predict)
#print('accuracy score is:',accuracy_scores)
#print('Classification report classifier %s:\n%s\n'%(SVC1,
     #                                                   classification_report(y_val,y_predict)))
#print('Confusion matrix:\n%s'%confusion_matrix(y_val,y_predict))


#线性SVM正则参数调优
'''def fit_grid_point_Linear(C,X_trian,y_trian,X_val,y_val):
    SVC2=LinearSVC(C=C,penalty='l2')
    SVC2.fit(X_trian,y_trian)
    #y_predict=SVC2.predict(X_val)
    #在校验集上返回正确率
    #accuracy=accuracy_score(y_val,y_predict)
    accuracy=SVC2.score(X_val,y_val)
    print("C={}: accuracy={}".format(C,accuracy))
    return accuracy

#需要调优的参数
C_s=np.logspace(-1,3,5)
accuracy_s=[]

for i,oneC in enumerate(C_s):
    tmp=fit_grid_point_Linear(oneC,X_trian_part,y_train_part,X_val,y_val)
    accuracy_s.append(tmp)
X_axis=np.log10(C_s)
plt.plot(X_axis,np.array(accuracy_s),'b-')
plt.legend()
plt.xlabel('log(C)')
plt.ylabel('accuracy')
plt.show()


#最后得到最佳超参数
index=np.argmax(accuracy_s,axis=None)
Best_C=C_s[index]
print(Best_C)
#找到最佳参数后，用全体训练数据

Best_C=0.1
SVC3=LinearSVC(C=Best_C)
SVC3.fit(X_trian,y_trian)
#保存模型，用于后续测试
import pickle as pk
pk.dump(SVC3,open("Pima_indians_LinearSVC.pkl",'wb'))'''


#C_range=np.logspace(-5,5,11)
#param_grid=dict(gamma=gamma_range)
'''for i,value in enumerate(C_range):
    svc.C=value
    grid=GridSearchCV(svc,param_grid=gamma_range,cv=5)
    grid.fit(X_trian_part,y_trian_part)
    #svc.fit(X_trian_part,y_trian_part)
    #grid.score(X_val,y_val)
    y_predict=grid.predict(X_val)

    accuracy=accuracy_score(y_val,y_predict)
    print("C={} accuracy={}".format(value,accuracy))
    print("params={} scores={}".format(grid.best_params_,grid.best_score_))'''

#svc=LinearSVC(C=0.01)
#param_range=[0.001,0.01,0.1,1,10,100,1000]
#param_grid=[{'svc_C':param_range,'svc_kernel':['linear']}]
parameters = {'kernel':['linear'], 'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.0001,0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(SVC(),param_grid=parameters,cv=5)
grid.fit(X_trian_part,y_trian_part)
y_predict=grid.predict(X_val)
accuracy=accuracy_score(y_val,y_predict)
print("accuracy={}".format(accuracy))
print("params={} scores={}".format(grid.best_params_,grid.best_score_))
print('*'*40)
#rbf核
parameters = {'kernel':['rbf'], 'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.0001,0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(SVC(),param_grid=parameters,cv=5)
grid.fit(X_trian_part,y_trian_part)
y_predict=grid.predict(X_val)
accuracy=accuracy_score(y_val,y_predict)
print("accuracy={}".format(accuracy))
print("params={} scores={}".format(grid.best_params_,grid.best_score_))
print('*'*40)

#linear核与rbf核一起调参
parameters = {'kernel':['rbf','linear'], 'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.0001,0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(SVC(),param_grid=parameters,cv=5)
grid.fit(X_trian_part,y_trian_part)
y_predict=grid.predict(X_val)
accuracy=accuracy_score(y_val,y_predict)
print("accuracy={}".format(accuracy))
print("params={} scores={}".format(grid.best_params_,grid.best_score_))