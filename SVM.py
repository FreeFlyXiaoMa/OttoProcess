import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.core.pylabtools import figsize
from sklearn.metrics import accuracy_score, confusion_matrix

train1=pd.read_csv('Otto_FE_train_org.csv')
train2=pd.read_csv('Otto_FE_train_tfidf.csv')

#print(train1.head())
#print(train2.head())
train2=train2.drop(['id','target'],axis=1)
train=pd.concat([train1,train2],axis=1,ignore_index=False)
#print(train.head())

y_train=train['target']
X_train=train.drop(['id','target'],axis=1)

feat_names=X_train.columns
from scipy.sparse import csr_matrix
X_train=csr_matrix(X_train)

from sklearn.model_selection import train_test_split
X_train_part,X_val,y_train_part,y_val=train_test_split(X_train,y_train,
                                                       test_size=0.2,
                                                    random_state=0)
#print(X_train_part.shape)
from sklearn.svm import LinearSVC
SVC1=LinearSVC()
SVC1.fit(X_train_part,y_train_part)
y_predict=SVC1.predict(X_val)

#print("accuracy is:",accuracy_score(y_val,y_predict))
#print('Confusion matrix:\n%s'%confusion_matrix(y_val,y_predict))
def fit_grid_point_Linear(C,X_train,y_train,X_val,y_val):
    SVC2=LinearSVC()
    SVC2=SVC2.fit(X_train_part,y_train_part)
    accuracy=SVC2.score(X_val,y_val)
    print("C={}: accuracy={}".format(C,accuracy))
    return accuracy
C_s=np.logspace(-1,3,5)
accuracy_s=[]
for i,oneC in enumerate(C_s):
    tmp=fit_grid_point_Linear(oneC,X_train_part,X_val,y_train_part,y_val)
    accuracy_s.append(tmp)
print(accuracy_s)
#X_axis=np.log10(C_s)
#plt.plot(X_axis,np.array(accuracy_s),'b--')
#plt.xlabel('log(C)')
#plt.ylabel('accuracy')
#plt.show()
