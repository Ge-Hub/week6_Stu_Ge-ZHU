"""
男女声音识别
Step1，数据加载
Step2，数据预处理
分离特征X 和Target y
使用标签编码，male -> 1, female -> 0
将特征X矩阵进行规范化
#标准差标准化，处理后的数据符合标准正态分布
scaler = StandardScaler()
Step3，数据集切分，train_test_split
Step4，模型训练
SVM，Linear SVM, RandomForestClassifier
Step5，模型预测
"""


# Step1: data loading
import pandas as pd
import numpy as np
df = pd.read_csv('voice.csv')
    #data exploring
    #display.max_columns: Using set_option(), we can change the default number of rows to be displayed. https://www.tutorialspoint.com/python_pandas/python_pandas_options_and_customization.htm
pd.set_option('display.max_columns',10)
#print(df)
#print(df.head)
#print(df.shape)
#print(df.isnull().sum())
print('sample size:{}'.format(df.shape[0]))
print('sample size:{}'.format(df[df.label == 'male'].shape[0]))
print('sample size:{}'.format(df[df.label == 'female'].shape[0]))

# separated eigenvalues and label
    #:-1 without last colums / -1 includ. last colums
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Step2: to identify type of X via label coding
from sklearn.preprocessing import LabelEncoder

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
print(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step3: data spliting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state=100)


# Step4: Model training 

## RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

rf=RandomForestClassifier(n_estimators=80,max_features=1,random_state=0)
# 从输出的准确率可以看出，当限制树高后，准确率下降，整体的泛化误差上升， 说明此时，提高模型准确率的方法，只能是对 max_features 进行调整。
# 因为 max_depth,min_samples_leaf 以及 min_sample_split 均为剪枝参数，是减小复杂度的参数。而此时我们需要增加模型的复杂度。max_features 既可以让模型复杂，也可以让模型简单。
#X, y = make_classification(n_samples=3168, n_features=21, n_informative=1, n_redundant=0, random_state=3)

rf.fit(X_train,y_train) #训练模型
y_pred = rf.predict(X_test)
#print(y_pred) #输出训练集上的预测结果
#print('RFC accurecy_score_train:',rf.score(X_train,y_train)) #输出训练集上的准确率
print('RFC accurecy_score_test:',rf.score(X_test,y_test)) #输出验证集上的准确率
#df.to_csv('submit_voice_.csv')

print('-'*118)

## SVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
#print('SVM predicted results:', y_pred)
#print('SVM predicted accurecy:', accuracy_score(y_test, y_pred))
#print('SVM accurecy_score_train:',rf.score(X_train,y_train)) 
print('SVM accurecy_score_test:', svc.score(X_test,y_test))

print('-'*118)

## Linear SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
#print('Linear SVM predicted results:', y_pred)
#print('Linear SVM predicted accurecy:', accuracy_score(y_test,y_pred))
#print('Linear SM accurecy_score_train:',rf.score(X_train,y_train)) 
print('Linear SVM accurecy_score_test:', svc.score(X_test,y_test))
