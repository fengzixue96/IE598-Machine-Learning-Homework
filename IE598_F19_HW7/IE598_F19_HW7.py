import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split,cross_validate
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 7/ccdefault.csv')
df1 = df.drop(['ID','DEFAULT'],axis=1)
X=df.drop(['ID','DEFAULT'],axis=1).values
y=df['DEFAULT'].values
print(X.shape,y.shape)
print(X[0], y[0])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=21)

#random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
mean_train=[]
mean_test=[]
for n in range(1,11):
    start = time.clock()
    forest = RandomForestClassifier(n_estimators=n, random_state=1)
    end = time.clock()
    print('Run time: ', end - start, 's')
    skf = StratifiedKFold(n_splits=10)
    forest.fit(X_train,y_train)
    scores = cross_validate(forest,X,y,cv=skf,scoring='accuracy',return_train_score=True)
    print('the train set score')
    print(scores['train_score'])
    print('the test set score')
    print(scores['test_score'])
    mean_train.append(np.mean(scores['train_score']))
    mean_test.append(np.mean(scores['test_score']))
print('mean of the train set score')
print(mean_train)
print('mean of the test set score')
print(mean_test)
plt.title('10_fold CV Score with different n_estimators')
plt.plot(range(1,11),mean_train)
plt.plot(range(1,11),mean_test)
plt.legend(['train', 'test'])
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.show()
max_n=mean_test.index(max(mean_test)) + 1
print('Optimal n_estimator is:{}'.format(max_n))
print('Optimal accuracy score is:{}'.format(max(mean_test)))
bestforest = RandomForestClassifier(n_estimators=max_n, random_state=1)
bestforest.fit(X_train,y_train)
importances = bestforest.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df1.columns
print("Feature Importances")
for f in range(X_train.shape[1]):
    print(feat_labels[indices[f]], importances[indices[f]])
plt.figure()
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align = 'center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation = 90)
plt.xlim([-1, X_train.shape[1]])

print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
