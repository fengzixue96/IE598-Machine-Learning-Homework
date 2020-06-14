import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split,cross_validate
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 6/ccdefault.csv')
X=df.drop(['ID','DEFAULT'],axis=1).values
y=df['DEFAULT'].values
print(X.shape,y.shape)
print(X[0], y[0])


#Decision Tree
#No need to be standardized
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
scores_train=[]
scores_test=[]
tree = DecisionTreeClassifier(criterion='gini',max_depth=20)
start = time.clock()
for n in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
    tree.fit(X_train,y_train)
    y_train_pred1 = tree.predict(X_train)
    y_pred1 = tree.predict(X_test)
    scores_train.append(metrics.accuracy_score(y_train, y_train_pred1))
    scores_test.append(metrics.accuracy_score(y_test, y_pred1))
end = time.clock()
print('Run time: ', end - start, 's')
print(scores_train)
print(scores_test)
print(np.mean(scores_train))
print(np.std(scores_train))
print(np.mean(scores_test))
print(np.std(scores_test))
plt.title('Holdout Score')
plt.plot(range(1,11),scores_train)
plt.plot(range(1,11),scores_test)
plt.legend(['train', 'test'])
plt.xlabel('random_state')
plt.ylabel('Accuracy')
plt.show()

#K_fold CV
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
start = time.clock()
scores = cross_validate(tree,X,y,cv=skf,scoring='accuracy',return_train_score=True)
end = time.clock()
print('Run time: ', end - start, 's')
print(scores.keys())
print(scores['train_score'])
print(np.mean(scores['train_score']))
print(np.std(scores['train_score']))
print(scores['test_score'])
print(np.mean(scores['test_score']))
print(np.std(scores['test_score']))
plt.title('K_fold CV Score')
plt.plot(range(1,11),scores['train_score'])
plt.plot(range(1,11),scores['test_score'])
plt.legend(['train', 'test'])
plt.xlabel('random_state')
plt.ylabel('Accuracy')
plt.show()


print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



