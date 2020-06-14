import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 2/Treasury Squeeze test - DS1.csv')
X=df.drop(['rowindex','contract','squeeze'],axis=1).values
y=df['squeeze'].values
print(X.shape,y.shape)
print(X[0], y[0])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)

#Decision Tree
#No need to be standardized
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=20,random_state=1)
tree.fit(X_train,y_train)
from sklearn import metrics
y_train_pred1 = tree.predict(X_train)
y_pred1 = tree.predict(X_test)
print('\nDecision Tree')
print('The accuracy score of its training set is:{}'.format(metrics.accuracy_score(y_train, y_train_pred1)))
print('The accuracy score of its testing set is:{}'.format(metrics.accuracy_score(y_test, y_pred1)))
target_names = ['TRUE', 'FALSE']
print('Its classification report is:')
print(metrics.classification_report(y_test, y_pred1, target_names=target_names))
print('Its confusion matrix is:')
print(metrics.confusion_matrix(y_test, y_pred1))

#Decision tree splitting rules
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree,filled=True,
                           rounded=True,
                           class_names=['TRUE','FALSE'],
                           feature_names=['price_crossing','price_distortion','roll_start',
                                          'roll_heart','near_minus_next','ctd_last_first',
                                          'ctd1_percent','delivery_cost','delivery_ratio'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

#Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
y_train_pred2 = forest.predict(X_train)
y_pred2 = forest.predict(X_test)
print('\nRandom Forest')
print('The accuracy score of its training set is:{}'.format(metrics.accuracy_score(y_train, y_train_pred2)))
print('The accuracy score of its testing set is:{}'.format(metrics.accuracy_score(y_test, y_pred2)))
target_names = ['TRUE', 'FALSE']
print('Its classification report is:')
print(metrics.classification_report(y_test, y_pred2, target_names=target_names))
print('Its confusion matrix is:')
print(metrics.confusion_matrix(y_test, y_pred2))

#Standardization for KNN
scaler = preprocessing.StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test= scaler.transform(X_test)

#the best number of K-nearest neighbors
#by for loop
from sklearn.neighbors import KNeighborsClassifier
k_range=range(1,26)
scores=[]
range=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred3=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred3))
    range.append(k)
print('\nThe best k by for loop')
print('Optimal k is:{}'.format(scores.index(max(scores))+1))
print('Optimal accuracy score is:{}'.format(max(scores)))
plt.plot(range,scores)
plt.title('The best k')
plt.xlabel('k')
plt.ylabel('score')
plt.show()

#by GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors': np.arange(1,26)}
knn = KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid)
knn_cv.fit(X_train, y_train)
print('\nThe best k by GridSearchCV')
print('Optimal k is:{}'.format(knn_cv.best_params_))
print('Optimal accuracy score is:{}'.format(knn_cv.best_score_))

print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



