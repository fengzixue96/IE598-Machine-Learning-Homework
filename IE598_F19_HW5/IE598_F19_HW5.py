import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 5/hw5.csv')
df = df.drop(['Date'],axis=1)
columns=df.columns[0:31]

#eliminate the missing data
df=df.dropna()

#Read and Summarize Data
print(df.shape)
row=df.shape[0]
column=df.shape[1]
print(df.head())
print(df.tail())
summary = df.describe()
print(summary)

#summary statistics for each feature/target column
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y
for i in range(0,6):
    sys.stdout.write("The summary statistics of " + str(columns[i]) + "\n")
    mean = np.mean(df.iloc[:,i])
    std = np.std(df.iloc[:,i])
    sys.stdout.write("Mean = " + '\t' + str(mean) + '\t\t' + "Standard Deviation = " + '\t ' + str(std) + "\n")
    percentiles = np.array([2.5, 25, 50, 75, 97.5])
    ptiles_vers = np.percentile(df.iloc[:,i], percentiles)
    sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
    print(ptiles_vers)
    sys.stdout.write(" \n")
    plt.figure()
    x_vers, y_vers=ecdf(df.iloc[:,i])
    plt.plot(x_vers, y_vers, '.')
    plt.xlabel(columns[i])
    plt.ylabel('ECDF')
    plt.title("Percentiles ECDF of column "+str(columns[i]))
    plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')
    plt.show()
sys.stdout.write("The summary statistics of 'Adj_Close'" + "\n")
mean = np.mean(df['Adj_Close'])
std = np.std(df['Adj_Close'])
sys.stdout.write("Mean = " + '\t' + str(mean) + '\t\t' + "Standard Deviation = " + '\t ' + str(std) + "\n")
percentiles = np.array([2.5, 25, 50, 75, 97.5])
ptiles_vers = np.percentile(df['Adj_Close'], percentiles)
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(ptiles_vers)
sys.stdout.write(" \n")
plt.figure()
x_vers, y_vers=ecdf(df['Adj_Close'])
plt.plot(x_vers, y_vers, '.')
plt.xlabel('Adj_Close')
plt.ylabel('ECDF')
plt.title("Percentiles ECDF of column 'Adj_Close'")
plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')
plt.show()

#Quantile‐Quantile Plot for each feature/target column
for i in range(0,6):
    plt.figure()
    stats.probplot(df.iloc[:,i], dist="norm",plot=plt)
    plt.show()
plt.figure()
stats.probplot(df['Adj_Close'], dist="norm",plot=plt)
plt.show()

#graphical summary of the relationships
cols=['SVENF01','SVENF02','SVENF03','SVENF04','SVENF05','Adj_Close']
plt.figure()
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

#split data
from sklearn.model_selection import train_test_split
X=df.drop('Adj_Close',axis=1).values
y=df['Adj_Close'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=42)

#PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
cov_mat = np.cov(X_test.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
print('Explained variance ratio: ', pca.explained_variance_ratio_)
print('Explained variance: ', pca.explained_variance_)
cum_var_exp = np.cumsum(var_exp)
plt.figure()
plt.bar(range(1,31),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,31),cum_var_exp,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

pca = PCA(n_components=3)
pca.fit(X)
cov_mat = np.cov(X_test.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
print('Explained variance ratio of the 3-component version: ', pca.explained_variance_ratio_)
print('Explained variance of the 3-component version: ', pca.explained_variance_)
cum_var_exp = np.cumsum(var_exp)
cum_var=np.cumsum(pca.explained_variance_)
print('Cumulative explained variance ratio of the 3-component version: ', cum_var_exp[2])
print('Cumulative explained variance of the 3-component version: ', cum_var[2])
plt.figure()
plt.bar(range(1,4),var_exp[0:3],alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,4),cum_var_exp[0:3],where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

X_pca = pca.transform(X)
X_train_pca,X_test_pca,y_train_pca,y_test_pca=train_test_split(X_pca,y,test_size=0.15,random_state=42)

#set function of CV accuracy score
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
def train_and_evaluate(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    print("Accuracy R2 score on training set:",clf.score(X_train, y_train))
    print("Accuracy R2 score on testing set:", clf.score(X_test, y_test))
    # create a k-fold cross validation iterator of k=5 folds
    cv = KFold(5, shuffle=True, random_state=33)
    scores_train = cross_val_score(clf, X_train, y_train, cv=cv)
    scores_test = cross_val_score(clf, X_test, y_test, cv=cv)
    print("Average accuracy R2 score on training set using 5-fold cross-validation:",np.mean(scores_train))
    print("Average accuracy R2 score on testing set using 5-fold cross-validation:",np.mean(scores_test))

#Linear Regression without PCA
from sklearn.linear_model import LinearRegression
clf_sgd = LinearRegression()
train_and_evaluate(clf_sgd,X_train,y_train)
y_train_pred1 = clf_sgd.predict(X_train)
y_test_pred1 = clf_sgd.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_train,y_train_pred1))
print("Root Mean Squared Error of Train Set: {}".format(rmse))
rmse = np.sqrt(mean_squared_error(y_test,y_test_pred1))
print("Root Mean Squared Error of Test Set: {}".format(rmse))
print('Slope:')
print(clf_sgd.coef_)
print('Intercept: %.3f' % clf_sgd.intercept_)
plt.figure()
plt.scatter(y_train_pred1, y_train_pred1 - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred1, y_test_pred1 - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Linear Regression without PCA')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#SVM without PCA
from sklearn import svm
clf_svr = svm.SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)
y_train_pred2 = clf_svr.predict(X_train)
y_test_pred2 = clf_svr.predict(X_test)
rmse_SVM1 = np.sqrt(mean_squared_error(y_train,y_train_pred2))
print("Root Mean Squared Error of Train Set: {}".format(rmse_SVM1))
rmse_SVM2 = np.sqrt(mean_squared_error(y_test,y_test_pred2))
print("Root Mean Squared Error of Test Set: {}".format(rmse_SVM2))
print('Slope:')
print(clf_svr.coef_)
print('Intercept: %.3f' % clf_svr.intercept_)
plt.figure()
plt.scatter(y_train_pred2, y_train_pred2 - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred2, y_test_pred2 - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('SVR without PCA')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#Linear Regression with PCA
clf_sgd_pca = LinearRegression()
clf_sgd_pca.fit(X_train_pca,y_train_pca)
print("Accuracy R2 score on training set:",clf_sgd_pca.score(X_train_pca, y_train_pca))
print("Accuracy R2 score on testing set:", clf_sgd_pca.score(X_test_pca, y_test_pca))
# create a k-fold cross validation iterator of k=5 folds
cv = KFold(5, shuffle=True, random_state=33)
scores_train1 = cross_val_score(clf_sgd_pca, X_train_pca, y_train_pca, cv=cv)
scores_test1 = cross_val_score(clf_sgd_pca, X_test_pca, y_test_pca, cv=cv)
print("Average accuracy R2 score on training set using 5-fold cross-validation:",np.mean(scores_train1))
print("Average accuracy R2 score on testing set using 5-fold cross-validation:",np.mean(scores_test1))
y_train_pred_pca1 = clf_sgd_pca.predict(X_train_pca)
y_test_pred_pca1 = clf_sgd_pca.predict(X_test_pca)
rmse_li_pca1 = np.sqrt(mean_squared_error(y_train_pca,y_train_pred_pca1))
print("Root Mean Squared Error of Train Set: {}".format(rmse_li_pca1))
rmse_li_pca2 = np.sqrt(mean_squared_error(y_test_pca,y_test_pred_pca1))
print("Root Mean Squared Error of Test Set: {}".format(rmse_li_pca2))
print('Slope:')
print(clf_sgd_pca.coef_)
print('Intercept: %.3f' % clf_sgd_pca.intercept_)
plt.figure()
plt.scatter(y_train_pred_pca1, y_train_pred_pca1 - y_train_pca, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred_pca1, y_test_pred_pca1 - y_test_pca, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Linear Regression with PCA')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

#SVM with PCA
clf_svr_pca = svm.SVR(kernel='linear')
clf_svr_pca.fit(X_train_pca,y_train_pca)
print("Accuracy R2 score on training set:",clf_svr_pca.score(X_train_pca, y_train_pca))
print("Accuracy R2 score on testing set:", clf_svr_pca.score(X_test_pca, y_test_pca))
# create a k-fold cross validation iterator of k=5 folds
cv = KFold(5, shuffle=True, random_state=33)
scores_train2 = cross_val_score(clf_svr_pca, X_train_pca, y_train_pca, cv=cv)
scores_test2 = cross_val_score(clf_svr_pca, X_test_pca, y_test_pca, cv=cv)
print("Average accuracy R2 score on training set using 5-fold cross-validation:",np.mean(scores_train2))
print("Average accuracy R2 score on testing set using 5-fold cross-validation:",np.mean(scores_test2))
y_train_pred_pca2 = clf_svr_pca.predict(X_train_pca)
y_test_pred_pca2 = clf_svr_pca.predict(X_test_pca)
rmse_SVM_pca1 = np.sqrt(mean_squared_error(y_train_pca,y_train_pred_pca2))
print("Root Mean Squared Error of Train Set: {}".format(rmse_SVM_pca1))
rmse_SVM_pca2 = np.sqrt(mean_squared_error(y_test_pca,y_test_pred_pca2))
print("Root Mean Squared Error of Test Set: {}".format(rmse_SVM_pca2))
print('Slope:')
print(clf_svr_pca.coef_)
print('Intercept: %.3f' % clf_svr_pca.intercept_)
plt.figure()
plt.scatter(y_train_pred_pca2, y_train_pred_pca2 - y_train_pca, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred_pca2, y_test_pred_pca2 - y_test_pca, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('SVR with PCA')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
