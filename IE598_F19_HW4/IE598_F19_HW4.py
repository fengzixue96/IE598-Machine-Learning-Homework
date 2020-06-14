import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 4/housing2.csv')
columns=df.columns[13:27]

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
for i in range(13,27):
    sys.stdout.write("The summary statistics of " + str(columns[i-13]) + "\n")
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
    plt.xlabel(columns[i-13])
    plt.ylabel('ECDF')
    plt.title("Percentiles ECDF of column "+str(columns[i-13]))
    plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')
    plt.show()

#Quantile‐Quantile Plot for each feature/target column
for i in range(13,27):
    plt.figure()
    stats.probplot(df.iloc[:,i], dist="norm",plot=plt)
    plt.show()

#graphical summary of the relationships
plt.figure()
sns.pairplot(df[columns], size=2.5)
plt.tight_layout()
plt.show()

cols=['ZN','CHAS','RM','DIS','B','MEDV']
plt.figure()
sns.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()

plt.figure()
cm = np.corrcoef(df[columns].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=columns, xticklabels=columns)
plt.show()

#split data
from sklearn.model_selection import train_test_split
X=df.drop('MEDV',axis=1).values
y=df['MEDV'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Common Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg = LinearRegression()
reg.fit(X_train,y_train)
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
for i in range(0,26):
    print('Slope'+str(i)+':'+str(reg.coef_[i]))
print('Intercept: %.3f' % reg.intercept_)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Common Regression')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.figure()
plt.show()
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
print("Root Mean Squared Error: {}".format(rmse))

#Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train,y_train)
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
for i in range(0,26):
    print('Slope'+str(i)+':'+str(ridge.coef_[i]))
print('Intercept: %.3f' % ridge.intercept_)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Ridge Regression')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.figure()
plt.show()
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
print("Root Mean Squared Error: {}".format(rmse))
#best alpha for Ridge
scores=[]
ran=[]
rmse=[]
for alpha in range(1,21):
    ridgeb = Ridge(alpha=alpha)
    ridgeb.fit(X_train, y_train)
    y_train_pred = ridgeb.predict(X_train)
    y_test_pred = ridgeb.predict(X_test)
    scores.append(ridgeb.score(X_test, y_test))
    rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    ran.append(alpha)
plt.figure()
plt.plot(ran,scores)
plt.title('The best alpha via R^2')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
plt.figure()
plt.plot(ran,rmse)
plt.title('The best alpha via Root Mean Squared Error')
plt.xlabel('alpha')
plt.ylabel('Root Mean Squared Error')
plt.show()

#Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
lasso.fit(X_train,y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred = lasso.predict(X_test)
for i in range(0,26):
    print('Slope'+str(i)+':'+str(lasso.coef_[i]))
print('Intercept: %.3f' % lasso.intercept_)
plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Lasso Regression')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.figure()
plt.show()
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
print("Root Mean Squared Error: {}".format(rmse))
#best alpha for Lasso
scores=[]
ran=[]
rmse=[]
for alpha in range(1,21):
    lassob = Lasso(alpha=alpha)
    lassob.fit(X_train, y_train)
    y_train_pred = lassob.predict(X_train)
    y_test_pred = lassob.predict(X_test)
    scores.append(lassob.score(X_test, y_test))
    rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    ran.append(alpha)
plt.figure()
plt.plot(ran,scores)
plt.title('The best alpha via R^2')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
plt.figure()
plt.plot(ran,rmse)
plt.title('The best alpha via Root Mean Squared Error')
plt.xlabel('alpha')
plt.ylabel('Root Mean Squared Error')
plt.show()

print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")