#Listing 2-1: Sizing Up a New Data Set
import numpy as np
import sys
#read data from uci data repository
with open('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 3/HY_Universe_corporate bond.csv') as file:
#arrange data into list for labels and list of lists for attributes
    xList = []
    labels = []
    next(file)
    for line in file:
    #split on comma
        row = line.strip().split(",")
        xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])))
print ("\n")

#Listing 2-2: Determining the Nature of Attributes
nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0]*3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' + "Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
    str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1
print ("\n")

#Listing 2-3: Summary Statistics for Numeric and Categorical Attributes
#generate summary statistics for column 9 (e.g.)
col = 9
colData = []
for row in xList:
    colData.append(float(row[col]))
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' + "Standard Deviation = " + '\t ' + str(colsd) + "\n")
#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")
#The last column contains categorical variables
col = 29
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)
#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*5
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

#Listing 2-4: Quantile‐Quantile Plot for 4th Rocks versus Mines Attribute
import matplotlib.pyplot as plot
import scipy.stats as stats
#generate summary statistics for column 13 (e.g.)
col = 13
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot=plot)
plot.show()

#Listing 2-5: Using Python Pandas to Read and Summarize Data
import pandas as pd
from pandas import DataFrame
#read data into pandas data frame
df = pd.read_csv('C:/Users/HP/Desktop/Lesson/598/Assignment/Assignment 3/HY_Universe_corporate bond.csv')
#print head and tail of data frame
print(df.head())
print(df.tail())
#print summary of data frame
summary = df.describe()
print(summary)

#Listing 2-6: Parallel Coordinates Graph for Real Attribute Visualization
dfdrop = df.drop(['CUSIP','Ticker','Issue Date','Maturity','1st Call Date','Moodys','S_and_P',
              'Fitch','Bloomberg Composite Rating','Maturity Type','Coupon Type','Industry',
              'Months in JNK','Months in HYG','Months in Both','IN_ETF'],axis=1)
print(dfdrop.head())
cols = list(dfdrop)
pcolor = []
for row in range(nrow):
    if dfdrop.iat[row,20] == 1:
        pcolor = "red"
    elif dfdrop.iat[row,20] == 2:
        pcolor = "blue"
    elif dfdrop.iat[row,20] == 3:
        pcolor = "green"
    elif dfdrop.iat[row, 20] == 4:
        pcolor = "yellow"
    elif dfdrop.iat[row, 20] == 5:
        pcolor = "purple"
    # plot rows of data as if they were series data
    dataRow = dfdrop.iloc[row, 0:20]
    dataRow.plot(color=pcolor, alpha=0.5)
plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

#Listing 2-7: Cross Plotting Pairs of Attributes
#calculate correlations between real-valued attributes
dataRow35 = df.iloc[:,35]
dataRow13 = df.iloc[:,13]
plot.scatter(dataRow35, dataRow13)
plot.xlabel("35st Attribute")
plot.ylabel(("13rd Attribute"))
plot.show()
dataRow15 = df.iloc[:,15]
plot.scatter(dataRow35, dataRow15)
plot.xlabel("35st Attribute")
plot.ylabel(("15st Attribute"))
plot.show()

# 2-8 Correlation between Classification Target and Real Attribute
target = []
for row in range(nrow):
    if df.iat[row,29] == 1:
        target.append(1.0)
    elif df.iat[row,29] == 2:
        target.append(2.0)
    elif df.iat[row,29] == 3:
        target.append(3.0)
    elif df.iat[row, 29] == 4:
        target.append(4.0)
    elif df.iat[row, 29] == 5:
        target.append(5.0)
dataRow35 = df.iloc[0:nrow,35]
plot.scatter(dataRow35,target)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()
#To improve the visualization, this version dithers the points a little
# and makes them somewhat transparent
from random import uniform
target = []
for row in range(nrow):
    if df.iat[row,29] == 1:
        target.append(1.0+ uniform(-0.3, 0.3))
    elif df.iat[row,29] == 2:
        target.append(2.0+ uniform(-0.3, 0.3))
    elif df.iat[row,29] == 3:
        target.append(3.0+ uniform(-0.3, 0.3))
    elif df.iat[row, 29] == 4:
        target.append(4.0+ uniform(-0.3, 0.3))
    elif df.iat[row, 29] == 5:
        target.append(5.0+ uniform(-0.3, 0.3))
dataRow35 = df.iloc[0:nrow,35]
plot.scatter(dataRow35,target, alpha=0.5, s=120)
plot.xlabel("Attribute Value")
plot.ylabel("Target Value")
plot.show()

#2-9 Pearson's Correlation Calculation
from math import sqrt
mean13 = 0.0; mean15 = 0.0; mean35 = 0.0
numElt = len(dataRow13)
for i in range(numElt):
    mean13 += dataRow13[i]/numElt
    mean15 += dataRow15[i]/numElt
    mean35 += dataRow35[i]/numElt
var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow13[i] - mean13) * (dataRow13[i] - mean13)/numElt
    var3 += (dataRow15[i] - mean15) * (dataRow15[i] - mean15)/numElt
    var21 += (dataRow35[i] - mean35) * (dataRow35[i] - mean35)/numElt
corr1315 = 0.0; corr1335 = 0.0
for i in range(numElt):
    corr1315 += (dataRow13[i] - mean13) * \
              (dataRow15[i] - mean15) / (sqrt(var2*var3) * numElt)
    corr1335 += (dataRow13[i] - mean13) * \
               (dataRow35[i] - mean35) / (sqrt(var2*var21) * numElt)
sys.stdout.write("Correlation between attribute 13 and 15 \n")
print(corr1315)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between attribute 13 and 35 \n")
print(corr1335)
sys.stdout.write(" \n")

#2-10 Presenting Attribute Correlations Visually
#calculate correlations between real-valued attributes
corMat = DataFrame(df.corr())
#visualize correlations using heatmap
plot.pcolor(corMat)
plot.show()

print("My name is Shixue Feng")
print("My NetID is: shixuef2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")