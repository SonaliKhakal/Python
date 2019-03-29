# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:26:51 2018

@author: User
"""


# Multiple Linear Regression using OLS
# Dataset: Sales data
# import the libraries
# --------------------
import pandas as pd
import numpy as np
import math
import pylab #(for graphs.)
import matplotlib.pyplot as plot


# uses Ordinary Least Squares (OLS) method
# -------------------------------------------
import statsmodels.api as sm

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import scipy.stats as stats

import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.svm import SVR
# VIF
# ---
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Feature selection
# -----------------
from sklearn.feature_selection import f_regression as fs

# </import libraries>


# read the input file
# --------------------
path="C:\\Users\\User\\Documents\\Python_project\\salesdata.csv"
sale = pd.read_csv(path)
sale.head()

# count of Rows and Columns
# -------------------------
sale.shape

# describe the dataset (R,C)
# --------------------------
sale.dtypes

# get the record count
# --------------------
sale.count()[1]

# view the dataset
# --------------------
sale.head(5)

# summarize the dataset
# clearer view. removed the 1st row as it contains same info (total records)
# ------------------------------------------------------------
desc = sale.describe()
desc = sale.drop(sale.index[0])  #to remove first index (count) which is not necessary.
desc

# check for NULLS, blanks and zeroes
# -------------------------------
cols = list(sale.columns)
type(cols)
cols.remove("Item_Outlet_Sales")   #we remove y variable.
print(cols)

for c in cols:
    if (len(sale[c][sale[c].isnull()])) > 0:
        print("WARNING: Column '{}' has NULL values".format(c))

    if (len(sale[c][sale[c] == 0])) > 0:
        print("WARNING: Column '{}' has value = 0".format(c))
  
#Drop columns Item_Identifier and Outlet_Identifier
del sale['Item_Identifier']
del sale['Outlet_Identifier']

sale.shape
          
#Gives different levels of particular column       
sale['Outlet_Location_Type'].unique()
sale['Outlet_Type'].unique()

#Replace NULL's in Item_Weight with median
weight_median = sale['Item_Weight'].median()
sale.Item_Weight[sale.Item_Weight.isnull()] = weight_median
sale.head(10)

#Levels in Item_Fat_Content
sale['Item_Fat_Content'].unique() 
       
#Reducing levels of column Item_Fat_Content
sale.Item_Fat_Content[sale.Item_Fat_Content == "low fat"] = "Low Fat"
sale.Item_Fat_Content[sale.Item_Fat_Content == "LF"] = "Low Fat"
sale.Item_Fat_Content[sale.Item_Fat_Content == "reg"] = "Regular"

#Replace "Low Fat" = 1 And "Regular" = 2
sale.Item_Fat_Content[sale.Item_Fat_Content == 'Low Fat'] = 1
sale.Item_Fat_Content[sale.Item_Fat_Content == 'Regular'] = 2

#Convert Item_Fat_Content to Numeric
sale[['Item_Fat_Content']] = sale[['Item_Fat_Content']].apply(pd.to_numeric)

sale.dtypes

#Levels in Item_Type 
sale['Item_Type'].unique()
      
#Replace levels in Item_Type with numbers
sale.Item_Type[sale.Item_Type == "Dairy"] = 1
sale.Item_Type[sale.Item_Type == "Soft Drinks"] = 2
sale.Item_Type[sale.Item_Type == "Meat"] = 3
sale.Item_Type[sale.Item_Type == "Fruits and Vegetables"] = 4
sale.Item_Type[sale.Item_Type == "Household"] = 5
sale.Item_Type[sale.Item_Type == "Baking Goods"] = 6
sale.Item_Type[sale.Item_Type == "Snack Foods"] = 7
sale.Item_Type[sale.Item_Type == "Frozen Foods"] = 8
sale.Item_Type[sale.Item_Type == "Breakfast"] = 9
sale.Item_Type[sale.Item_Type == "Health and Hygiene"] = 10
sale.Item_Type[sale.Item_Type == "Hard Drinks"] = 11
sale.Item_Type[sale.Item_Type == "Canned"] = 12
sale.Item_Type[sale.Item_Type == "Breads"] = 13
sale.Item_Type[sale.Item_Type == "Starchy Foods"] = 14
sale.Item_Type[sale.Item_Type == "Others"] = 15
sale.Item_Type[sale.Item_Type == "Seafood"] = 16

#Convert Item_Type to Numeric
sale[['Item_Type']] = sale[['Item_Type']].apply(pd.to_numeric)
sale.dtypes

#Levels in Outlet_Size
sale['Outlet_Size'].unique()

#Replace 'nan' with 'unknown'
sale.Outlet_Size = sale[['Outlet_Size']].convert_objects(convert_numeric=True).fillna('Unknown')

#Replace levels in Outlet_Size  with numbers
#"High" = 1,"Medium" = 2  "Small" = 3, "Unknown" = 4
sale.Outlet_Size[sale.Outlet_Size == 'High'] = 1
sale.Outlet_Size[sale.Outlet_Size == 'Medium'] = 2
sale.Outlet_Size[sale.Outlet_Size == 'Small'] = 3
sale.Outlet_Size[sale.Outlet_Size == 'Unknown'] = 4

#Convert Outlet_Size to Numeric
sale[['Outlet_Size']] = sale[['Outlet_Size']].apply(pd.to_numeric)
sale.dtypes

#Levels in Outlet_Location_Type
sale['Outlet_Location_Type'].unique()

#Replace levels in Outlet_Location_Type with numbers
# "Tier 1" =1, "Tier 2" =2, "Tier 3" =3
sale.Outlet_Location_Type[sale.Outlet_Location_Type == 'Tier 1'] = 1
sale.Outlet_Location_Type[sale.Outlet_Location_Type == 'Tier 2'] = 2
sale.Outlet_Location_Type[sale.Outlet_Location_Type == 'Tier 3'] = 3

#Convert Outlet_Location_Type to Numeric
sale[['Outlet_Location_Type']] = sale[['Outlet_Location_Type']].apply(pd.to_numeric)
sale.dtypes

#Levels in Outlet_Size
sale['Outlet_Type'].unique()

#Replace levels in Outlet_Type with numbers
# "Supermarket Type1" = 1,"Supermarket Type2"= 2,
# "Supermarket Type3" = 3,"Grocery Store" =4
sale.Outlet_Type[sale.Outlet_Type == 'Supermarket Type1'] = 1
sale.Outlet_Type[sale.Outlet_Type == 'Supermarket Type2'] = 2
sale.Outlet_Type[sale.Outlet_Type == 'Supermarket Type3'] = 3
sale.Outlet_Type[sale.Outlet_Type == 'Grocery Store'] = 4

#Convert Outlet_Type to Numeric
sale[['Outlet_Type']] = sale[['Outlet_Type']].apply(pd.to_numeric)
sale.dtypes

#Gives count of levels of column
sale.groupby('Item_Fat_Content').size()
sale.groupby('Outlet_Size').size()
sale.groupby('Outlet_Location_Type').size()
sale.groupby('Outlet_Type').size()

#Data Visualization
----------------------------------------
# check for outliers in dataset
# --------------------------------------
sale.boxplot(column='Item_Weight')
sale.boxplot(column='Item_Visibility')
sale.boxplot(column='Item_MRP')
sale.boxplot(column='Outlet_Establishment_Year ')

#Histogram for Item_MRP
plt.hist(sale.Item_MRP)
plt.title('Item_MRP')
plt.show()

#Histogram for Item_Weight
plt.hist(sale.Item_Weight)
plt.title('Item_Weight')
plt.show()

#Histogram for Item_Visibility
plt.hist(sale.Item_Visibility)
plt.title('Item_Visibility')
plt.show()

#Histogram for Outlet_Establishment_Year
plt.hist(sale.Outlet_Establishment_Year)
plt.title('Outlet_Establishment_Year')
plt.show()

sales.boxplot('Item_Weight','Outlet_Size',figsize = (5,6))
sales.boxplot('Item_Weight','Item_Type',figsize = (22,8))

#Scatterplot 
sns.pairplot(sale, x_vars = ['Item_Weight', 'Item_Fat_Content','Item_Visibility',
'Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type'],y_vars = 'Item_Outlet_Sales',size = 7, aspect = 0.7)

# to find the correlation among variables (Multicollinearity)
sale.corr()
cor = sale.iloc[:,0:9].corr()
print(cor)
    
# correlation using visualization
# -------------------------------
# cor --> defined above as the correlation amongst the x-variables
sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns)


# split the dataset into train and test
# --------------------------------------
cols = list(sale.columns)
sale.shape

training  = sale[sale["Item_Outlet_Sales"].notnull()][cols]
testing = sale[sale["Item_Outlet_Sales"].isnull() ][cols]

training.shape
testing.shape

# split the training dataset into Train and test
# sample = sample(2, nrow(diab), replace = T, prob = c(0.7,0.3))
training_x ,testing_x = train_test_split(training,test_size=0.3)

training_x.shape
testing_x.shape

# split the training_x and testing_x into X and Y variables
# ------------------------------------------------
train_x = training_x.iloc[:,0:9]; train_y = training_x.iloc[:,9]
test_x  = testing_x.iloc[:,0:9];  test_y = testing_x.iloc[:,9]

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

train_x.head()
train_y.head()

# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
training_x.dtypes    #int and float64
testing_x.dtypes

#count of records
sale.count()

sale.info()

#function -> getresiduals()
# -------------------------------
def getresiduals(lm,train_x,train_y):
    predicted = lm.predict(train_x)
    actual = train_y
    residual = actual-predicted
    
    return(residual)


#OLS Regression Model -1
# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# ----------------------------------------------------------
    
train_x = sm.add_constant(train_x)    #to get values of intercept and slope.
test_x = sm.add_constant(test_x)
lm1 = sm.OLS(train_y, train_x).fit()       #  R-squared : 0.420  &  AIC : 1.026e+05
lm1.summary()



# interpret the result
# =====================
# 1) significant variables: having high |t| or low P values ( < 0.5)
# 2) coefficients = average(coeff(0.025,0.975))
lm1.summary()
# coefficients
lm1.params

### validating the assumptions
# ----------------------------
residuals = getresiduals(lm1,train_x,train_y)
print(residuals)

# 1) Residual mean is 0
# ----------------------------
print(residuals.mean())      # 9.812058283989322e-11

# 2) Residuals have constant variance
# ------------------------------------
y = lm1.predict(train_x)
sns.set(style="whitegrid")
sns.residplot(residuals,y,lowess=True,color="g")

# 3) Residuals are normally distributed
# --------------------------------------
stats.probplot(residuals,dist="norm",plot=pylab)
pylab.show()

# 4) rows > columns
# ------------------
sale.shape

# VIF (Variance Inflation Factor)  ---->check multicollinearity
# -------------------------------  
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(train_x.values, i) for i in range(train_x.shape[1])]
vif["features"] = train_x.columns
print(vif) ---- > variables with high VIF remove any of them
cols


# predict
# -----------------
pdct1 = lm1.predict(test_x)
print(pdct1)

# mean square error
# -----------------
mse = np.mean((pdct1 - test_y)**2)
print("MSE = {0}, RMSE = {1}".format(mse,math.sqrt(mse)))

# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual = list(test_y.head(50))
predicted = np.round(np.array(list(pdct1.head(50))),2)
print(predicted)

df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})
print(df_results)

# feature selection
# ------------------
X1  = train_x.iloc[:,1:10]
features = fs(X1,train_y,center=True)
list(features[0]) # 0 will give you the score. and 1 will give the p-value.
# pd.DataFrame({'column':cols[1:9], 'coefficieint':coefficients})

features = pd.DataFrame({"columns":train_x.columns[1:10], 
                            "score":features[0],
                            "p-val":features[1]
                            })  
print(features)  ----> select variables with high score
 


#Build model -2 with significant variables
# -----------------------------------

sale1=sale.loc[:,['Item_MRP','Outlet_Type','Outlet_Size','Item_Visibility','Outlet_Location_Type','Item_Outlet_Sales']]

# split the dataset into train and test
# --------------------------------------
cols1 = list(sale1.columns)
sale1.shape

training1  = sale1[sale1["Item_Outlet_Sales"].notnull()][cols1]
testing1 = sale1[sale1["Item_Outlet_Sales"].isnull() ][cols1]

training1.shape
testing1.shape

# split the training dataset into Train and test
# sample = sample(2, nrow(diab), replace = T, prob = c(0.7,0.3))
training_x1 ,testing_x1 = train_test_split(training1,test_size=0.3)

training_x1.shape
testing_x1.shape

# split the training_x and testing_x into X and Y variables
# ------------------------------------------------
train_x1 = training_x1.iloc[:,0:5]; train_y1 = training_x1.iloc[:,5]
test_x1  = testing_x1.iloc[:,0:5];  test_y1 = testing_x1.iloc[:,5]

print(train_x1.shape)
print(train_y1.shape)
print(test_x1.shape)
print(test_y1.shape)

# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
training_x1.dtypes    #int and float64
testing_x1.dtypes

#OLS Regression Model - 2
# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# Xn = ccomp,slag,flyash.....
# ----------------------------------------------------------
train_x1 = sm.add_constant(train_x1)    #to get values of intercept and slope.
test_x1 = sm.add_constant(test_x1)
lm2 = sm.OLS(train_y1, train_x1).fit()
lm2.summary()

# predict
# -----------------
pdct2 = lm2.predict(test_x1)
print(pdct2)

# mean square error
# -----------------
mse1 = np.mean((pdct2 - test_y1)**2)
print("MSE = {0}, RMSE = {1}".format(mse1,math.sqrt(mse1)))

# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual1 = list(test_y1.head(50))
predicted1 = np.round(np.array(list(pdct2.head(50))),2)
print(predicted1)

df_results1 = pd.DataFrame({'actual':actual1, 'predicted':predicted1})
print(df_results1)

#Model -3 

#Build model -3 with significant variables
# -----------------------------------
sale2=sale.loc[:,['Item_MRP','Outlet_Type','Outlet_Size','Item_Visibility','Item_Outlet_Sales']]

# split the dataset into train and test
# --------------------------------------
cols2 = list(sale2.columns)
sale2.shape

training2  = sale2[sale2["Item_Outlet_Sales"].notnull()][cols2]
testing2 = sale2[sale2["Item_Outlet_Sales"].isnull() ][cols2]

training2.shape
testing2.shape

# split the training dataset into Train and test
# sample = sample(2, nrow(diab), replace = T, prob = c(0.7,0.3))
training_x2 ,testing_x2 = train_test_split(training2,test_size=0.3)

training_x2.shape
testing_x2.shape

# split the training_x and testing_x into X and Y variables
# ------------------------------------------------
train_x2 = training_x2.iloc[:,0:4]; train_y2 = training_x2.iloc[:,4]
test_x2  = testing_x2.iloc[:,0:4];  test_y2 = testing_x2.iloc[:,4]

# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
training_x2.dtypes    #int and float64
testing_x2.dtypes

#OLS Regression Model - 3
# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# ----------------------------------------------------------
train_x2 = sm.add_constant(train_x2)    #to get values of intercept and slope.
test_x2 = sm.add_constant(test_x2)
lm3 = sm.OLS(train_y2, train_x2).fit()
lm3.summary()

# predict
# -----------------
pdct3 = lm3.predict(test_x2)
print(pdct3)

# mean square error
# -----------------
mse2 = np.mean((pdct3 - test_y2)**2)
print("MSE = {0}, RMSE = {1}".format(mse2,math.sqrt(mse2)))

# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual2 = list(test_y2.head(50))
predicted2 = np.round(np.array(list(pdct3.head(50))),2)
print(predicted)

df_results2 = pd.DataFrame({'actual':actual2, 'predicted':predicted2})
print(df_results2)

#Model - 4 ------> Predict on actual testing set using Model - 2 

# split the dataset into train and test
# --------------------------------------
cols3 = list(sale1.columns)
sale1.shape

training3  = sale1[sale1["Item_Outlet_Sales"].notnull()][cols3]
testing3 = sale1[sale1["Item_Outlet_Sales"].isnull() ][cols3]

# split the training3 and testing3 into X and Y variables
# ------------------------------------------------
train_x3 = training3.iloc[:,0:5]; train_y3 = training3.iloc[:,5]
test_x3  = testing3.iloc[:,0:5];  test_y3 = testing3.iloc[:,5]


# ensure that the X variables are all numeric for regression
# ----------------------------------------------------------
training_x3.dtypes    #int and float64
testing_x3.dtypes

#OLS Regression Model - 4
# To add the constant term A (Y = A + B1X1 + B2X2 + ... + BnXn)
# ----------------------------------------------------------
train_x3 = sm.add_constant(train_x3)    #to get values of intercept and slope.
test_x3 = sm.add_constant(test_x3)
lm4 = sm.OLS(train_y3, train_x3).fit()
lm4.summary()

# predict
# -----------------
pdct4 = lm4.predict(test_x3)
print(pdct4)

# mean square error
# -----------------
mse3 = np.mean((pdct4 - test_y3)**2)
print("MSE = {0}, RMSE = {1}".format(mse3,math.sqrt(mse3)))

# store the actual and predicted values in a dataframe for comparison
# -------------------------------------------------------------------
actual3 = list(test_y3.head(50))
predicted3 = np.round(np.array(list(pdct4.head(50))),2)
print(predicted3)

df_results3 = pd.DataFrame({'actual':actual3, 'predicted':predicted3})
print(df_results3)
