# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:11:49 2015

Project for GA Data Science # 17

#Submitted by Anil Gawande
"""
###########Include the libraries###############################################
import pandas as pd
import numpy as np
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import time
import calendar
import datetime
#Scikit Regression Models
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge 
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn import tree
###############################################################################

## first read the training data to do basic exploration
bikedatatrain = pd.read_csv('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/train.csv')# Read the training data provided by Kaggle team
bikedatatrain.head()
#Explore the data
bikedatatrain.columns #print name of columns read
bikedatatrain.shape
#check the counts for registered, casual and total users.
bikedatatrain['registered'].describe()
bikedatatrain['casual'].describe()
bikedatatrain['count'].describe()
########################################################
# Show the count of missing values in each column
print "Missing values by columns \n", bikedatatrain.isnull().sum() # There are no missing values
##########################################################
def datasetup(inDF): #function to setup the data for analysis
    #create new variables to help model the data
    # the column workingday is 1 if it is not a holiday and not a weekend and 0 otherwise
    # the column holiday is is 1 if it is holiday 0 otherwise
    # looking at the data holiday does not include weekend; therefore adding a new column weekend
    # weekend = 1 if neither holiday and nor workingday, 0 otherwise
    # peak hours differ between regular and casual users
    custom_dict_season = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'} 
    inDF['season_mapped'] = inDF['season'].map(custom_dict_season).astype(str)
    inDF['weekend'] =  np.where(((inDF['holiday'] == 0) & (inDF['workingday'] ==0)), 1, 0)
    # brutual but got the hr of day
    inDF['hrofday']= inDF.datetime.map(str).map(lambda x: x [11:13]).astype(int)
    inDF['hrofday']=inDF['hrofday'].astype(int)
    inDF['dayofmth'] = inDF.datetime.map(str).map(lambda x: x [8:10]).astype(int) 
    inDF['mth'] = inDF.datetime.map(str).map(lambda x: x [5:7]).astype(int)
    inDF['year'] = inDF.datetime.map(str).map(lambda x:x[0:4]).astype(int)
    dt = pd.DatetimeIndex(inDF['datetime'])
    inDF['dayofweek'] = dt.dayofweek +1 
    #peak hours vary between workingday and weekends. see graphs below.
    inDF['peak'] = np.where(((inDF['hrofday'] > 6) & (inDF['hrofday'] < 9) & (inDF['workingday'] == 1)), 1, 0)
    inDF['peak'] = np.where(((inDF['hrofday'] > 15) & (inDF['hrofday'] < 19) & (inDF['workingday'] == 1)), 1, inDF['peak'])
    inDF['peak'] = np.where(((inDF['hrofday'] > 9) & (inDF['hrofday'] < 19) & (inDF['workingday'] == 0)), 1, inDF['peak'])
    #create a rolling month variable
    inDF['rollingmth'] = np.where((inDF['year'].astype(int) == 2012), inDF['mth'].astype(int)+12, inDF['mth'].astype(int))

###########################################################################
datasetup(bikedatatrain) # transform and add new columns
bikedatatrain.columns
#make sure there are no missing values after new columns are added
# Show the count of missing values in each column
print "Missing values by columns \n", bikedatatrain.isnull().sum() # There are no missing values
#visualization#######################################################
# plot the data for individual independent variable. Need to understand how these functions work, can't get labels and data to print together
# first for the season
#plt.xlabel('1-Spring,2-Summer,3-Fall,4-Winter\n   Seasons')
#plt.ylabel('Count of total rentals')
plt.scatter(bikedatatrain['season'], bikedatatrain['count'])
#Surprisingly  there are more rentals in fall and winter?
plt.scatter(bikedatatrain['holiday'], bikedatatrain['count'])
#more rentals on non-holidays
plt.scatter(bikedatatrain['workingday'], bikedatatrain['count'])
#more rentals on working days - seems like rentals are more often used to commute to work and back
plt.scatter(bikedatatrain['weather'], bikedatatrain['count'])
#as expected the rentals drop when weather is  really bad. However  light snow
# or light rain or clouds do affect rentals as much 
plt.scatter(bikedatatrain['temp'], bikedatatrain['count'])
# most rentals between 20 and 30 Celsius  when weather is mild
plt.scatter(bikedatatrain['atemp'], bikedatatrain['count'])
# most rentals between 30 and 40 Celsius  (feels like). Seems like humidity and weather have strong influence on rental
# and not just the temp. -
plt.scatter(bikedatatrain['humidity'], bikedatatrain['count'])
#unless humidity is really low (<20%) or high (>80%) the rental is not
#much impacted by humidity.
plt.scatter(bikedatatrain['hrofday'], bikedatatrain['count'])
#high windspeed reduces the bike rental demand
plt.scatter(bikedatatrain['windspeed'], bikedatatrain['count'])
#check if the demand has been increasing as the word spreads out about the rental service
plt.scatter(bikedatatrain['rollingmth'],bikedatatrain['casual'])
plt.scatter(bikedatatrain['rollingmth'],bikedatatrain['registered'])
#impact of hour of the day whether it is working day or holiday
by_hour = bikedatatrain.copy().groupby(['hrofday', 'workingday'])['count'].agg('sum').unstack()
by_hour.plot(kind='bar', figsize=(8,4), width=0.8, title='All Users');
by_casual = bikedatatrain.copy().groupby(['hrofday', 'workingday'])['casual'].agg('sum').unstack()
by_casual.plot(kind='bar', figsize=(8,4), width=0.8, title='Casual Users');
by_reg = bikedatatrain.copy().groupby(['hrofday', 'workingday'])['registered'].agg('sum').unstack()
by_reg.plot(kind='bar', figsize=(8,4), width=0.8, title='Registered Users');
# scatter plot in Seaborn
# include a "regression line"
sns.pairplot(bikedatatrain, x_vars=['weather', 'atemp', 'humidity', 'windspeed'], y_vars='casual', size=4.5, aspect=0.7)
sns.pairplot(bikedatatrain, x_vars=['weather', 'atemp', 'humidity', 'windspeed'], y_vars='registered', size=4.5, aspect=0.7)
##############################################################################
# define a function that accepts X and y and computes testing RMSE
def run_model(model, X, Y,feature_cols, X_pred,model_type, renter_type): # function copied from class excerises
    # get only the features that we think influence the renter count
    X_in=X[feature_cols] 
    # split into test train set;# using default 75:25 ratio
    X_train, X_test, y_train, y_test = train_test_split(X_in, Y, random_state = 1) 
    # fit the model
    model.fit(X_train, y_train)
    #now run the cross validation to see how good the model is of different subsets.
    scores = cross_val_score(model, X_in, Y, cv=10, scoring='mean_squared_error')
    mse_scores = -scores
    rmse_scores = np.sqrt(mse_scores)
    rmse_scores_mean = rmse_scores.mean()
    print 'For %s model for %s type of renter the RMSE for 10 fold CV is: %f' %(model_type, renter_type, rmse_scores_mean)
    #Now predict the output for the test set
    X_out = X_pred[feature_cols]
    Y_out = model.predict(X_out)
    #Negative values for # of renters does not make sense.
    Y_out[Y_out < 0] = 0 
   # return np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    return(Y_out, rmse_scores_mean)
###########################
#Read the test data
bikedatatest = pd.read_csv('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/test.csv')# Read the training data provided by Kaggle team
bikedatatest.head()
#Explore the data
bikedatatest.columns #print name of columns read
bikedatatest.shape
########################################################
# Show the count of missing values in each column
print "Missing values by columns \n", bikedatatest.isnull().sum() # There are no missing values
datasetup(bikedatatest) # transform and add new columns
bikedatatest.shape
bikedatatest.head
#initialize linear regression model for casual user
linreg = LinearRegression()
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak', 'rollingmth', 'dayofweek', 'weekend']
X = bikedatatrain
Y = bikedatatrain['casual'] 
X.columns
Y.shape
(Y_casual, rmse) = run_model(linreg, X, Y, feature_cols, bikedatatest, "Linear Regression", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual) #make sure that we get the correct number of prediction
#initialize linear regression model for registered user
linreg = LinearRegression()
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak', 'rollingmth', 'dayofweek', 'weekend']
X = bikedatatrain
Y = bikedatatrain['registered'] 
(Y_reg, rmse) = run_model(linreg, X, Y, feature_cols, bikedatatest, "Linear Regression", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg) #make sure that we get the correct number of prediction
Y_total = Y_casual + Y_reg # total # prediction
#write out the prediction if we want to submit
f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_lr.csv', 'w')
for strout in Y_total:
   f.write(str(strout))
f.close
#### Ridge Regression #########################################
ridgereg = Ridge(alpha=.5)
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak']
X = bikedatatrain
Y = bikedatatrain['casual'] 
X
Y.shape
(Y_casual, rmse) = run_model(ridgereg, X, Y, feature_cols, bikedatatest, "Ridge Regression", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual) #make sure that we get the correct number of prediction
#initialize ridge regression model for registered user
ridgereg = Ridge(alpha=.5)
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak']
X = bikedatatrain
Y = bikedatatrain['registered'] 
Y.describe()
(Y_reg, rmse) = run_model(ridgereg, X, Y, feature_cols, bikedatatest, "Ridge Regression", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg) #make sure that we get the correct number of prediction
Y_total = Y_casual + Y_reg # total # prediction
#write out the prediction if we want to submit
f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_rr.csv', 'w')
for strout in Y_total:
   f.write(str(strout))
f.close
#### Support Vector(Linear) Model #############################
svr_lin = SVR(kernel='linear')
feature_cols = [ 'weather', 'temp', 
    'workingday', 'season','holiday', 'hrofday']
X = bikedatatrain
Y = bikedatatrain['casual'] 
X
Y.shape
(Y_casual, rmse) = run_model(svr_lin, X, Y, feature_cols, bikedatatest, "SVR Linear", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual) #make sure that we get the correct number of prediction
#initialize SVR Linear regression model for registered user
svr_lin = SVR(kernel='linear')
feature_cols = [ 'weather', 'temp', 
    'workingday', 'season','holiday', 'hrofday']
X = bikedatatrain
Y = bikedatatrain['registered'] 
(Y_reg, rmse) = run_model(svr_lin, X, Y, feature_cols, bikedatatest, "SVR Linear", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg) #make sure that we get the correct number of prediction
Y_total = Y_casual + Y_reg # total # prediction
#write out the prediction if we want to submit
f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_svr_lin.csv', 'w')
for strout in Y_total:
   f.write(str(strout))
f.close
#### Support Vector(RBF) Model #############################
svr_lin = SVR(kernel='rbf')
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak','rollingmth']
X = bikedatatrain
Y = bikedatatrain['casual'] 
X
Y.shape
(Y_casual, rmse) = run_model(svr_lin, X, Y, feature_cols, bikedatatest, "SVR RBF", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual) #make sure that we get the correct number of prediction
#initialize SVR RBF model for registered user
svr_lin = SVR(kernel='rbf')
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed',
    'workingday', 'season', 'holiday', 'hrofday', 'peak','rollingmth']
X = bikedatatrain
Y = bikedatatrain['registered'] 
(Y_reg, rmse) = run_model(svr_lin, X, Y, feature_cols, bikedatatest, "SVR RBF", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg) #make sure that we get the correct number of prediction
Y_total = Y_casual + Y_reg # total # prediction
#write out the prediction if we want to submit
f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_svr_rbf.csv', 'w')
for strout in Y_total:
   f.write(str(strout))
f.close
#### RandomForest##############################################
#initialize Random Forest Regression model
rforest = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=0, min_samples_split=5, n_jobs=-1)
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed','hrofday','workingday']
X = bikedatatrain[feature_cols]
Y = bikedatatrain['casual'] 
(Y_casual, rmse) = run_model(rforest, X, Y, feature_cols, bikedatatest, "Random Forest", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual)
#now run for registered users
rforest = RandomForestRegressor(n_estimators=1000, max_depth=15, random_state=0, min_samples_split=5, n_jobs=-1)
feature_cols = [ 'weather', 'temp', 'atemp', 'windspeed','hrofday','workingday']
X = bikedatatrain[feature_cols]
Y = bikedatatrain['registered'] 
(Y_reg, rmse) = run_model(rforest, X, Y, feature_cols, bikedatatest, "Random Forest", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg)
Y_total = Y_casual + Y_reg
print Y_total

f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_rf.csv', 'w')
for strout in Y_total:
   wrt = str(strout) + "\n"
   f.write(str(wrt))
f.close()
######ElasticNet #####################################
enet_model = ElasticNet(alpha=0.1, l1_ratio=0.7)
feature_cols = [ 'weather', 'atemp', 'windspeed', 'humidity']
X = bikedatatrain
Y = bikedatatrain['casual'] 
X
Y.shape
(Y_casual, rmse) = run_model(enet_model, X, Y, feature_cols, bikedatatest, "ElasticNet", "Casual")
Y_casual = Y_casual.astype(int)
len(Y_casual) #make sure that we get the correct number of prediction
#initialize elastic model for registered user
enet_model = ElasticNet(alpha=0.1, l1_ratio=0.7)
feature_cols = [ 'weather', 'atemp', 'windspeed','humidity']
X = bikedatatrain
Y = bikedatatrain['registered'] 
(Y_reg, rmse) = run_model(enet_model, X, Y, feature_cols, bikedatatest, "ElasticNet", "Registered")
Y_reg=Y_reg.astype(int)
len(Y_reg) #make sure that we get the correct number of prediction
Y_total = Y_casual + Y_reg # total # prediction
#write out the prediction if we want to submit
f = open('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/testout_enet.csv', 'w')
for strout in Y_total:
   f.write(str(strout))
f.close
#############################################################################