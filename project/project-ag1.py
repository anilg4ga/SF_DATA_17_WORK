# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:11:49 2015

@author: anil

Initial data exploration for the 
Forecast use of a city bikeshare system project data provided.

#Submitted by Anil Gawande
"""

import pandas as pd
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## first read the training data to do basic exploration
bikedatatrain = pd.read_csv('C:/Users/anil/ga_ds/SF_DATA_17_WORK/project/train.csv')# Read the training data provided by Kaggle team
bikedatatrain.head()

#Explore the data
bikedatatrain.columns #print name of columns read
bikedatatrain.shape
########################################################
# Show the count of missing values in each column
print "Missing values by columns \n", bikedatatrain.isnull().sum() # There are no missing values
#now maps some of the fields from numermic to literal so the plots are easy to understand.
custom_dict_season = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}  #dictionary for season
bikedatatrain['season_mapped'] = bikedatatrain['season'].map(custom_dict_season)
bikedatatrain.columns
bikedatatrain.head()
bikedatatrain.tail()
#visualization
# plot the data for individual independent variable
# first for the season
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
plt.scatter(bikedatatrain['windspeed'], bikedatatrain['count'])
#high windspeed reduces the bike rental demand