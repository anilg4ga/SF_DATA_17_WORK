'''
Move this code into your OWN SF_DAT_15_WORK repo

Please complete each question using 100% python code

If you have any questions, ask a peer or one of the instructors!

When you are done, add, commit, and push up to your repo

This is due 9/30/2015

Submitted by Anil Gawande
'''

import pandas as pd
# pd.set_option('max_colwidth', 50)
# set this if you need to

killings = pd.read_csv('hw/data/police-killings.csv')# anilg - changed the folder where the data file is located.
killings.head()

# 1. Make the following changed to column names:
# lawenforcementagency -> agency
# raceethnicity        -> race
killings.columns #print name of columns read
killings.rename(columns={'lawenforcementagency':'agency', 'raceethnicity':'race'}, inplace=True)
killings.columns # confirm the column names have changed.
########################################################
# 2. Show the count of missing values in each column
print "Missing values by columns \n", killings.isnull().sum()
########################################################
# 3. replace each null value in the dataframe with the string "Unknown"
killings.fillna(value='Unknown', inplace=True) #make sure to use inplace parameter
print "Missing values by columns \n", killings.isnull().sum() #confirm no missing values
########################################################
# 4. How many killings were there so far in 2015?
print "Killings so far in 2015: ", killings.shape[0]
########################################################
# 5. Of all killings, how many were male and how many female?
gender = killings['gender'].value_counts()
print "The count of people killed by gender is", "\n", gender
########################################################
# 6. How many killings were of unarmed people?
armed = killings['armed'].value_counts() # first get the counts for different armed status
unarmed = armed['No'] #get the count where the armed status is No. Not considering unknown or other
print "Number of killings of unarmed people: ", unarmed
########################################################
# 7. What percentage of all killings were unarmed?
#Use the count of rows with armed status == 'No' and divide by total number of rows
percentunarmed = (float(unarmed)/float(killings.armed.count()))*100
print "Percent of all killings that were unarmed: ", percentunarmed, "%"
########################################################
# 8. What are the 5 states with the most killings?
top5statecount = killings['state'].value_counts(sort=True).head(5)
print "The top 5 states with most killings \n", top5statecount
########################################################
# 9. Show a value counts of deaths for each race
race_deaths = killings['race'].value_counts()
print "The value counts of death for each race: ","\n", race_deaths
########################################################
# 10. Display a histogram of ages of all killings
killings.age.hist()
########################################################
# 11. Show 6 histograms of ages by race
killings.age.hist(by=killings.race, sharex=True, sharey=True)
########################################################
# 12. What is the average age of death by race?
print "The average age at death by race is \n", killings.groupby('race').age.mean()
########################################################
# 13. Show a bar chart with counts of deaths every month
custom_dict = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6}  #to sort data by month
killings['monthsort'] = killings['month'].map(custom_dict)
killings.sort(columns=['monthsort'], inplace=True)
mthcount = killings['monthsort'].value_counts(sort=False)
mthcount
mthcount.index= ['January', 'February', 'March', 'April', 'May', 'June']
mthcount
mthcount.plot(kind='bar', title='No of Police Killings by Month')
#########################################################
###################
### Less Morbid ###
###################

majors = pd.read_csv('hw/data/college-majors.csv') #changed the folder where the data file is located.
majors.head()

# 1. Delete the columns (employed_full_time_year_round, major_code)
majors.drop(['Employed_full_time_year_round'], axis=1, inplace=True) #complete removal
majors.drop(['Major_code'], axis=1, inplace=True)
majors.columns
#########################################################
# 2. Show the cout of missing values in each column
print "Missing values by columns \n", majors.isnull().sum()
#########################################################
# 3. What are the top 10 highest paying majors?
#listing the top 10 majors with largest average
paysort = majors.sort_index(by='Median', ascending=False).head(10)
print "Top 10 highest paying majors\n", paysort['Major'].head(10).values
#########################################################
# 4. Plot the data from the last question in a bar chart, include proper title, and labels!
paysort.plot(kind='bar', title='Top 10 highest paying major',x='Major' , y='Median', legend=True)
#########################################################

# 5. What is the average median salary for each major category?
avg_cat = majors.groupby('Major_category').Median.mean()
print "The average median salary for each major category is \n", avg_cat
#########################################################
# 6. Show only the top 5 paying major categories
maj_cat =  majors.groupby('Major_category').Median.mean()
maj_cat
maj_cat.sort(ascending=False, inplace=True)
print 'The top 5 paying major categories are\n', maj_cat.head(5)
# 7. Plot a histogram of the distribution of median salaries
majors.Median.hist()
#########################################################
# 8. Plot a histogram of the distribution of median salaries by major category
maj_cat.hist()
#########################################################
# 9. What are the top 10 most UNemployed majors?
# What are the unemployment rates?
unemp_majors = majors.sort_index(by='Unemployment_rate', ascending=False)
print 'Top 10 most Unemployed majors:\n', unemp_majors.head(10).Major,  "\n", unemp_majors.head(10).Unemployment_rate
print 'Top 10 most Unemployed rate:\n', unemp_majors.head(10).Major
print 'Top 10 most Unemployed rate:\n', unemp_majors.head(10).Unemployment_rate
#########################################################
# 10. What are the top 10 most UNemployed majors CATEGORIES? Use the mean for each category
# What are the unemployment rates?
unemp_cat = majors.groupby('Major_category').Unemployment_rate.mean()
unemp_cat.sort(ascending=False, inplace=True)
print 'Top 10 most Unemployed major category and rate:\n', unemp_cat.head(10)
###########################################################################
# 11. the total and employed column refer to the people that were surveyed.
# Create a new column showing the emlpoyment rate of the people surveyed for each major
# call it "sample_employment_rate"
# Example the first row has total: 128148 and employed: 90245. it's 
# sample_employment_rate should be 90245.0 / 128148.0 = .7042
majors['sample_employment_rate'] = majors['Employed']/majors['Total']
majors['sample_employment_rate']
#########################################################
# 12. Create a "sample_unemployment_rate" colun
# this column should be 1 - "sample_employment_rate"
majors['sample_unemployment_rate'] = 1- majors['sample_employment_rate']
majors['sample_unemployment_rate']
########################################################