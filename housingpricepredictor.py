
# import all import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HousingPrices.csv') # save the data in a dataframe
df

# view the top 5 rows of the dataframe
df.head()

# Shows all info for number of rows and number of columns and the types of data.
df.info()

# the bottom 5 records
df.tail()

df.shape # shape is an attribute for all dataframes

df.isnull().sum()
# we don't have any missing data so its okay!

# calculate the percentage of missing data per column
df.isnull().sum()/df.shape[0]*100
# usually if the percentage of missing data is more than 50% for a certain
# column - we delete it

# gives the count for duplicated values
df.duplicated().sum()

# we did this to check for any garbage values
for i in df.select_dtypes(include='object').columns:
    print(df[i].value_counts())
    print('***'*10)

# .T is for transposing
# df.describe() without any parameters is automatically for numerical data
df.describe().T

# df.describe() with a parameter 'include' is for categorical data (since
# include = 'object)
df.describe(include='object').T

# this is for histogram
for i in df.select_dtypes(include='number').columns:
    sns.histplot(df[i])
    plt.show()

for i in df.select_dtypes(include='number').columns:
    sns.boxplot(data=df, x=i) # x = i means I want my col to be my x-axis
    # this creates a horizontal boxplot as shown
    plt.show()

# dependent variable - house price
# independent variable - all other factors
# create scatter plots to check for relationship between variables
for i in ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
    sns.scatterplot(data=df, x=i, y='price')
    plt.show()

# Only do outlier treatment for continuous numerical data (no categorical data)
# only price and area are continuous, everything else is discrete

def whisker(col):
  q1 = np.percentile(col, 25)
  q3 = np.percentile(col, 75)
  iqr = q3-q1
  lw = q1 - .5*iqr
  uw = q3 + 1.5*iqr
  return lw, uw

whisker(df['price'])

for i in ['price', 'area']:
  lw, uw = whisker(df[i])
  df[i] = np.where(df[i]>uw, uw, df[i])
  df[i] = np.where(df[i]<lw, lw, df[i])

for i in ['price', 'area']:
  sns.boxplot(data=df, x=i)
  plt.show()

  # notice how no more outliers, the data has been changed, outliers have been
# treated

# encoding data - changing object data to numerical data

# time to split data into train and test
from sklearn.model_selection import train_test_split

train, test = train_test_split(df[['area', 'price']])

train

test

train.shape

test.shape

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

predictors = ['area']
target = "price"

reg.fit(train[predictors], train['price'])

predictions = reg.predict(test[predictors])

predictions

test

from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test['price'], predictions)

df.describe()['price']
# error should be below standard deviation -

1181211.3557141407 < 1731752

test.shape

test['predictions'] = predictions

test

errors = test['price'] - test['predictions']
errors
