#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


# In[65]:


df_train = pd.read_csv('/Users/derya_ak/Desktop/walmartdataset/train.csv')
df_features = pd.read_csv('/Users/derya_ak/Desktop/walmartdataset/features.csv')
df_stores = pd.read_csv('/Users/derya_ak/Desktop/walmartdataset/stores.csv')


# In[66]:


df_train.head()


# In[67]:


df_features.head()


# In[68]:


df_stores.head()


# In[69]:


# check dataframe shape

df_features.shape, df_train.shape, df_stores.shape


# In[70]:


#merge 3 different datasets into one
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner')              .merge(df_stores, on=['Store'], how='inner')
df.head()


# In[71]:


# removing dublicated column
df.drop(['IsHoliday_y'], axis=1,inplace=True) 
# rename the column
df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True)


# In[147]:


df.size


# In[73]:


# Check out the number of unique values
df.nunique()


# #there are 45 stores and 99 department numbers but some of department numbers are missing.

# In[74]:


#find out which week sales are negative
df.query("Weekly_Sales < 0")


# In[75]:


#find the average of weekly_sales by department for each store
sales_grouped_dept = df.groupby(['Store','Dept']).agg({'Weekly_Sales' : 'mean'})

# Check out negative values
sales_grouped_dept.query("Weekly_Sales < 0")


# #I assume that sales vales cannot be negative. Therefore, negative values will be replaced with NaN

# In[76]:


df.loc[df['Weekly_Sales'] < 0, 'Weekly_Sales'] = np.nan
df[df["Weekly_Sales"] < 0]["Weekly_Sales"] 


# In[77]:


# Check out the missing values
df.isnull().sum()


# Promotional markdowns are discounts that come from any type of promotional sale such as a temporary 
# price reduction, coupons, endcap promotions and more. Discounts have usually
# an effect on sales and I want to evaluate this. However, Markdown columns contain too many NaN.
# I will change them to 0, no discount on this date.

# In[78]:


# fill missing values in markdown columns with 0
list_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df[list_cols] = df[list_cols].fillna(0)


# In[79]:


#fill weekly_sales which are negative values by propagating last valid observation forward to next valid
df["Weekly_Sales"].fillna(method='ffill',inplace=True)


# In[80]:


df.isnull().sum()


# In[127]:


# convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

df['Week'] = df.Date.dt.week # for the week data
df['Year'] = df.Date.dt.year # for the year data


# In[128]:


df.info()


# In[131]:


# check out min and max values in date column(first and last dates of the datset)
df["Date"].min(), df["Date"].max()


# The dataset covers the time period from the 5th of February 2010 to the 26th of October 2012

# In[132]:


#Visualizing the Type of the Stores along with their percentage

import matplotlib.pyplot as plt


labels = df_stores["Type"].value_counts()[:10].index
values = df_stores["Type"].value_counts()[:10].values
colors = ['orange', 'green', 'blue']

plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

plt.axis('equal')

plt.title('Percentage of Store Types')

plt.show()


# In[136]:



weekly_sales_2010 = df[df['Year']==2010]['Weekly_Sales'].groupby(df['Week']).mean()

plt.figure(figsize=(10, 6)) 
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values) # for plotting then lineplot

plt.title('Average Weekly Sales in 2010', fontsize=14)
plt.xlabel('Week', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)

plt.show()


# In[137]:


weekly_sales_2011 = df[df['Year']==2011]['Weekly_Sales'].groupby(df['Week']).mean()

plt.figure(figsize=(10, 6)) 
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values) # for plotting then lineplot

plt.title('Average Weekly Sales in 2011', fontsize=14)
plt.xlabel('Week', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)

plt.show()


# In[138]:


weekly_sales_2012 = df[df['Year']==2012]['Weekly_Sales'].groupby(df['Week']).mean()

plt.figure(figsize=(10, 6)) 
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values) # for plotting then lineplot

plt.title('Average Weekly Sales in 2012', fontsize=14)
plt.xlabel('Week', fontsize=12)
plt.ylabel('Average Sales', fontsize=12)

plt.show()


# #datset has data until 26th of October 2012

# In[139]:


# Plotting the above three plot together 
plt.figure(figsize=(20,8))
sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
plt.grid()
plt.xticks(np.arange(1,60, step=1))
plt.legend(['2010', '2011', '2012'], loc='best', fontsize=16)
plt.title('Average Weekly Sales Per Year', fontsize=20)
plt.ylabel('Sales', fontsize=16)
plt.xlabel('Week', fontsize=16)
plt.show()


# In[141]:


# Average Sales per Department

weekly_sales = df['Weekly_Sales'].groupby(df['Dept']).mean()
plt.figure(figsize=(25,12))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales per Department', fontsize=20)
plt.xlabel('Department', fontsize=16)
plt.ylabel('Sales', fontsize=16)
plt.show()


# There are 99 department but some of departments are missing.

# In[142]:


# Average Sales per Store

weekly_sales = df['Weekly_Sales'].groupby(df['Store']).mean()
plt.figure(figsize=(20,12))
sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')
plt.grid()
plt.title('Average Sales per Store', fontsize=20)
plt.xlabel('Store', fontsize=16)
plt.ylabel('Sales', fontsize=16)
plt.show()


# In[146]:





# In[ ]:




