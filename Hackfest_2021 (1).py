#!/usr/bin/env python
# coding: utf-8

# <img src="download.jfif" width="250"/>

# <img src="ISM_logo.png" width="300"/>

# <img src="Team_logo.png" width="300"/>

# # HackFest 2021

# * Import the required ML Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# * Importing the dataset

# In[3]:


df = pd.read_csv('corona_tested.csv') 


# * Basic Data Exploration

# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


for col in df:
    print(col)
    print(df[col].unique())


# In[11]:


df.isnull().sum()


# * Label Encoding for features of dataset having string value

# In[19]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


cols = ["corona_result","age_60_and_above","gender"]


# In[21]:


le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])


# In[22]:


df.head()


# * Preparing the data

# In[24]:


df1 = df.drop("test_date",axis=1)


# In[25]:


df2 = df1.drop("test_indication",axis=1)


# In[26]:


df2.head()


# In[27]:


X = df2.drop('corona_result', axis=1)
Y = df2['corona_result']


# In[28]:


X.head()


# In[29]:


Y.head()


# * Splitting data for training and testing

# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


# In[31]:


X_train.head()


# In[32]:


Y_train.head()


# In[33]:


X_test.head()


# In[34]:


Y_test.head()


# #  Implementing Random Forests Classifier 

# In[ ]:




