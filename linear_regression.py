#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv(r"C:\Users\dhaim\OneDrive\Desktop\python\py-master\ML\1_linear_reg\homeprices.csv")
df


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,marker='+')


# In[5]:


new_df = df.drop('price',axis='columns')
new_df


# In[8]:


price = df.price
price


# In[10]:


reg = linear_model.LinearRegression()
reg.fit(new_df,price)


# In[12]:


reg.predict([[330]])

