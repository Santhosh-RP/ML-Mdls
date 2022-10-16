#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model


# In[6]:


data=pd.read_csv('l.csv')


# In[7]:


data


# In[21]:


# ploting the scatter plot of land and prize
plt.figure(figsize=(12,9))
plt.scatter(data['Land(Acres)'],data['prize'],color='g',marker="o",lw=10)

plt.colorbar()


# In[26]:


x=data[['Land(Acres)']]
y=data[['prize']]


# In[28]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(x,y)


# # predict the price of land(Acres) for 9 acres

# In[30]:


reg.predict([[9]])


# In[31]:


reg.coef_


# In[32]:


reg.intercept_


# In[43]:


plt.scatter(data['Land(Acres)'],data['prize'],color='g') # Actual points of the data
#plt.plot(data['Land(Acres)'],reg.predict(x),color='red
plt.scatter(data['Land(Acres)'],reg.predict(x),color='red') # predicated values of a data


# In[47]:


plt.scatter(data['Land(Acres)'],data['prize'],color='g')
plt.plot(data['Land(Acres)'],reg.predict(x),color='red')


# In[48]:


reg.predict(x)


# In[51]:


reg.predict([[2]])


# In[52]:


data


# In[ ]:




