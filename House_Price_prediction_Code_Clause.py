#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv("data.csv")


# In[5]:


data


# In[6]:


data.info()


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


from sklearn.model_selection import train_test_split

x = data.drop(['country'], axis=1)
y = data['country']


# In[12]:


x


# In[13]:


y


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) 


# In[15]:


train_data = x_train.join(y_train)


# In[16]:


train_data


# In[17]:


train_data.hist(figsize = (20, 10))


# In[18]:


numeric_data = train_data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
train_data['date'] = pd.to_datetime(train_data['date'])


# In[19]:


sns.displot(data['price'])


# In[20]:


sns.pairplot(data)


# In[21]:


numeric_data = train_data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()


plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')


# In[24]:


numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()


# In[26]:


train_data['bedrooms'] = np.log(train_data['bedrooms'] + 1)
train_data['bathrooms'] = np.log(train_data['bathrooms'] +11)
train_data['floors'] = np.log(train_data['floors'] + 1)
train_data['waterfront'] = np.log(train_data['waterfront'] + 1)


# In[27]:


train_data.hist(figsize=(20, 10))


# In[18]:


data


# In[19]:


train_data


# In[28]:


train_data.sqft_living.value_counts()


# In[31]:


X = data[['sqft_living', 'floors', 'bedrooms',
               'bathrooms', 'yr_built']]

y = data['price']


# In[32]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 


# In[33]:


from sklearn.linear_model import LinearRegression 

lm = LinearRegression() 

lm.fit(X_train,y_train) 


# In[34]:


print(lm.intercept_)


# In[38]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[39]:


predictions = lm.predict(X_test)  


# In[40]:


plt.scatter(y_test,predictions)


# In[41]:


sns.distplot((y_test-predictions),bins=50); 


# In[43]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

