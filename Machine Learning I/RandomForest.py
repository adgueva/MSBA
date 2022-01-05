#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import os


# In[2]:


from __future__ import division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore')


# Load the data set
train = pd.read_csv('Train_sample_shuffle.csv')
train.columns = ["col1","id","click","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model",
"device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]
# delete first column
train = train.iloc[: , 1:]


# In[3]:


train_cols = [x for x in train.columns if x!='id' and x!='click']
data = train


# In[4]:


data.head()


# In[5]:


# Feature Engineering

# get the day information
data['day'] = data['hour'].apply(lambda x: str(x)[4:6]).astype(int)

# function to turn 'hour' into date
def get_date(hour):
    y = '20'+str(hour)[:2]
    m = str(hour)[2:4]
    d = str(hour)[4:6]
    return y+'-'+m+'-'+d

# extra the hour of the day
def tran_hour(x):
    x = x % 100
    while x in [23,0]:
        return 23
    while x in [1,2]:
        return 1
    while x in [3,4]:
        return 3
    while x in [5,6]:
        return 5
    while x in [7,8]:
        return 7
    while x in [9,10]:
        return 9
    while x in [11,12]:
        return 11
    while x in [13,14]:
        return 13
    while x in [15,16]:
        return 15
    while x in [17,18]:
        return 17
    while x in [19,20]:
        return 19
    while x in [21,22]:
        return 21


# In[6]:


data['day_hour'] = data.hour.apply(tran_hour)


# In[7]:


data.head()


# In[8]:


data_cols = [x for x in data.columns if x!='id' and x!='click']
X = data[data_cols]                  
y = data['click'].astype('category') #transforming click into a factor


# In[9]:


num_cols = X.select_dtypes(include = ['int','float']).columns.tolist()
categorical_cols = X.select_dtypes(include = ['object']).columns.tolist() #identifying non-numerical variables


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, 
)


# In[11]:


#creating factors for categorical variables
for col in categorical_cols:
    X_train[col] = X_train[col].apply(lambda x: hash(x))
    
for col in categorical_cols:
    X_test[col] = X_test[col].apply(lambda x:hash(x))


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV


# In[17]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [10,20,30], 
    'n_estimators':[50,100,500]
}

rf = RandomForestClassifier(random_state=123, max_features=2)

rf_search = RandomizedSearchCV(estimator = rf, param_distributions=param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)


# In[18]:


model = rf_search.fit(X_train,y_train)


# In[21]:


model.best_params_


# In[22]:


rf_cl = RandomForestClassifier(n_estimators=100, max_depth=20, max_features=2, random_state=123)


# In[23]:


rf_cl.fit(X_train, y_train)


# In[24]:


rf_cl_prob = rf_cl.predict_proba(X_test)


# In[26]:


from sklearn.metrics import log_loss


# In[27]:


log_loss(y_test, rf_cl_prob, eps=1e-15, normalize=True)

