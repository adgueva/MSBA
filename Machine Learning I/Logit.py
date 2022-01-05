#!/usr/bin/env python
# coding: utf-8

# In[4]:


######################################### Load Libraries and Modules #########################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import gzip
import os
import gc


# In[6]:


######################################### Load Libraries and Data #########################################

# To write a Python 2/3 compatible codebase, the first step is to add this line to the top of each module
from __future__ import division, print_function, unicode_literals

# Import necessary libraries and modules 
# Matplotlib inline allows the output of plotting commands will be displayed inline
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model        # The sklearn.linear_model module implements generalized linear models. LR is part of this module
import warnings
warnings.filterwarnings('ignore')


# Load the data set
# this is a shuffled data of about 1.5 million rows
# we used it instead of the original data for computational speed
train = pd.read_csv('Train_sample_shuffle.csv')

# add column names
train.columns = ["col1","id","click","hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model",
"device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]


# In[7]:


# plot to see if y has balanced 0 and 1
sns.countplot(x = 'click', hue = "click", data = train)
plt.show()
# here we can see it is unbalanced, however, later when we add class_weight='balanced' to the logistic regression,
# we noticed that the model trained by balanced dataset actually performs worse on both unbalanced and balanced test data
# therefore, we decided not to balance it for this regression


# In[8]:


# delete first column as it is created by shuffling and has no meaning
train = train.iloc[:,1:]
data = train


# In[10]:


# Feature Engineering
# hour of the day
data['hour_new']=data['hour'].map(lambda x:  x.hour)


# In[11]:


# transform the data type of 'hour'
from datetime import datetime
data['hour']=data['hour'].map(lambda x: datetime.strptime(str(x),"%y%m%d%H"))

# adding day of the week
data['dayoftheweek']=data['hour'].map(lambda x:  x.weekday())


# In[12]:


# time stamp with the beginning of the data as 0
# get the day information
data['day'] = data['hour'].apply(lambda x: str(x)[4:6]).astype(int)
data['time']=(data['day'].values - data['day'].min()) * 24  + data['hour'].values


# In[13]:


# we created this in the hope that its interaction with other terms will help with creating more meaningful features
# it was not that helpful but we included the code here
data['device_ip'].value_counts()
data['user_id'] = data['device_id'] +  '_' + data['device_ip']+  '_'  +  data['device_model']


# In[14]:


# sanity check
data.dtypes


# In[16]:


# one-hot-encoding

from sklearn.preprocessing import LabelEncoder
for col in data.columns:
    if col!='id' and col!='click': 
        if data[col].dtypes == 'O':
            print(col)
            data[col+'_labelencode'] = LabelEncoder().fit_transform(data[col].values)


# In[17]:


# sanity check again
data.head()


# In[18]:


train_cols = [x for x in data.columns if x!='id' and x!='click'and data[x].dtypes!='O' and data[x].dtypes!='datetime64[ns]']

# using a smaller dataset because randomized search was taking a long time
data_2 = data.iloc[:2000,:]

X = data_2[train_cols]                  # Specify attributes
y = data_2['click'].astype('category') # Specify target variable


# In[19]:


# variables used for training
train_cols


# In[20]:


X.info()


# In[21]:


#########################################  Without randomized search  ########################################
############################################    Split the Data   ############################################

# Split validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, 
    #stratify=y
)

#################################### Train the Logistic Regression Model ####################################

# We create an instance of the Classifier
# Logistic Regression (aka logit) classifier.

clf = linear_model.LogisticRegression(multi_class='auto', #accomondates multi-class categorical target variable
                                      penalty='l1', # use lasso to perform feature selection
                                      solver='liblinear' # using this solver because the default one cannot handle lasso
                                      ) 
                                                                                
# Train the model (fit the data)
# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, 
# of size [n_samples, n_features] holding the training samples, and an array Y of integer values, size [n_samples], 
# holding the class labels for the training samples:
clf = clf.fit(X_train, y_train)
print('The weights of the attributes are:', clf.coef_)


#################################### Apply the Logistic Regression Model ####################################

y_pred = clf.predict(X_test)             # Classification prediction
y_pred_prob = clf.predict_proba(X_test)  # Class probabilities
print(y_pred[0], y_pred_prob[0], np.sum(y_pred_prob[0]))


# In[22]:


################################### Evaluate the Logistic Regression Model ##################################

# Build a text report showing the main classification metrics (out-of-sample performance)
print(classification_report(y_test, y_pred))


# In[23]:


# Check log loss
from sklearn.metrics import log_loss

log_loss(y_test, y_pred_prob, eps=1e-15, normalize=True)


# In[24]:


data.shape


# In[25]:


from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

# using a smaller dataset because randomized search was taking a long time
data_3 = data.iloc[:1597000,:]
X_3 = data_3[train_cols]                  # Specify attributes
y_3 = data_3['click'].astype('category') # Specify target variable


# In[26]:


# define model
model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none','l1','elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=10, scoring='neg_log_loss', n_jobs=-1, cv=cv, random_state=1)


# In[ ]:


# execute search
result = search.fit(X_3, y_3)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

