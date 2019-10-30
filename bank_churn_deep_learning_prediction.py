#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pa
import matplotlib.pyplot as plt


# In[2]:


data = pa.read_csv(r'C:\Users\SACHIN K M\Desktop\python\data\datasets\kaggle\bank_churn_prediction\Churn_Modelling.csv')


# In[3]:


x = data.iloc[:, 3:13]
y = data.iloc[:, 13]


# In[4]:


x.head()


# In[5]:


y.head()


# In[6]:


geography = pa.get_dummies(x['Geography'], drop_first=True)
gender = pa.get_dummies(x['Gender'], drop_first=True)


# In[7]:


x = pa.concat([x, geography, gender], axis = 1)


# In[8]:


x = x.drop(['Geography', 'Gender'], axis =1)


# In[9]:


x.head()


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[12]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout


# In[13]:


#initialising the sequential library
classifier = Sequential()


# In[14]:


#input layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="he_uniform"))
#hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="he_uniform"))
#adding output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))


# In[15]:


classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[16]:


model_history = classifier.fit(x_train, y_train, validation_split= 0.33, batch_size = 10, nb_epoch = 100)


# In[17]:


print(model_history.history.keys())


# In[18]:


classifier.summary()


# In[20]:


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)


# In[21]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[22]:


cm


# In[23]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)


# In[24]:


score


# ### This indicates that our model is accurate to 85.95%

# #### Now we will try to optimize the accuracy by changiong the activation function and number hidden layes and 
# activation function

# In[25]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout


# In[26]:


#initialising the sequential library
classifier = Sequential()


# In[27]:


#input layer
classifier.add(Dense(activation="relu", input_dim=11, units=10, kernel_initializer="he_normal"))
#hidden layer
classifier.add(Dense(activation="relu", units=20, kernel_initializer="he_normal"))
classifier.add(Dense(activation="relu", units=20, kernel_initializer="he_normal"))
#adding output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="glorot_uniform"))


# In[28]:


classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[29]:


model_history = classifier.fit(x_train, y_train, validation_split= 0.33, batch_size = 10, nb_epoch = 100)


# In[30]:


classifier.summary()


# In[31]:


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)


# In[32]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[33]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)


# In[34]:


score


# #### We can use the dropout aswell but as my data is only have 10000 record we are not using for now

# #### On Test data we are able to improve the score to 86.2%, Still this can be improved by using hyperparameter tuning
