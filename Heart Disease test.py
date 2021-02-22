#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Setuppo i dati
import pandas as pd
import numpy as np
heart_disease = pd.read_csv("/heart-disease.csv")
heart_disease


# In[28]:


# Features matrix
x = heart_disease.drop("target", axis=1)
# Labels
y = heart_disease["target"]


# In[30]:


# Scelta del modello e hyperparameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.get_params()


# In[32]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[34]:


clf.fit(x_train, y_train);


# In[38]:


# Prediction
y_preds = clf.predict(x_test)
y_preds


# In[39]:


y_test


# In[45]:


# Valutazione modello sui train data
clf.score(x_train, y_train)


# In[46]:


#Valutazione modello sui test data
clf.score(x_test, y_test)


# In[48]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(classification_report(y_test, y_preds))


# In[49]:


confusion_matrix(y_test, y_preds)


# In[50]:


accuracy_score(y_test, y_preds)


# In[ ]:




