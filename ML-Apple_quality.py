#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt


# In[5]:


a=pd.read_csv("apple.csv")
a


# In[15]:


#Drop column A_id

b=a.drop(['A_id'],axis=1)
b


# In[24]:


#Drop last row
b=b.drop(index=4000,axis=0)
b


# In[22]:


b.head()


# In[25]:


#Split data

x=b.drop(["Quality"],axis=1)
y=b["Quality"]
display(x)
display(y)


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
display(x_test)
display(x_train)


# In[56]:


dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(x_train,y_train)
dt_predictions=dt_classifier.predict(x_test)
print("Decision_tree accuracy: ",accuracy_score(y_test,dt_predictions))
print("\nDecision_tree confusion matrix: \n",confusion_matrix(y_test,dt_predictions))
print("\nDecision_tree classification report: \n",classification_report(y_test,dt_predictions))


# In[57]:


nb_classifier=GaussianNB()
nb_classifier.fit(x_train,y_train)
nb_predictions=nb_classifier.predict(x_test)
print("Naive bayes accuracy: ",accuracy_score(y_test,nb_predictions))
print("\nNaive bayes confusion matrix: \n",confusion_matrix(y_test,nb_predictions))
print("\nNaive bayes classification report: \n",classification_report(y_test,nb_predictions))


# In[58]:


svm_classifier=SVC()
svm_classifier.fit(x_train,y_train)
svm_predictions=svm_classifier.predict(x_test)
print("svm accuracy: ",accuracy_score(y_test,svm_predictions))
print("\nsvm confusion matrix: \n",confusion_matrix(y_test,svm_predictions))
print("\nsvm classification report: \n",classification_report(y_test,svm_predictions))


# In[61]:


rf_classifier=RandomForestClassifier()
rf_classifier.fit(x_train,y_train)
rf_predictions=rf_classifier.predict(x_test)
print("random forest accuracy: ",accuracy_score(y_test,rf_predictions))
print("\nrandom forest confusion matrix: \n",confusion_matrix(y_test,rf_predictions))
print("\nrandom forest classification report: \n",classification_report(y_test,rf_predictions))


# In[64]:


clf=DecisionTreeClassifier()
clf.fit(x,y)
plt.figure()
plot_tree(clf,filled=True)
plt.show()


# In[ ]:





# In[ ]:




