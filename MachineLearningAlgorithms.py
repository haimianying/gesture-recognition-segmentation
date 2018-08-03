
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import random
random.seed(25)


# ### Preparing data

# In[3]:

train = pd.read_csv('training/train.csv')
test = pd.read_csv('test/test.csv')


# In[4]:

y_test = test['label']
y_train = train['label']


# In[5]:

X_test = test.drop('label',axis=1)
X_train = train.drop('label',axis=1)


# ### Gaussian Naive Bayes

# In[6]:

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)
accuracy_gnb = clf_gnb.score(X_test, y_test)
print "Accuracy with Gaussian Naive Bayes => "+ str(round(accuracy_gnb*100, 2)) + " %"


# ### Logistic Regression

# In[7]:

from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
accuracy_lr = clf_lr.score(X_test, y_test)
print "Accuracy with Logistic Regression => "+ str(round(accuracy_lr*100, 2)) + " %"


# ### Decision Tree Classifier

# In[8]:

from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier()
clf_dt.fit(X_train, y_train)
accuracy_dt = clf_dt.score(X_test, y_test)
print "Accuracy with Decision Tree Classifier => "+ str(round(accuracy_dt*100, 2)) + " %"


# ### Nearest Neighbor Classifier

# In[9]:

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)
accuracy_knn = clf_knn.score(X_test, y_test)
print "Accuracy with K Nearest Neighbor Classifier => "+ str(round(accuracy_knn*100, 2)) + " %"


# ### SVM Classifier

# In[10]:

from sklearn.svm import SVC
clf_svm = SVC()
clf_svm.fit(X_train, y_train)
accuracy_svm = clf_svm.score(X_test, y_test)
print "Accuracy with SVM Classifier => "+ str(round(accuracy_svm*100, 2)) + " %"


# ### Random Forest

# In[11]:

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
clf_rf.fit(X_train, y_train)
accuracy_rf = clf_rf.score(X_test, y_test)
print "Accuracy with Random Forest Classifier => "+ str(round(accuracy_rf*100, 2)) + " %"


# ### Adaboost

# In[12]:

from sklearn.ensemble import AdaBoostClassifier
clf_ab = AdaBoostClassifier()
clf_ab.fit(X_train, y_train)
accuracy_ab = clf_ab.score(X_test, y_test)
print "Accuracy with Adaboost Classifier => "+ str(round(accuracy_ab*100, 2)) + " %"

