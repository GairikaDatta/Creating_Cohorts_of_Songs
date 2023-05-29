#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('rolling_stones_spotify.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# # Find Ten Least Popular Songs in the Spotify Dataset.

# In[6]:


sorted_df = df.sort_values('popularity', ascending= True).head(10)
sorted_df


# In[7]:


df.describe().transpose()


# In[8]:


df = df.drop(['id','uri','name','album','release_date'], axis=1)


# In[9]:


df


# In[10]:


df['song_like_dislike'] = df['popularity'].apply(lambda x: 0 if x < 15 else 1)


# In[11]:


df


# In[12]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(df[df['song_like_dislike']==1]['danceability'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['danceability'], color='blue', bins=40)
plt.legend((1,0))
plt.subplot(4,4,2)
sns.distplot(df[df['song_like_dislike']==1]['energy'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['energy'], color='blue', bins=40)
plt.legend((1,0))
plt.subplot(4,4,3)
sns.distplot(df[df['song_like_dislike']==1]['loudness'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['loudness'], color='blue', bins=40)
plt.legend((1,0))


# In[13]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(df[df['song_like_dislike']==1]['acousticness'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['acousticness'], color='blue', bins=40)
plt.subplot(4,4,2)
sns.distplot(df[df['song_like_dislike']==1]['instrumentalness'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['instrumentalness'], color='blue', bins=40)
plt.subplot(4,4,3)
sns.distplot(df[df['song_like_dislike']==1]['liveness'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['liveness'], color='blue', bins=40)


# In[14]:


plt.figure(figsize=(16,16))
plt.subplot(4,4,1)
sns.distplot(df[df['song_like_dislike']==1]['valence'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['valence'], color='blue', bins=40)
plt.subplot(4,4,2)
sns.distplot(df[df['song_like_dislike']==1]['tempo'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['tempo'], color='blue', bins=40)
plt.subplot(4,4,3)
sns.distplot(df[df['song_like_dislike']==1]['duration_ms'], color='red', bins=40)
sns.distplot(df[df['song_like_dislike']==0]['duration_ms'], color='blue', bins=40)
plt.legend((1,0))


# In[15]:


X = df.drop(['song_like_dislike'],axis=1)
y=df['song_like_dislike']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)


# In[18]:


X_train.head()


# In[19]:


X_test.head()


# In[20]:


y_train.tail()


# In[21]:


y_test.head()


# In[22]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier


# In[23]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)
xr = XGBRegressor()
xr.fit(X_train,y_train)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)


# In[24]:


lr_pred = lr.predict(X_test)
print(confusion_matrix(y_test, lr_pred))
print('\n')
print(classification_report(y_test, lr_pred))


# In[25]:


knn_pred = knn_model.predict(X_test)
print(confusion_matrix(y_test, knn_pred))
print('\n')
print(classification_report(y_test, knn_pred))


# In[27]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)
y_pred5 = xr.predict(X_test)
y_pred6 = knn_model.predict(X_test)
df1 = pd.DataFrame({'Actual': y_test, 'LR': y_pred1, 'SVM': y_pred2, 'RF': y_pred3, 'GR': y_pred4, 'XR':y_pred5, 'KNN':y_pred6})


# In[35]:


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[28]:


df1


# In[40]:


precision = precision_score(y_test, lr_pred)
recall = recall_score(y_test, lr_pred)
print('Precision:', precision)
print('Recall:', recall)


# In[34]:


f1 = f1_score(y_test, rf_pred)
print('F1-score:', f1)


# In[39]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred4)
auc_roc = roc_auc_score(y_test, y_pred4)

# plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




