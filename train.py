#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import gensim
import codecs


# In[64]:


from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib


# In[6]:


def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs,dtype='float')

def buildVecs(data,model):
    new_vec = []
    for line in data:
        vecs = getWordVecs(line,model)
        if len(vecs) > 0:
            vecsArray = sum(np.array(vecs)) / len(vecs)
            new_vec.append(vecsArray)
    return new_vec


# In[35]:


df = pd.read_csv('./data.csv')
content = df['content'].tolist()
sents = [eval(cont) for cont in content]
print(len(sents))
labels = df['label'].tolist()
print(len(labels))


# In[33]:


model = gensim.models.KeyedVectors.load_word2vec_format('semi.txt',binary=False)


# In[45]:


data_vec = []
data_label = []
for i in range(len(sents)):
    senl= []
    sent = sents[i]
    for word in sent:
        try:
            senl.append(model[word])
        except KeyError:
             continue
    
    sen_arr = np.array(senl,dtype='float')
#     print(sen_arr.shape)
    if sen_arr.shape[0] > 0:
        sen_mean = sum(np.array(sen_arr)) / len(sen_arr)
        data_vec.append(sen_mean)
        data_label.append(labels[i])


# In[46]:


print(len(data_vec))
print(len(data_label))


# In[47]:


data_vec[0]


# In[48]:


data_label[0]


# In[51]:


clf = svm.SVC(C=2,probability=True)
clf.fit(data_vec,data_label)


# In[52]:


clf.score(data_vecec,data_label)


# In[53]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


print(np.array(data_vec).shape)
pred_probas = clf.predict_proba(data_vec)[:,1]
print(pred_probas.shape)
fpr,tpr,_ = metrics.roc_curve(data_label,pred_probas)
roc_auc = metrics.auc(fpr,tpr)
plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc = 'lower right')
plt.show()


# In[65]:


joblib.dump(clf,'semi_mode.m')


# In[ ]:




