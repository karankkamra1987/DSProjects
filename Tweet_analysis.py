
# coding: utf-8

# In[90]:


get_ipython().magic('matplotlib inline')
import pandas as pd
from nltk.stem import PorterStemmer
import string


# In[91]:


# data = pd.read_csv('pure.csv',encoding='ISO-8859-1')
data = pd.read_csv('review_b_cybg.csv',encoding='ISO-8859-1')
data.rename(columns = {'text':'final_Combined'},inplace=True)
data['final_Combined'] = data['final_Combined'].str.lower()
data['final_Combined'] = data['final_Combined'].apply(lambda x: x.strip('[').strip(']').split(' '))


# In[93]:


data.groupby('score').size()


# In[94]:


flag_words = ['error','request','fatal','issue','inconsitency','crash','defect','reopen','no','not','uninstall',
             'unable','issue','please','plz','distract','despite','doesn\'t','slow','wrong','won\'t','poorly','terrible'
             ,'rubbish','tried','cannot','unreliable','unhappy','never','poor','crap','reported','failed']


# In[95]:


def strip_space(x):
    return [elem.strip(' ') for elem in x]

def check_flag_words(m):
    for elem in flag_words:
        if elem in m:
            return elem
    return 'no problemo'


# In[96]:


data['final_Combined'] = data['final_Combined'].apply(strip_space)


# In[97]:


ps = PorterStemmer()
data['final_Combined']= data['final_Combined'].apply(lambda x: [ps.stem(elem) for elem in x] )
flag_words = [ps.stem(elem) for elem in flag_words]


# In[98]:


data['type']=data['final_Combined'].apply(check_flag_words)
data = data[data['type']!='no problemo']
# data.groupby('type').size().plot.bar()


# In[100]:


data.groupby('score').size()


# In[103]:


data[['final_Combined','type','score']].to_csv('pure_solved.csv')


# In[ ]:




