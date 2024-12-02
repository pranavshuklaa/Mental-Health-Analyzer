# import libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
import string
import re
import joblib
import json
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
import keras.api._v2.keras as keras
import tensorflow.keras
#
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPooling1D, SimpleRNN, GRU, LSTM, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[2]:


# download dependencies 

# uncomment if running for the first time
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


# In[3]:

  
# Opening  JSON file
import json
file = open('D:\coding_pranav\inheritance\chatbot\mentalhealth.json')
data = json.load(file)
data 
# returns JSON object as 
# Mental_Health_FAQ (1).json
# a dictionary
#data = json.load(f)
  
  

#file = pd.read_csv(r'D:\coding_pranav\inheritance\chatbot\mentalhealth.csv')

  
# returns JSON object as 
# Mental_Health_FAQ (1).json
# a dictionary
#data = json.load(f)
  
  

#file = pd.read_csv(r'D:\coding_pranav\inheritance\chatbot\mentalhealth.csv')
# data = json.load(file)
# data


# In[4]:


# convert to dataframes 
 
def frame_data(feat_1,feat_2,is_pattern):
  is_pattern = is_pattern
  df = pd.DataFrame(columns=[feat_1,feat_2])
  for intent in data['intents']:
    if is_pattern:
      for pattern in intent['patterns']:
        w = pattern
        df_to_append = pd.Series([w,intent['tag']], index = df.columns)
        df = df.append(df_to_append,ignore_index=True)
    else:
      for response in intent['responses']:
        w = response
        df_to_append = pd.Series([w,intent['tag']], index = df.columns)
        df = df.append(df_to_append,ignore_index=True)
  return df


# In[5]:


df1 = frame_data('questions','labels',True)
df1.head()


# In[6]:


# no of patterns

(df1.labels.value_counts(sort=False))


# In[7]:


df2 = frame_data('response','labels',False)
df2.head()


# ## Data preprocessing

# In[8]:


# preprocessing text

lemmatizer = WordNetLemmatizer()

vocab = Counter()
labels = []
def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

def remove_stop_words(tokenizer,df,feature):
    doc_without_stopwords = []
    for entry in df[feature]:
        tokens = tokenizer(entry)
        joblib.dump(tokens,'tokens.pkl')
        doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return


# In[9]:


def create_vocab(tokenizer,df,feature):
    for entry in df[feature]:
        tokens = tokenizer(entry)   
        vocab.update(tokens)
    joblib.dump(vocab,'vocab.pkl')
    return


# In[10]:


import nltk
nltk.download('omw-1.4')

create_vocab(tokenizer,df1,'questions')


# In[11]:


vocab


# In[12]:


vocab_size = len(vocab)
vocab_size


# In[13]:


df1.groupby(by='labels',as_index=False).first()['questions']


# In[14]:


# test_list contains the first element of questions

test_list = list(df1.groupby(by='labels',as_index=False).first()['questions'])
test_list


# In[15]:


# indices of the testing dataset

test_index = []
for i,_ in enumerate(test_list):
    idx = df1[df1.questions == test_list[i]].index[0]
    test_index.append(idx)
test_index


# In[16]:


# train indices are the all indices minus the testing indices 

train_index = [i for i in df1.index if i not in test_index]
train_index 


# In[17]:


def convert_seq(df,feature):
#     text = ' '.join(list(vocab.keys()))
    t = Tokenizer()
    entries = [entry for entry in df[feature]]
    print(entries)
    print('----')
    t.fit_on_texts(entries)
    joblib.dump(t,'tokenizer_t.pkl')   # why a pkl file
    vocab_size = len(t.word_index) +1 # +1 for oov 
    print(t.word_index)
    entries = [entry for entry in df[feature]]
    max_length = max([len(s.split()) for s in entries])
    print('----')
    print("max length of string is : ",max_length)
    print('----')
    encoded = t.texts_to_sequences(entries)
    print(encoded)
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    print('----')
    print(padded)
    return padded, vocab_size


# **fit_on_texts** updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency. 0 is reserved for padding. So lower integer means more frequent word (often the first few are stop words because they appear a lot).

# Now that we have a vocabulary of words in the dataset, **each of the patterns can be encoded into numerical features for modeling, using any of the common text encoding techniques—count vectorizer**, term frequency-inverse document frequency (TF-IDF), hashing, etc.
# 
# Using TensorFlow.Keras text_to_sequence, we can **encode each pattern corpus to vectorize a text corpus by turning each text into either a sequence of integers** (each integer being the index of a token in a dictionary) or into a vector where the coefficient for each token could be binary, based on word count which is based on TF-IDF. The resulting vectors will be post-padded with zeros so as to equal the length of the vectors.

# In[18]:


X,vocab_size = convert_seq(df1,'questions')


# In[19]:


with open('tokenizer_t.pkl', 'rb') as f:
    data = pickle.load(f)


# In[20]:


data.index_word


# In[21]:


data.word_counts


# In[22]:


X


# In[23]:


vocab_size


# In[24]:


df_encoded = pd.DataFrame(X)


# In[25]:


df_encoded


# In[26]:


df1.head(10)


# In[27]:


df_encoded['labels'] = df1.labels
df_encoded.head(10)


# In[28]:


df_encoded


# In[29]:


from sklearn.preprocessing import LabelEncoder
lable_enc = LabelEncoder()

# encoding the labels

labl = lable_enc.fit_transform(df_encoded.labels)
labl


# In[30]:


len(labl)


# In[31]:


mapper = {}
for index,key in enumerate(df_encoded.labels):
    if key not in mapper.keys():
        mapper[key] = labl[index]
mapper


# Repeat the same for df2

# In[32]:


df2.head()


# In[33]:


df2


# In[34]:


df2.labels = df2.labels.map(mapper).astype({'labels': 'int32'})
df2.head()


# In[35]:


df2.to_csv('response.csv',index=False)


# In[36]:


df_encoded.head()


# In[37]:


train_index


# In[38]:


test_index


# In[39]:


train = df_encoded.loc[train_index]
test = df_encoded.loc[test_index]


# ## Training and testing

# In[40]:


train


# In[41]:


test.head()


# In[42]:


train.labels.value_counts()


# In[43]:


test.labels.value_counts()


# In[44]:


train


# In[45]:


X_train = train.drop(columns=['labels'],axis=1)
y_train = train.labels
X_test = test.drop(columns=['labels'],axis=1)
y_test = test.labels


# In[46]:


X_train.head()


# In[47]:


y_train =pd.get_dummies(y_train).values
y_test =pd.get_dummies(y_test).values


# In[48]:


X_test


# In[49]:


X_train


# In[50]:


y_train


# In[51]:


y_train[0]


# In[52]:


y_test


# In[53]:


y_train[0].shape,y_test[0].shape


# In[54]:


X_train.shape


# In[55]:


X_test.shape


# In[56]:


max_length = X_train.shape[1]
output = 16                  # no of classes


# Reference for the model below:
# 
# *   https://keras.io/api/callbacks/model_checkpoint/
# *   https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau

# In[57]:


early_stopping = EarlyStopping(monitor='val_loss',patience=10) #patience : number of epochs with no improvement after which training will be stopped

checkpoint = ModelCheckpoint("model-v1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

callbacks = [early_stopping,checkpoint,reduce_lr]



import tensorflow as tf
tf.get_logger().setLevel('ERROR')


# In[85]:


def get_text(str_text):
    # print(str_text)
    input_text  = [str_text]
    df_input = pd.DataFrame(input_text,columns=['questions'])
    df_input
    return df_input


# In[86]:



# correct response