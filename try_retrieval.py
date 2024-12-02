#!/usr/bin/env python
# coding: utf-8

# ## Loading data and preliminary analysis

# In[1]:


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
import json
  
# Opening  JSON file
file = open('D:\coding_pranav\inheritance\chatbot\mentalhealth.json')
  
# returns JSON object as 
# Mental_Health_FAQ (1).json
# a dictionary
#data = json.load(f)
  
  

#file = pd.read_csv(r'D:\coding_pranav\inheritance\chatbot\mentalhealth.csv')
data = json.load(file)
data
  
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


# References : 
# * Word embeddings - https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# * 2D CNN when we have 3D features, such as RGB - 
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
# * Pooling layers reduce the size of the representation to speed up the computation and make features robust
# * Add a "flatten" layer which prepares a vector for the fully connected layers, for example using Sequential.add(Flatten()) -  
# https://missinglink.ai/guides/keras/using-keras-flatten-operation-cnn-models-code-examples/
# * Dense layer - A fully connected layer also known as the dense layer, in which the results of the convolutional layers are fed through one or more neural layers to generate a prediction
# * Activation functions - https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6 

# ## Vanilla RNN

# * Why use embedding layer before RNN/ LSTM layer -
# https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12
# * Learning curves - https://www.dataquest.io/blog/learning-curves-machine-learning/
# 
# 
# 
# 

# In[58]:


def define_model1(vocab_size, max_length):
    model1 = Sequential()
    model1.add(Embedding(vocab_size,100, input_length=max_length))
    model1.add(SimpleRNN(100))
    model1.add(Dense(10, activation='softmax'))   
    
    model1.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    # summarize defined model
    model1.summary()
    # plot_model(model1, to_file='model_1.png', show_shapes=True)
    return model1


# In[59]:


model1 = define_model1(vocab_size, max_length)


# In[60]:


history1 = model1.fit(X_train, y_train, epochs=10, verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)#,callbacks=callbacks)


# In[61]:


# Learning curves 

acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']
loss=history1.history['loss']
val_loss=history1.history['val_loss']

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ## CNN

# In[62]:


def define_model2(vocab_size, max_length):
    model2 = Sequential()
    model2.add(Embedding(vocab_size,300, input_length=max_length))
    model2.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model2.add(MaxPooling1D(pool_size = 4))
    model2.add(Flatten())
    model2.add(Dense(32, activation='relu'))
    model2.add(Dense(10, activation='softmax'))
    
    model2.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    # summarize defined model
    model2.summary()
    return model2


# In[63]:


model2 = define_model2(vocab_size, max_length)


# In[64]:


history = model2.fit(X_train, y_train, epochs=15, verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)


# In[65]:


# Learning curves 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ## LSTM

# In[66]:


def define_model3(vocab_size, max_length):
    model3 = Sequential()
    model3.add(Embedding(vocab_size,300, input_length=max_length))
    model3.add(LSTM(500))
    model3.add(Dense(10, activation='softmax'))
    
    model3.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    # summarize defined model
    model3.summary()
    return model3


# In[67]:


model3 = define_model3(vocab_size, max_length)


# In[68]:


history = model3.fit(X_train, y_train, epochs=15, verbose=1,validation_data=(X_test,y_test))


# In[69]:


# Learning curves 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ## GRU

# In[70]:


def define_model3(vocab_size, max_length):
    model3 = Sequential()
    model3.add(Embedding(vocab_size,300, input_length=max_length))
    model3.add(GRU(500))
    model3.add(Dense(10, activation='softmax'))
    
    model3.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    # summarize defined model
    model3.summary()
    return model3


# In[71]:


model3 = define_model3(vocab_size, max_length)


# In[72]:


history = model3.fit(X_train, y_train, epochs=15, verbose=1,validation_data=(X_test,y_test))


# In[73]:


# Learning curves 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# ## BiLSTM
# 

# In[74]:


def define_model3(vocab_size, max_length):
    model3 = Sequential()
    model3.add(Embedding(vocab_size,300, input_length=max_length))
    model3.add(Bidirectional(LSTM(500)))
    model3.add(Dense(10, activation='softmax'))
    
    model3.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    # summarize defined model
    model3.summary()
    return model3


# In[75]:


model3 = define_model3(vocab_size, max_length)


# In[76]:


history = model3.fit(X_train, y_train, epochs=10, verbose=1,validation_data=(X_test,y_test))


# In[77]:


# Learning curves 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(16,8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# Future scope -
# * embedding layer : GloVe
# * cross validation for testing
# * grid search CV

# ## Predictions

# In[84]:


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


from tensorflow.keras.models import load_model
model = model3
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')


# In[87]:


def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    # stop_words = set(stopwords.words('english'))
    # tokens = [w for w in tokens if not w in stop_words]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens


# In[88]:


def remove_stop_words_for_input(tokenizer,df,feature):
    doc_without_stopwords = []
    entry = df[feature][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df


# In[89]:


def encode_input_text(tokenizer_t,df,feature):
    t = tokenizer_t
    entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=10, padding='post')
    return padded


# In[90]:


def get_pred(model,encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred


# In[91]:


def bot_precausion(df_input,pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred


# In[92]:


def get_response(df2,pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]


# In[93]:


def bot_response(response,):
    print(response)


# In[96]:


# correct response
while(input != "exit"):
    df_input = get_text(input("Enter some text: "))

    # load artifacts 
    tokenizer_t = joblib.load('tokenizer_t.pkl')
    vocab = joblib.load('vocab.pkl')

    df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
    encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

    pred = get_pred(model1,encoded_input)
    pred = bot_precausion(df_input,pred)

    response = get_response(df2,pred)
    bot_response(response)


# In[95]:


# # wrong response

# df_input = get_text("what is mental health")

# #load artifacts 
# tokenizer_t = joblib.load('tokenizer_t.pkl')
# vocab = joblib.load('vocab.pkl')

# df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
# encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

# pred = get_pred(model1,encoded_input)
# pred = bot_precausion(df_input,pred)

# response = get_response(df2,pred)
# bot_response(response)


# # In[ ]:





# # In[ ]:




