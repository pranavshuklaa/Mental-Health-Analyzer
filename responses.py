# from model import *
# from func import *
# # from func import get_text
# # from func import remove_stop_words_for_input
# # from func import encode_input_text
# # from func import get_pred
# # from func import model1
# # from func import bot_precausion
# # from func import get_response
# # from func import bot_response
# # from func import joblib
# # from func import  tokenizer
# # from func import df2


# # In[86]:






# # ## Predictions

# # In[84]:




# df_input = get_text('How to find mental health professional for myself?')

# # load artifacts 
# tokenizer_t = joblib.load('tokenizer_t.pkl')
# vocab = joblib.load('vocab.pkl')

# df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
# encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

# pred = get_pred(model1,encoded_input)
# pred = bot_precausion(df_input,pred)

# response = get_response(df2,pred)
# bot_response(response)

# import libraries
#-----------------------------------------------------------------------------------------------------------------
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
import tensorflow.keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense,LayerNormalization, Flatten, Conv1D, MaxPooling1D, SimpleRNN, GRU, LSTM, LSTM, Input, Embedding, TimeDistributed, Flatten, Dropout,Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# download dependencies 

# uncomment if running for the first time
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# load data
# from google.colab import files
# uploaded = files.upload()
# with open('mentalhealth - Copy.json') as file:
#   data = json.load(file)
import json
file = open('D:\coding_pranav\inheritance\chatbot\mentalhealth - Copy.json')
data = json.load(file)
data 
# data

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

df1 = frame_data('questions','labels',True)
df1.head()

# no of patterns

(df1.labels.value_counts(sort=False))

df2 = frame_data('response','labels',False)
df2.head()

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

def create_vocab(tokenizer,df,feature):
    for entry in df[feature]:
        tokens = tokenizer(entry)   
        vocab.update(tokens)
    joblib.dump(vocab,'vocab.pkl')
    return

import nltk
nltk.download('omw-1.4')

create_vocab(tokenizer,df1,'questions')

vocab

vocab_size = len(vocab)
vocab_size

df1.groupby(by='labels',as_index=False).first()['questions']
# test_list contains the first element of questions

test_list = list(df1.groupby(by='labels',as_index=False).first()['questions'])
test_list

# indices of the testing dataset

test_index = []
for i,_ in enumerate(test_list):
    idx = df1[df1.questions == test_list[i]].index[0]
    test_index.append(idx)
test_index

# train indices are the all indices minus the testing indices 

train_index = [i for i in df1.index if i not in test_index]
train_index 

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

X,vocab_size = convert_seq(df1,'questions')
with open('tokenizer_t.pkl', 'rb') as f:
    data = pickle.load(f)

data.index_word
data.word_counts
X
vocab_size
df_encoded = pd.DataFrame(X)
df_encoded = pd.DataFrame(X)
df1.head(10)
df_encoded['labels'] = df1.labels
df_encoded.head(10)
df_encoded['labels'] = df1.labels
df_encoded.head(10)
from sklearn.preprocessing import LabelEncoder
lable_enc = LabelEncoder()

# encoding the labels

labl = lable_enc.fit_transform(df_encoded.labels)
labl
len(labl)
mapper = {}
for index,key in enumerate(df_encoded.labels):
    if key not in mapper.keys():
        mapper[key] = labl[index]
mapper
df2
df2.head()
df2.to_csv('response.csv',index=False)
df_encoded.head()
train_index
test_index
train = df_encoded.loc[train_index]
test = df_encoded.loc[test_index]
train
test.head()
train.labels.value_counts()
test.labels.value_counts()
train
X_train = train.drop(columns=['labels'],axis=1)
y_train = train.labels
X_test = test.drop(columns=['labels'],axis=1)
y_test = test.labels
X_train.head()
y_train =pd.get_dummies(y_train).values
y_test =pd.get_dummies(y_test).values
X_test
X_train
y_train
y_train[0]
y_test
y_train[0].shape,y_test[0].shape
X_train.shape
X_test.shape
max_length = X_train.shape[1]
output = 16                  # no of classes
early_stopping = EarlyStopping(monitor='val_loss',patience=10) #patience : number of epochs with no improvement after which training will be stopped

checkpoint = ModelCheckpoint("model-v1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

callbacks = [early_stopping,checkpoint,reduce_lr]
import json
f = open('D:\coding_pranav\inheritance\chatbot\mentalhealth.json')
data = json.load(f)


df = pd.DataFrame(data['intents'])
df

df = pd.DataFrame(data['intents'])
df

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)
df
df['tag'].unique()
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()



from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')
print('X shape = ', X.shape)

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])
print('y shape = ', y.shape)
print('num of classes = ', len(np.unique(y)))
def define_model1(vocab_size, max_length):
    model1 = Sequential()
    model1.add(Input(shape=(X.shape[1])))
    model1.add(Embedding(input_dim=vocab_size+1, output_dim=100, mask_zero=True))
    model1.add(LSTM(32, return_sequences=True))
    model1.add(LayerNormalization())
    model1.add(LSTM(32, return_sequences=True))
    model1.add(LayerNormalization())
    model1.add(LSTM(32))
    model1.add(LayerNormalization())
    model1.add(Dense(128, activation="relu"))
    model1.add(LayerNormalization())
    model1.add(Dropout(0.2))
    model1.add(Dense(128, activation="relu"))
    model1.add(LayerNormalization())
    model1.add(Dropout(0.2))
    model1.add(Dense(len(np.unique(y)), activation="softmax"))
    model1.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    # summarize defined model
    model1.summary()
    #plot_model(model1, to_file='model_1.png', show_shapes=True)
    return model1

print(vocab_size,max_length)
model1 = define_model1(vocab_size, max_length)
history1 = model1.fit(x=X,
                          y=y,
                          batch_size=10,
                          callbacks=[tensorflow.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)],
                          epochs=70)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
def get_text(str_text):
    print(str_text)
    input_text  = [str_text]
    df_input = pd.DataFrame(input_text,columns=['questions'])
    df_input
    return df_input
from tensorflow.keras.models import load_model
model = model1
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')
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
def remove_stop_words_for_input(tokenizer,df,feature):
    doc_without_stopwords = []
    entry = df[feature][0]
    tokens = tokenizer(entry)
    doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df
def encode_input_text(tokenizer_t,df,feature):
    t = tokenizer_t
    entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=18, padding='post')
    return padded
def get_pred(model,encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred
def bot_precausion(df_input,pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab])==0 :
        pred = 1
    return pred
def get_response(df2,pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0,upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]
def bot_response(response,):
    print(response)
# correct response
question='hi'
df_input = get_text(question)

# load artifacts 
tokenizer_t = joblib.load('tokenizer_t.pkl')
vocab = joblib.load('vocab.pkl')

df_input = remove_stop_words_for_input(tokenizer,df_input,'questions')
encoded_input = encode_input_text(tokenizer_t,df_input,'questions')

pred = get_pred(model1,encoded_input)
pred = bot_precausion(df_input,pred)

response = get_response(df2,pred)
bot_response(response)







