from func import *

# References : 
# * Word embeddings - https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# * 2D CNN when we have 3D features, such as RGB - 
# https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
# * Pooling layers reduce the size of the representation to speed up the computation and make features robust
# * Add a "flatten" layer which prepares a vector for the fulpython ly connected layers, for example using Sequential.add(Flatten()) -  
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
    #plot_model(model1, to_file='model_1.png', show_shapes=True)
    return model1


# In[59]:


model1 = define_model1(vocab_size, max_length)


# In[60]:


history1 = model1.fit(X_train, y_train, epochs=10, verbose=1,validation_data=(X_test,y_test),callbacks=callbacks)#,callbacks=callbacks)


# In[61]:


# Learning curves 

# acc = history1.history['accuracy']
# val_acc = history1.history['val_accuracy']
# loss=history1.history['loss']
# val_loss=history1.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


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

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


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

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


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

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


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

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']

# plt.figure(figsize=(16,8))
# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()


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

