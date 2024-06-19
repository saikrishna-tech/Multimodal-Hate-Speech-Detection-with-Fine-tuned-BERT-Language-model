#!/usr/bin/env python
# coding: utf-8

# <h2 align="center">BERT tutorial: Classify spam vs no spam emails</h2>

# In[22]:


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import pandas as pd

df = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t", quoting = 3)
df.head(5)


# In[23]:


df.groupby('Liked').describe()


# In[24]:


df['Liked'].value_counts()


# In[25]:


def determine_happiness(value):
    return 'happy' if value == 1 else 'not happy'

# Add a new column 'happiness'
df['happiness'] = df['Liked'].apply(determine_happiness)
df.head()


# <h4>Split it into training and test data set</h4>

# In[26]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Review'],df['Liked'], random_state=10)


# In[27]:


X_train.head(4)


# <h4>Now lets import BERT model and get embeding vectors for few sample statements</h4>

# <h4>Get embeding vectors for few sample words. Compare them using cosine similarity</h4>

# <h4>Build Model</h4>

# In[28]:


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout

# Load pre-trained BERT model from TensorFlow Hub
bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")

# Define a custom Keras layer for BERT preprocessing
class BERTPreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_preprocess(inputs)

# Define a custom Keras layer for BERT encoding
class BERTEncoderLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_encoder(inputs)

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = BERTPreprocessLayer()(text_input)
encoded_text = BERTEncoderLayer()(preprocessed_text)

# Extract BERT outputs
sequence_output = encoded_text['sequence_output']  # Get BERT's sequence output

# Define an RNN layer

lstm_layer = LSTM(128, return_sequences=False)

# Bidirectional wrapper for LSTM layer
bidirectional_lstm = Bidirectional(lstm_layer)(sequence_output)

# Dropout layer for regularization
dropout = Dropout(0.1)(bidirectional_lstm)

# Output layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

# Construct final model
model = tf.keras.Model(inputs=text_input, outputs=output)


# In[29]:


model.summary()


# In[30]:


METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=METRICS)


# <h4>Train the model</h4>

# In[31]:


model.fit(X_train, y_train, epochs=6, validation_split=0.2)


# In[32]:


model.evaluate(X_test, y_test)


# In[33]:


y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()


# In[34]:


import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted


# In[35]:


from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_predicted)
cm


# In[36]:


from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[37]:


print(classification_report(y_test, y_predicted))


# In[39]:


model.save("hate_speech_classification_model_09_30.h5")

