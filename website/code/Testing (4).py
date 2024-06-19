#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import tensorflow_text as text
import numpy as np
import speech_recognition as sr
import tensorflow_hub as hub
from flask import render_template
import pyaudio
import wave
import threading
import os


# In[8]:


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

#print("Before loading model")
loaded_model = tf.keras.models.load_model(
    'hate_speech_classification_model_09_30.h5',
    custom_objects={
        'BERTPreprocessLayer': BERTPreprocessLayer,
        'BERTEncoderLayer': BERTEncoderLayer
    }
)
print("After loading model")


# In[11]:


def analyze_text_sentiment(input_text):
    prediction = loaded_model.predict([input_text])   
    return prediction

input_text = "i am very good boy"

prediction = analyze_text_sentiment(input_text)
if prediction[0][0]>0.7:
    print("The Sentiment detected is Not Offence")
else:
    print("The Sentiment detected is Offence")


# In[ ]:




