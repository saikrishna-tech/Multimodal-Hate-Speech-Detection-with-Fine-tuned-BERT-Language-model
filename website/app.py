
from flask import Flask, render_template, request, jsonify
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
from flask import Flask, render_template, request, redirect


app = Flask(__name__)

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
    'code/my_BERT_modelfinal.h5',
    custom_objects={
        'BERTPreprocessLayer': BERTPreprocessLayer,
        'BERTEncoderLayer': BERTEncoderLayer
    }
)
print("After loading model")

# Add similar print statements in other critical sections

@app.route('/')
def index():
    return render_template('new.html')

def analyze_text_sentiment(input_text):
    prediction = loaded_model.predict([input_text])   
    return prediction

def analyze_audio_sentiment(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        input_text = recognizer.recognize_google(audio)
        prediction = analyze_text_sentiment(input_text)
        return prediction

    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error accessing the speech recognition service: {e}"
    except Exception as e:
        return f"Error processing audio file: {e}"



@app.route('/analyze_input1', methods=['POST'])
def analyze_input1():

    input_text = request.form['input_text']
    prediction = analyze_text_sentiment(input_text)
    print(prediction)
    
    out = ""
    if prediction[0][0]>0.7:
        out = "The Sentiment detected is Not Offence"
    else:
        out = "The Sentiment detected is Offence"
    return render_template('output.html', sentiment_result=out)


@app.route('/analyze_input2', methods=['POST'])
def analyze_input2():

    audio_file = request.files['audio_file']
    prediction = analyze_audio_sentiment(audio_file)
    
    print(prediction)
    
    out = ""
    if prediction[0][0]>0.7:
        out = "The Sentiment detected is Not Offence "
    else:
        out = "The Sentiment detected is Offence"
    return render_template('output.html', sentiment_result=out)



class Recorder:
    def __init__(self, file_name, save_dir, chunk=1024, channels=2, rate=44100, format=pyaudio.paInt16):
        self.file_name = file_name
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, file_name)
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.format = format
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=format, channels=channels,
                                      rate=rate, input=True,
                                      frames_per_buffer=chunk)
        self.recording = False
        self.lock = threading.Lock()

    def start_recording(self):
        self.recording = True
        while self.recording:
            data = self.stream.read(self.chunk)
            with self.lock:
                self.frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        with wave.open(self.file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            with self.lock:
                wf.writeframes(b''.join(self.frames))

recorder = None
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/start')
def start():
    global recorder
    if recorder is None or not recorder.recording:
        recorder = Recorder("recorded_audio.wav", app.config['UPLOAD_FOLDER'])
        threading.Thread(target=recorder.start_recording).start()
    return redirect('/')

@app.route('/stop')
def stop():
    global recorder
    if recorder is not None and recorder.recording:
        threading.Thread(target=recorder.stop_recording).start()
    return redirect('/')






@app.route('/analyse_live')
def analyse_live():

    audio_file = "uploads/recorded_audio.wav"
    prediction = analyze_audio_sentiment(audio_file)
    
    print(prediction)
    
    out = ""
    if prediction[0][0]>0.7:
        out = "The Sentiment detected is Not Offence "
    else:
        out = "The Sentiment detected is Offence"
    return render_template('output.html', sentiment_result=out)




if __name__ == '__main__':
    app.run(debug=False)