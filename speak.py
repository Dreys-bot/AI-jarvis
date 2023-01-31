import numpy as np
import nltk #for NLP
from nltk.stem.porter import PorterStemmer
import pyttsx3
import speech_recognition as sr
import datetime

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    print(f'A.I : {audio}')
    engine.say(audio)
    engine.runAndWait()

