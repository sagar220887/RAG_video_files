import speech_recognition as sr
import google.generativeai as genai

from dotenv import load_dotenv
import os
from gtts import gTTS

audio_text_array = []

def get_voice_data_from_microphone():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print('Could not understand the audio source')
        return None
    except sr.RequestError as e:
        print('Exception  - ', e)
        return None


def get_voice_data_from_audio_file(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print('Could not understand the audio source')
        return None
    except sr.RequestError as e:
        print('Exception  - ', e)
        return None
    

def get_text_from_multiple_audio_files():
    folder_path = './chunked/'
    audio_files = os.listdir(folder_path)
    for each_file in audio_files:
        if ('.wav' in each_file):
            audio_text = get_voice_data_from_audio_file(os.path.join(folder_path, each_file))
            audio_text_array.append(audio_text)

    return " ".join(audio_text_array)



def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("speech.mp3")