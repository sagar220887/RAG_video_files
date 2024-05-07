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

    # Saving the audio file contents into a text file
    merged_text = " ".join(audio_text_array)
    save_audio_to_text_file(merged_text)
    return read_audio_text_file()


def save_audio_to_text_file(content):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(ROOT_DIR,'audio.txt')
    audio_text_file = open(audio_path,"w+")
    audio_text_file.write(content)
    audio_text_file.close()

def read_audio_text_file():
    # Read the audio text file
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(ROOT_DIR,'audio.txt')
    file1 = open(audio_path, "r+")
    file_content = file1.read()
    file1.close()
    return file_content


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    tts.save("speech.mp3")