import pyaudio
import wave
from pydub import AudioSegment 
from pydub.utils import make_chunks
import os

audio = pyaudio.PyAudio()

AUDIO_FORMAT = pyaudio.paInt16
NO_OF_AUDIO_CHANNEL=1
STREAM_RATE = 44100
FRAMES_PER_BUFFER = 1024

OUTPUT_FILENAME = "output.wav"
frames = []

def get_stream():
    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=NO_OF_AUDIO_CHANNEL,
        rate=STREAM_RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
    )
    return stream

def generate_audio_file(stream, filename):
    try:
        print('stopping stream recording')
        stream.stop_stream()
        stream.close()
        audio.terminate()
    except:
        print('Exception in record_audio')

    print('generate_audio_file =================================')
    output_wf = wave.open(filename, "wb")
    output_wf.setnchannels(NO_OF_AUDIO_CHANNEL)
    output_wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
    output_wf.setframerate(STREAM_RATE)
    output_wf.writeframes(b"".join(frames))
    output_wf.close()
    return output_wf



def record_audio():
    try:
        stream = get_stream()
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        pass

    return generate_audio_file(stream, OUTPUT_FILENAME)


def process_large_audio_file(file_name):
    myaudio = AudioSegment.from_file(file_name, "wav") 
    chunk_length_ms = 2000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        chunk_name = './chunked/' + file_name + "_{0}.wav".format(i) 
        print ("exporting", chunk_name) 
        chunk.export(chunk_name, format="wav")



def split_audio_file():

    all_file_names = os.listdir()
    try:
        os.makedirs('chunked') # creating a folder named chunked
    except:
        pass
    for each_file in all_file_names:
        if ('.wav' in each_file):
            process_large_audio_file(each_file)



if __name__ == "__main__":
    record_audio()
    split_audio_file()
    






