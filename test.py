import streamlit as st
from src.audio_to_text import *
import speech_recognition as sr
import google.generativeai as genai

from dotenv import load_dotenv
import os
from gtts import gTTS

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers

import streamlit as st
import os
import time
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import UnstructuredURLLoader, MergedDataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import HuggingFaceHub
from transformers import  AutoTokenizer
from ctransformers import AutoModelForCausalLM

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_API_KEY

VECTOR_DB_DIRECTORY = "./model/vectordb"

audio_text_array = []
def get_voice_data_from_audio_file(audio_file):
    print('Parsing audio file ==> ' + audio_file)
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        print('Data retrieved ==> ', text)
        return text
    except sr.UnknownValueError:
        print('Could not understand the audio source')
        return ''
    except sr.RequestError as e:
        print('Exception  - ', e)
        return ''
    

def get_text_from_multiple_audio_files():
    folder_path = './chunked/'
    audio_files = os.listdir(folder_path)
    for each_file in audio_files:
        if ('.wav' in each_file):
            audio_text = get_voice_data_from_audio_file(os.path.join(folder_path, each_file))
            audio_text_array.append(audio_text)

    return " ".join(audio_text_array)


def get_documnets_from_text_file(text_file):
    loader = TextLoader(text_file)
    data = loader.load()
    print('get_documnets_from_text_file ==>', data)
    return data

def get_data_chunks(data):
    recursive_char_text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)
    documents=recursive_char_text_splitter.split_documents(data)
    # print('documents - ', documents)
    print('get_data_chunks :: documents type - ', type(documents))
    print('get_data_chunks :: documents length - ', len(documents))
    return documents



def create_embeddings():
    embeddings=HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', 
            model_kwargs={'device':'cpu'}
    )
    return embeddings



def load_vectordb(stored_directory, embeddings):
    loaded_vector_db = FAISS.load_local(stored_directory, embeddings)
    return loaded_vector_db

def store_data_in_vectordb(documents, embeddings):
    existing_knowledge_base = None
    try:
        existing_knowledge_base = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)
    except :
        print('Error in loading the existing vector DB')

    new_knowledge_base =FAISS.from_documents(documents, embeddings)

    if existing_knowledge_base:
        print('Merging the new data base into the existing knowledge base')
        existing_knowledge_base.merge_from(new_knowledge_base)
        existing_knowledge_base.save_local(VECTOR_DB_DIRECTORY)
    else:
        print('Saving the new data base')
        new_knowledge_base.save_local(VECTOR_DB_DIRECTORY)

    final_loaded_knowledge_base = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)   

    return final_loaded_knowledge_base



def get_prompt():
    template="""Use the following pieces of information to answer the user's question.
            If you dont know the answer just say you dont know, don't try to make up an answer.

            Context:{context}
            Question:{question}

            Only return the helpful answer below and nothing else
            Helpful answer
            """
    

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    print('Prompt created')
    return prompt


def create_chain(llm, vector_store, prompt):
    chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': prompt}
    )
    print('Chain created')
  

def llm_model():
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name = "gemini-pro")
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.2,convert_system_message_to_human=True)
    return model

def get_response(chain):
    # response = model.generate_content(user_query)
    # result = response.text
    # print('result - ', result)
    # return result
    result = chain({'query':'tell me a summary of the file context'}, return_only_outputs=True)
    print('result - ', result)
    ans = result['result']
    print(f"Answer:{ans}")
    return ans


def main():

    '''
    converted video to audio file format - 
    1. recording the video file and converting to audio wave file
    2. The large audio wave file is converted into small chunks of audio wave files
    3. The below script is executed and the video is played
    4. when the user wants to stop as per keyboard interuption - the file is saved as audio wave file
    5. The audio wave file is converted into many chunks of the same audio wave
    6. The files are saved under the directory - chunked which is input to below code

    script : src/python voice_recorder.py

    '''
    

    audio_text_file = get_text_from_multiple_audio_files()
    print('audio_text_file ==> ', audio_text_file)

    documents = get_documnets_from_text_file(audio_text_file)

    # #Split Text into Chunks
    data_chunks = get_data_chunks(documents)

    # #Load the Embedding Model
    embeddings = create_embeddings()

    # #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    vector_db=store_data_in_vectordb(data_chunks, embeddings)
    
    llm = llm_model()

    qa_prompt = get_prompt()

    chain = create_chain(llm, vector_db, qa_prompt)

    get_response(qa_prompt)

    



if __name__ == "__main__":
    main()