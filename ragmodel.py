import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings # to get embedding model
from langchain_core.documents import Document # to store text and metadata
from langchain_text_splitters import CharacterTextSplitter # to split large text into smaller chunks
from langchain_community.vectorstores import FAISS # to store embeddings for similarity search

key = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash')

def load_embedding():
    return HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

st.set_page_config('RAG ASSISTANT')

st.title('RAG Assistant :blue[Using Embedding and LLM]🎯')
st.subheader(':green[Your Intelligent Document Assistant 📌👤]')

with st.spinner('Loading embedding model...'):
    embedding_model = load_embedding()
    
uploaded_file = st.file_uploader('Upload the document here in PDF Format', type = ['pdf'])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    
    raw_text = ''
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    if raw_text.strip(): # ensures the pdf is not empty, removes spaces and check any content exist
        doc = Document(page_content=raw_text)
        
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        # max char in each chunk = 1000
        # overlap to maintains between context = 200
        chunk_text = splitter.split_documents([doc]) # splits the document into multiple smaller chunks
        
        text = [i.page_content for i in chunk_text]
        # converts the chunks into simple text in list
        
        vector_db = FAISS.from_texts(text, embedding_model)
        retrive = vector_db.as_retriever()
        
        st.success('Document processed an saved successfully!!! Ask your question')
        
        query = st.text_input('Enter your query here: ')
        
        if query:
            with st.chat_message('human'): # display msg n chat format
                with st.spinner('Analyzing the documents.....'):
                    relevant_data = retrive.invoke(query) # find the most similar text chunks using FAISS
                    
                    content = '\n\n'.join([i.page_content for i in relevant_data])
                    # merges all relevent chunk into one text
                    
                    prompt = f'''
                    You are on AI expert. Use the content generated to answer the
                    query asked by the user. If you are unsure, you should say
                    'I am unsure about question asked'
                    
                    content : {content}
                    Query : {query}
                    Result :
                    '''
                    
                    response = model.generate_content(prompt)
                    
                    st.markdown('### :green[Result]')
                    st.write(response.text)
    
    else:
        st.warning('Drop the file in PDF format')