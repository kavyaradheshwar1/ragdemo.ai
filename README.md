# ragdemo.ai
Retrieval Augmented Generation(RAG) is a technique that enhances the capabilities of LLMs by combining information retrieval with text generation. Instead of relying on pre-trained knowledge, RAG fetch relevant data from external sources and use it to generate more accurate responses.
### Packages
streamlit
python-dotenv
PyPDF2
google-generativeai

langchain  # core framework
langchain-huggingface #connect huggingface models to perform embedding
faiss-cpu #fast  vector db to store embedded data
langchain-community #extra integration
langchain-text-splitters #to split large data into smaller chunks
sentence-transformers #pretrained embedding models to convert text to vectors
langchain-core #document,chain,etc..

text -> split text -> convert vector -> store in db -> search similar content -> send to LLM -> Get answers for questions

'all-MiniLLM-L6-v2' -> simple hugging face model which splits the text and converts the text into vectors