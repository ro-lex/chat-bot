# Upload file
# Convert PDF file to text - use PDF Reader
# Read the contents of PDF file 
# iterate through pages and read text context
# Create chunks from text - from langchain.text_splitter - RecursiveCharacterTextSplitter
# Create Embeddings - uses Open AI. Needs Key. from langchain_community.embeddings.openai - OpenAIEmbeddings
# Create vector stores using chunks and embeddings. from langchain_community.vectorstores - FAISS
# Capture user input
# Perform similarity search in vector store - semantic search
# Chain -> query, matches, Ask Model to create response out of similarity. 
# from langchain.chains.question_answering import load_qa_chain
# llm - from langchain.community.chat_models import ChatOpenAI


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY=""

st.header("Chat Bot")

with st.sidebar:
    st.write("Your documents")
    file = st.file_uploader("Upload your document", type="pdf")
    
if file is not None:
    pdf_file = PdfReader(file)
    
    text=""
    for page in pdf_file.pages:
        text+=page.extract_text()
    #st.write(text)
    
    text_splitter = RecursiveCharacterTextSplitter(separators="\n",
                                   chunk_size=1000,
                                   chunk_overlap=150,
                                   length_function=len
                                   )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)
    
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #st.write(embeddings)
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    #st.write(vector_store)
    
    # What are the themes relating to this company show as bullet points with maximum three words per bullet
    query=st.text_input("Enter your question")
    
    if query:
        match=vector_store.similarity_search(query=query)
        #st.write(match)
        
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )
        
        chain = load_qa_chain(llm, chain_type="stuff")
        results = chain.run(input_documents=match, question=query)
        print(results)
        st.write(results)
        
        
    