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
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
from langchain_core.documents import Document

OPENAI_API_KEY="<<>>"

def get_text_from_pdf(pdf_file):
    text=""
    for page in pdf_file.pages:
        text+=page.extract_text()
    return text

def create_langchain_document(documents, chunks, ticker):
    for i in range(0, len(chunks)-1):
         document = Document(
            page_content=chunks[i], metadata={"ticker" : ticker}
        )
    documents.append(document)

st.header("Theme Machine")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

documents = []

if(st.session_state.vector_store is None):
    print('Initializing vector store')
    pdf_file1 = PdfReader('./cvx.pdf')
    text1 = get_text_from_pdf(pdf_file1)

    pdf_file2 = PdfReader('./tsla.pdf')
    text2 = get_text_from_pdf(pdf_file2)

    pdf_file3 = PdfReader('./shc.pdf')
    text3 = get_text_from_pdf(pdf_file3)

    text_splitter = RecursiveCharacterTextSplitter(separators="\n",
                                chunk_size=1000,
                                chunk_overlap=150,
                                length_function=len
                                )
    chunks1 = text_splitter.split_text(text1)
    chunks2 = text_splitter.split_text(text2)
    chunks3 = text_splitter.split_text(text3)

    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    create_langchain_document(documents, chunks1, "cvx")
    create_langchain_document(documents, chunks2, "tsla")
    create_langchain_document(documents, chunks3, "shc")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents, ids=uuids)
    st.session_state.vector_store=vector_store
    print('Vector store initialized')
else:
    print('Using Vector store from session')
    vector_store=st.session_state.vector_store

# What are the themes relating to this company show as bullet points with maximum three words per bullet
query=st.text_input("Enter your question")
company = st.selectbox(
    "Select the company relevant for your search",
    ("All", "Tesla", "Sotera", "Chevron"),
)


if query:
    
    if company == "Tesla":
        filter = {"ticker":"tsla"}
    elif company == "Sotera":
        filter = {"ticker":"shc"}
    elif company == "Chevron":
        filter = {"ticker":"cvx"}
    else:
        filter = {}
        
    print('Filter:' + str(filter))

    match=vector_store.similarity_search(query=query, filter=filter)
    result_tickers = set()
    for doc in match:
        print('Match Doc: -->' + doc.page_content + '<--')
        print('Match Tic: ==>' + doc.metadata["ticker"] + '<==')
        result_tickers.add(doc.metadata["ticker"])
    print('Results found in the following tickers:')
    print(result_tickers)
    
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
    with st.container():
        st.write("The index related to this search would conist of:")
        st.write(result_tickers)
        
# Prompt1: What are the themes relating to this company bulleted as single words
# What are the sectors relating to this company bulleted as single words
