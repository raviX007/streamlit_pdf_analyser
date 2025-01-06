import streamlit as st
from PyPDF2 import PdfReader
import io
import os
import logging
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Enhanced text cleaning function"""
    cleaned = ''.join(char if ord(char) < 128 else ' ' for char in text)
    return ' '.join(cleaned.split())

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        try:
            pdf_reader = PdfReader(pdf_file)
            text = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(clean_text(page_text))
            return ' '.join(text)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    @staticmethod
    def create_documents(text: str) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        documents = []
        for i, chunk in enumerate(chunks):
            cleaned_chunk = clean_text(chunk)
            if cleaned_chunk.strip():
                doc = Document(page_content=cleaned_chunk, metadata={"chunk": i})
                documents.append(doc)
        logger.info(f"Created {len(documents)} documents")
        return documents

def process_pdf(uploaded_file, api_key: str) -> Chroma:
    try:
        processor = PDFProcessor()
        with st.spinner("Extracting text from PDF..."):
            file_text = processor.extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
            if not file_text:
                st.error("No text could be extracted from the PDF.")
                st.stop()
            st.success("Text extracted successfully")
            logger.info("PDF text extraction completed")

            documents = processor.create_documents(file_text)
            if not documents:
                st.error("No valid text chunks could be created from the PDF.")
                st.stop()
            st.success("Text processed successfully")
            logger.info(f"Created {len(documents)} documents")

            with st.spinner("Creating embeddings and vector store..."):
                embedding_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",  # This is the default model
                openai_api_key=api_key
                )
                persist_directory = "chroma_db"
                os.makedirs(persist_directory, exist_ok=True)

                client = chromadb.PersistentClient(path=persist_directory)
                vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_model,
                    client=client,
                    collection_name="pdf_qa"
                )
                
                st.success("Vector store created successfully")
                logger.info("Chroma vector store creation completed")
                return vector_store

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None

def initialize_conversation_chain(vector_store: Chroma, api_key: str) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def main():
    st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    st.title("PDF RAG Chat Assistant")
    st.write("Upload a PDF and ask questions about its content!")

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file and api_key:
        try:
            if st.session_state.vector_store is None:
                vector_store = process_pdf(uploaded_file, api_key)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.conversation_chain = initialize_conversation_chain(vector_store, api_key)
        except Exception as e:
            logger.error(f"Setup error: {str(e)}")
            st.error(f"Error: {str(e)}")

    if st.session_state.conversation_chain is not None:
        question = st.chat_input("Ask a question about your PDF:")
        
        if question:
            with st.spinner("Thinking..."):
                try:
                    logger.info("Processing question")
                    cleaned_question = clean_text(question)
                    
                    response = st.session_state.conversation_chain({
                        "question": cleaned_question,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = clean_text(response["answer"])
                    source_docs = response["source_documents"]
                    
                    st.session_state.chat_history.extend([
                        HumanMessage(content=cleaned_question),
                        AIMessage(content=answer)
                    ])
                    
                    st.session_state.messages.append({"role": "user", "content": cleaned_question})
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        with st.expander("View sources"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(clean_text(doc.page_content))
                                st.divider()

                except Exception as e:
                    logger.error(f"Error in chat response: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.exception("Full traceback:")
                    st.error(f"An error occurred while generating the response: {str(e)}")

        for message in st.session_state.messages[:-2]:
            try:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(clean_text(message["content"]))
                else:
                    with st.chat_message("assistant"):
                        st.markdown(clean_text(message["content"]))
            except Exception as e:
                logger.error(f"Error displaying message: {str(e)}")
                continue

    else:
        st.info("Please upload a PDF and enter your OpenAI API key to start asking questions!")

if __name__ == "__main__":
    main()