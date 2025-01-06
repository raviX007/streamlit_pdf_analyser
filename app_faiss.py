import streamlit as st
from PyPDF2 import PdfReader
import re
import os
import tempfile
import io
from typing import List, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text"""
        # Basic cleaning - only keep ASCII characters
        cleaned_text = ''
        for char in text:
            if ord(char) < 128:  # Keep only ASCII characters
                cleaned_text += char
            else:
                cleaned_text += ' '  # Replace non-ASCII with space
        
        # Normalize whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text

    @staticmethod
    def create_documents(text: str, batch_size: int = 100) -> List[Document]:
        """Split text into chunks and create documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        chunks = text_splitter.split_text(text)
        documents = []
        logger.info(f"Created {len(chunks)} text chunks")
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "chunk": i,
                        "source": f"chunk-{i}",
                    }
                )
                documents.append(doc)

        return documents

def process_pdf(uploaded_file, api_key: str) -> FAISS:
    """Process PDF and create FAISS vector store"""
    try:
        processor = PDFProcessor()
        batch_size = 100  # Adjust based on your needs
        
        with st.spinner("Extracting text from PDF..."):
            file_text = processor.extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
            if not file_text:
                st.error("No text could be extracted from the PDF.")
                st.stop()
            st.success("Text extracted successfully")
            logger.info("PDF text extraction completed")

        with st.spinner("Processing text..."):
            cleaned_text = processor.clean_text(file_text)
            documents = processor.create_documents(cleaned_text)
            if not documents:
                st.error("No valid text chunks could be created from the PDF.")
                st.stop()
            st.success("Text processed successfully")
            logger.info(f"Created {len(documents)} documents")

        with st.spinner("Creating embeddings and vector store..."):
            embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
            
            # Initialize FAISS with the first batch
            first_batch = documents[:batch_size]
            vector_store = FAISS.from_documents(
                first_batch, 
                embedding_model, 
                distance_strategy=DistanceStrategy.COSINE
            )
            
            # Add remaining documents in batches
            remaining_docs = documents[batch_size:]
            for i in range(0, len(remaining_docs), batch_size):
                batch = remaining_docs[i:i+batch_size]
                vector_store.add_documents(batch)
                
            st.success("Vector store created successfully")
            logger.info("FAISS vector store creation completed")

        return vector_store

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")
        return None

def initialize_chat_chain(vector_store: FAISS, api_key: str) -> ConversationalRetrievalChain:
    """Initialize the chat chain"""
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
    )

def main():
    st.set_page_config(page_title="PDF Q&A Assistant", page_icon=None, layout="wide")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chain" not in st.session_state:
        st.session_state.chain = None

    st.title("PDF RAG Chat Assistant")
    st.write("Upload a PDF and ask questions about its content!")

    # Sidebar for API key and file upload
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
                        st.session_state.chain = initialize_chat_chain(vector_store, api_key)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Main chat interface
    if st.session_state.chain is not None:
        question = st.chat_input("Ask a question about your PDF:")
        
        if question:
            with st.spinner("Thinking..."):
                try:
                    # Get response from chain
                    response = st.session_state.chain(
                        {"question": question, "chat_history": st.session_state.chat_history}
                    )
                    
                    # Update chat history
                    answer = response["answer"]
                    sources = response["source_documents"]
                    
                    st.session_state.chat_history.append(HumanMessage(content=question))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                    # Add to messages for display
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    # Show sources in an expander
                    try:
                        with st.expander("View sources"):
                            for doc in sources:
                                st.markdown(f"""
                                **Chunk {doc.metadata['chunk']}**
                                {doc.page_content[:200]}...
                                """)
                                st.divider()
                    except:
                        pass
    else:
        st.info("Please upload a PDF and enter your OpenAI API key to start asking questions!")

if __name__ == "__main__":
    main()