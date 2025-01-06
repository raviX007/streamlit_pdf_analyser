# PDF RAG Chat Assistant

An intelligent document question-answering system that uses Retrieval Augmented Generation (RAG) to provide accurate responses to questions about PDF content. Built with LangChain, OpenAI, and Chroma vector store.

## 🌟 Features

- 📄 PDF text extraction and processing
- 💡 Smart text chunking with overlap
- 🔍 Advanced RAG implementation
- 💬 Interactive chat interface
- 🎯 Source attribution for answers
- 🔄 Conversation history tracking
- 🧹 Automatic text cleaning

## 🏗️ Architecture

![alt text](<Screenshot 2025-01-06 at 6.22.29 AM.png>)

## 🛠️ Technical Stack

- **LangChain**: RAG implementation and chain orchestration
- **OpenAI**: Embeddings and chat completion
- **ChromaDB**: Vector store for document embeddings
- **Streamlit**: Interactive web interface
- **PyPDF2**: PDF processing

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-rag-chat-assistant.git
cd pdf-rag-chat-assistant
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create requirements.txt with:

```
streamlit
langchain
openai
chromadb
pypdf2
```

## 🔑 Configuration

Required:

- OpenAI API key
- PDF document(s) for processing

## 🚀 Usage

1. Start the application:

```bash
streamlit run app.py
```

2. In the web interface:
   - Enter your OpenAI API key
   - Upload a PDF document
   - Wait for processing to complete
   - Start asking questions about your document

## 💡 Key Components

### PDF Processing

- Text extraction from PDFs
- Text cleaning and normalization
- Chunk creation with overlap
- Document metadata handling

### RAG Implementation

- OpenAI embeddings generation
- Chroma vector store integration
- Semantic search functionality
- Source document retrieval

### Chat Interface

- Conversation history tracking
- Interactive Q&A
- Source attribution display
- Error handling

## 📊 Project Structure

```
pdf-rag-chat-assistant/
├── app.py              # Main application
├── chroma_db/          # Vector store data
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── .gitignore         # Git ignore file
```

## ⚠️ Limitations

- Requires OpenAI API key
- Text-based PDF processing only
- Memory-based conversation history
- Processing time depends on PDF size
- Rate limits based on OpenAI API

## Screenshot of the working Application

![alt text](<Screenshot 2025-01-06 at 6.24.08 AM.png>)
