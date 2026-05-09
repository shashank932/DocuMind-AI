# 🧠 DocuMind AI

DocuMind AI is an intelligent, multi-document assistant powered by Google Gemini AI. It allows you to upload multiple PDF documents and ask questions directly from the content. The system uses RAG (Retrieval-Augmented Generation) to fetch the most relevant context and formulate highly accurate, contextual answers.

## ✨ Features

- **Multi-PDF Support:** Upload and process multiple PDF files simultaneously.
- **Persistent Chat History:** Keeps track of your conversation context.
- **Smart Embeddings:** Leverages state-of-the-art Google Gemini AI embeddings and FAISS for rapid document retrieval.
- **Modern UI:** Built with an eye-friendly Dark Mode and responsive layout using Streamlit.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shashank932/DocuMind-AI.git
   cd DocuMind-AI
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY="your_google_api_key_here"
   ```

### Running the Application

To start the DocuMind AI interface, run:
```bash
streamlit run app.py
```

## 🛠 Built With
- [Streamlit](https://streamlit.io/) - The web framework used.
- [LangChain](https://python.langchain.com/) - Application framework for LLMs.
- [Google Gemini API](https://ai.google.dev/) - Large Language Model.
- [FAISS](https://faiss.ai/) - Vector database for fast similarity search.
- [PyPDF2](https://pypi.org/project/PyPDF2/) - PDF parsing.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
