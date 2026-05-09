# 🧠 DocuMind AI: Intelligent Multi-Document Assistant

DocuMind AI is a state-of-the-art **Retrieval-Augmented Generation (RAG)** application that allows users to have natural conversations with multiple PDF documents. Built with **LangChain**, **Google Gemini AI**, and **Streamlit**, it features advanced capabilities like Citations, Voice Interaction, and Automated Summarization.

## 🚀 Live Demo
Check out the live application here: **[https://8eownyduk5mz4gnnuyjfsu.streamlit.app/](https://8eownyduk5mz4gnnuyjfsu.streamlit.app/)**

## ✨ Key Features
- **Multi-PDF Support**: Upload and chat with multiple documents simultaneously.
- **Hybrid Intelligence**: Answers questions based on document context AND general world knowledge.
- **Voice Interaction**: Integrated Speech-to-Text (Voice Input) and Text-to-Speech (Audio Output).
- **Source Citations**: Automatically cites the exact page number and document name for every answer.
- **Smart Summary**: Generate a professional summary of entire documents with a single click.
- **Premium UI**: Modern dark-mode interface with glassmorphism design and responsive layout.
- **Chat Persistence**: Maintain chat history during the session with export options.

## 🛠️ Tech Stack
- **Frontend**: Streamlit (Python)
- **AI Framework**: LangChain (LCEL)
- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Google Gemini Embedding 2
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyPDF2
- **Voice Logic**: gTTS (Google Text-to-Speech) & Streamlit Mic Recorder

## 📋 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shashank932/DocuMind-AI.git
   cd DocuMind-AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Key**
   Create a `.env` file and add your Google API Key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🛡️ Privacy & Security
All processing is done via secure API calls to Google Generative AI. Documents are processed into local vector stores and are not stored permanently.

---
Developed with ❤️ by **Shashank**
