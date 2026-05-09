import streamlit as st
from PyPDF2 import PdfReader
import os
import io
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Environment Variables
load_dotenv()

def get_api_key():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")

api_key = get_api_key()
if api_key:
    api_key = api_key.strip().strip("'").strip('"')
    genai.configure(api_key=api_key)

def get_pdf_documents(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"source": pdf.name, "page": i + 1}))
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(docs)

def get_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=api_key)
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return False

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata['source']} (Page {doc.metadata['page']})\nContent: {doc.page_content}" for doc in docs)

def get_conversational_chain():
    prompt_template = """
    You are DocuMind AI, a helpful assistant. Use the context to answer the question.
    If not in context, answer using your general knowledge but clearly state so.
    Context: {context}
    Question: {input}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | model | StrOutputParser()

def get_summary(text):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3, google_api_key=api_key)
        chain = ChatPromptTemplate.from_template("Summarize this document in professional bullet points:\n\n{text}") | model | StrOutputParser()
        return chain.invoke({"text": text[:15000]})
    except Exception as e:
        return f"Summary Error: {str(e)}"

def text_to_speech(text):
    clean_text = text.split("\n\n*Sources:")[0]
    tts = gTTS(clean_text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=api_key)
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process a PDF first.")
        return False
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    sources = set([f"{doc.metadata['source']} (Page {doc.metadata['page']})" for doc in docs])
    source_text = "\n\n*Sources: " + ", ".join(sources) + "*" if sources else ""
    chain = get_conversational_chain()
    try:
        response = chain.invoke({"context": format_docs(docs), "input": user_question})
    except Exception as e:
        st.error(f"⚠️ API Error: {str(e)}")
        return False
    full_response = str(response) + source_text
    audio_bytes = text_to_speech(full_response)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "audio": audio_bytes})
    return True

def main():
    st.set_page_config(page_title="DocuMind AI Chatbot", page_icon="📄", layout="wide")
    
    # Matching the User's provided screenshot style (Clean White & Minimal)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
        
        * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stApp { background-color: #ffffff; }
        
        /* Sidebar styling to match the orange/white theme */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #dee2e6;
        }
        
        .sidebar-header {
            color: #d9480f; /* Orange color from screenshot */
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: uppercase;
        }

        /* Main Header matching the Green style */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2b8a3e; /* Green color from screenshot */
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #495057;
            margin-bottom: 1rem;
        }

        /* Chat styles matching the screenshot (No bubbles, plain text with colors) */
        .user-msg {
            color: #2b8a3e; /* Green for user */
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .bot-msg {
            color: #d9480f; /* Orange for chatbot */
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .chat-content {
            color: #343a40;
            margin-bottom: 20px;
            line-height: 1.6;
        }

        .summary-box {
            background-color: #f1f3f5;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #2b8a3e;
            margin-bottom: 20px;
        }

        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "raw_text" not in st.session_state: st.session_state.raw_text = ""
    if "doc_summary" not in st.session_state: st.session_state.doc_summary = ""

    # SIDEBAR
    with st.sidebar:
        st.markdown('<div class="sidebar-header">UPLOAD YOUR DOCUMENT HERE (PDF Only)</div>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload File", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")
        
        if st.button("🚀 Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    docs = get_pdf_documents(pdf_docs)
                    st.session_state.raw_text = " ".join([d.page_content for d in docs])
                    text_chunks = get_text_chunks(docs)
                    if get_vector_store(text_chunks):
                        st.success("Successfully Processed!")
            else: st.warning("Please upload a PDF")

        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # MAIN AREA
    st.markdown('<div class="main-header">RAG Based CHATBOT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Follow the steps to use this application:</div>', unsafe_allow_html=True)
    st.markdown("""
    - Upload your PDF document in the sidebar.
    - Write your query and start chatting with the bot.
    """)
    st.markdown("---")

    # Summary Button (Centered in main area as requested)
    if st.session_state.raw_text:
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s2:
            if st.button("✨ Generate Summary"):
                with st.spinner("Summarizing..."):
                    st.session_state.doc_summary = get_summary(st.session_state.raw_text)
        
        if st.session_state.doc_summary:
            st.markdown(f'<div class="summary-box"><b>📝 Document Summary:</b><br>{st.session_state.doc_summary}</div>', unsafe_allow_html=True)

    # Chat Display matching the screenshot plain text style
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-msg">User: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">Chatbot:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-content">{message["content"]}</div>', unsafe_allow_html=True)
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # Chat Input & Voice
    voice_text = speech_to_text(language='en', start_prompt="🎤 Speak to Chat", stop_prompt="⏹️ Stop", key='STT')
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        if process_user_input(user_question): st.rerun()
    elif voice_text:
        if process_user_input(voice_text): st.rerun()

if __name__ == "__main__":
    main()
