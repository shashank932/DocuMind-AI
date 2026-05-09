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
        st.error(f"⚠️ Google API Error: Your API key might be invalid or expired. Please check your key! (Error details: {str(e)})")
        return False

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata['source']} (Page {doc.metadata['page']})\nContent: {doc.page_content}" for doc in docs)

def get_conversational_chain():
    prompt_template = """
    You are DocuMind AI, a professional document assistant. Use the provided context to answer the user's question accurately.
    
    1. If the answer is present in the context, provide a detailed response and cite the source.
    2. If the answer is NOT in the context, use your general knowledge to answer helpfully, but mention that the info is from your general knowledge.
    
    Context:
    {context}
    
    Question: 
    {input}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3, google_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    return chain

def get_summary(text):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3, google_api_key=api_key)
        response = model.invoke(f"Please provide a concise and professional summary of the following text in 3-4 bullet points:\n\n{text[:15000]}")
        return response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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
        response = chain.invoke({
            "context": format_docs(docs),
            "input": user_question
        })
    except Exception as e:
        st.error(f"⚠️ Google API Error while answering: (Error details: {str(e)})")
        return False

    full_response = response + source_text
    audio_bytes = text_to_speech(full_response)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "audio": audio_bytes})
    return True

def main():
    st.set_page_config(page_title="DocuMind AI - Intelligent PDF Assistant", page_icon="🧠", layout="wide")
    
    if not api_key:
        st.error("🚨 Google Gemini API Key is missing! Please add it in the Streamlit Advanced Settings -> Secrets.")
        st.stop()
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        
        * { font-family: 'Outfit', sans-serif; }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
        }
        
        .hero-section {
            background: rgba(255, 255, 255, 0.05);
            padding: 3rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }
        
        .main-title {
            font-size: 4rem;
            background: linear-gradient(to right, #60a5fa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        
        .sidebar-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1rem;
        }
        
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            padding: 0.6rem;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <h1 class="main-title">🧠 DocuMind AI</h1>
            <p style="font-size: 1.2rem; color: #94a3b8; max-width: 800px; margin: 0 auto;">
                The ultimate intelligent document assistant. Upload multiple PDFs and chat with them using 
                state-of-the-art Generative AI, Citations, and Voice interaction.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    col1, col2 = st.columns([1, 2.5], gap="large")

    with col1:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("📚 Knowledge Base")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        if st.button("🚀 Process Documents"):
            if pdf_docs:
                with st.spinner("Analyzing documents..."):
                    docs = get_pdf_documents(pdf_docs)
                    st.session_state.raw_text = " ".join([d.page_content for d in docs])
                    text_chunks = get_text_chunks(docs)
                    success = get_vector_store(text_chunks)
                    if success:
                        st.success("Docs Ready!")
            else:
                st.warning("Upload PDFs first.")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.raw_text:
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            st.subheader("📝 Smart Insights")
            if st.button("✨ Quick Summary"):
                with st.spinner("Summarizing..."):
                    summary = get_summary(st.session_state.raw_text)
                    st.info(summary)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.subheader("⚙️ Actions")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.session_state.chat_history:
            chat_text = "DocuMind AI History\n\n"
            for msg in st.session_state.chat_history:
                chat_text += f"{msg['role'].upper()}: {msg['content']}\n\n"
            st.download_button("💾 Export Chat", chat_text, "chat.txt")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "audio" in message:
                        st.audio(message["audio"], format="audio/mp3")
                
        # Input row
        c1, c2 = st.columns([4, 1])
        with c1:
            user_question = st.chat_input("Ask anything...")
        with c2:
            voice_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        
        if user_question:
            if process_user_input(user_question):
                st.rerun()
        elif voice_text:
            if process_user_input(voice_text):
                st.rerun()

if __name__ == "__main__":
    main()
