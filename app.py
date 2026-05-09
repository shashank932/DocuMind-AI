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
    You are DocuMind AI, a professional assistant. Answer helpfully using the context provided.
    If not in context, answer from your general knowledge but mention it.
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
        response = model.invoke(f"Summarize this text in 3-4 professional bullet points:\n\n{text[:15000]}")
        return response.content
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
    full_response = response + source_text
    audio_bytes = text_to_speech(full_response)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "audio": audio_bytes})
    return True

def main():
    st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")
    if not api_key:
        st.error("🚨 API Key Missing!")
        st.stop()
    
    # Advanced ChatGPT-like CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #212121;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #171717 !important;
            width: 260px !important;
        }

        /* Center Content like ChatGPT */
        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-bottom: 10rem;
        }

        .main-header {
            font-size: 2rem;
            font-weight: 600;
            color: #ececf1;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Chat Message Styling */
        .stChatMessage {
            background-color: transparent !important;
            border: none !important;
            padding: 1.5rem 0 !important;
        }
        
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: #2f2f2f10 !important;
        }

        /* Hide Streamlit elements for cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Voice button subtle styling */
        .voice-container {
            position: fixed;
            bottom: 85px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    # SIDEBAR
    with st.sidebar:
        st.markdown("<h2 style='color: white; margin-bottom: 20px;'>DocuMind AI</h2>", unsafe_allow_html=True)
        
        with st.expander("📁 Upload Documents", expanded=True):
            pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'], label_visibility="collapsed")
            if st.button("🚀 Process", use_container_width=True):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        docs = get_pdf_documents(pdf_docs)
                        st.session_state.raw_text = " ".join([d.page_content for d in docs])
                        text_chunks = get_text_chunks(docs)
                        if get_vector_store(text_chunks):
                            st.success("Ready!")
                else:
                    st.warning("Upload PDF")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.raw_text:
            if st.button("✨ Summarize", use_container_width=True):
                with st.spinner("Summarizing..."):
                    summary = get_summary(st.session_state.raw_text)
                    st.info(summary)
        
        st.markdown("<div style='position: fixed; bottom: 20px; width: 220px;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # MAIN AREA
    st.markdown('<div class="main-header">DocuMind AI</div>', unsafe_allow_html=True)

    # Chat Messages in centered container
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # Fixed Voice Button above Chat Input
    st.markdown('<div class="voice-container">', unsafe_allow_html=True)
    voice_text = speech_to_text(language='en', start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", key='STT')
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat Input (Anchored at bottom, centered automatically by block-container max-width)
    user_question = st.chat_input("Ask anything...")
    
    if user_question:
        if process_user_input(user_question):
            st.rerun()
    elif voice_text:
        if process_user_input(voice_text):
            st.rerun()

if __name__ == "__main__":
    main()
