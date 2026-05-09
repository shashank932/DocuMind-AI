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
        st.error(f"⚠️ Google API Error: {str(e)}")
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
        st.error("🚨 API Key Missing in Secrets!")
        st.stop()
    
    # ChatGPT-like CSS Styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #212121; color: #ececf1; }
        .stSidebar { background-color: #171717 !important; border-right: 1px solid #333; }
        .main-header { font-size: 2.5rem; font-weight: 700; text-align: center; margin-top: 1rem; color: #4facfe; }
        .stChatMessage { border-radius: 15px; margin-bottom: 1rem; }
        .stChatInput { position: fixed; bottom: 20px; }
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""

    # SIDEBAR: ALL OPTIONS HERE
    with st.sidebar:
        st.title("⚙️ Control Panel")
        st.markdown("---")
        
        st.subheader("📁 Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if st.button("🚀 Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    docs = get_pdf_documents(pdf_docs)
                    st.session_state.raw_text = " ".join([d.page_content for d in docs])
                    text_chunks = get_text_chunks(docs)
                    if get_vector_store(text_chunks):
                        st.success("Ready!")
            else:
                st.warning("Select PDF")

        if st.session_state.raw_text:
            st.markdown("---")
            st.subheader("✨ Intelligence")
            if st.button("📝 Summarize Document"):
                with st.spinner("Writing summary..."):
                    summary = get_summary(st.session_state.raw_text)
                    st.info(summary)
        
        st.markdown("---")
        st.subheader("🛠️ Actions")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.session_state.chat_history:
            chat_export = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button("💾 Export Chat", chat_export, "chat.txt")

    # MAIN AREA: CHAT ONLY
    st.markdown('<div class="main-header">🧠 DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8e8ea0;'>Intelligent RAG Assistant</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Native Chat Interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # Input fixed at bottom
    c1, c2 = st.columns([6, 1])
    with c1:
        user_question = st.chat_input("How can I help you today?")
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
