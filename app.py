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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata['source']} (Page {doc.metadata['page']})\nContent: {doc.page_content}" for doc in docs)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context just say, "The answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{input}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | model | StrOutputParser()
    return chain

def text_to_speech(text):
    # Remove source citations from audio to make it sound natural
    clean_text = text.split("\n\n*Sources:")[0]
    tts = gTTS(clean_text, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process a PDF first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Extract unique sources for citation
    sources = set([f"{doc.metadata['source']} (Page {doc.metadata['page']})" for doc in docs])
    source_text = "\n\n*Sources: " + ", ".join(sources) + "*" if sources else ""

    chain = get_conversational_chain()
    
    response = chain.invoke({
        "context": format_docs(docs),
        "input": user_question
    })

    full_response = response + source_text
    audio_bytes = text_to_speech(full_response)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "audio": audio_bytes})

def main():
    st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")
    
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .main-header {
            font-size: 3rem;
            color: #4facfe;
            text-align: center;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .stButton>button {
            background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🧠 DocuMind AI</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your Intelligent Multi-Document Assistant with Voice & Citations</p>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("📚 Knowledge Base")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    docs = get_pdf_documents(pdf_docs)
                    text_chunks = get_text_chunks(docs)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload PDF files first.")
        
        st.markdown("---")
        st.subheader("⚙️ Options")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
        # Download Chat Button
        if st.session_state.chat_history:
            chat_text = "DocuMind AI - Chat History\n\n"
            for msg in st.session_state.chat_history:
                role = "You" if msg["role"] == "user" else "DocuMind AI"
                chat_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                label="💾 Download Chat",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    with col2:
        st.subheader("💬 Chat")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**🧑‍💻 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 DocuMind AI:** {message['content']}")
                if "audio" in message:
                    st.audio(message["audio"], format="audio/mp3")
                
        # Input options: Text or Voice
        c1, c2 = st.columns([5, 1])
        with c1:
            user_question = st.chat_input("Ask a question about your documents...")
        with c2:
            st.write("Or use voice:")
            voice_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
        
        # Process whichever input is provided
        if user_question:
            process_user_input(user_question)
            st.rerun()
        elif voice_text:
            process_user_input(voice_text)
            st.rerun()

if __name__ == "__main__":
    main()
