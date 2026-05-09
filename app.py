import streamlit as st
from PyPDF2 import PdfReader
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Use ONLY stable core imports to prevent ModuleNotFound errors entirely
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load Environment Variables
load_dotenv()
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{input}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Pure LCEL - Zero reliance on langchain.chains
    chain = prompt | model | StrOutputParser()
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if not os.path.exists("faiss_index"):
        st.error("Please upload and process a PDF first.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    # Invoke the chain by passing the formatted context and the user input
    response = chain.invoke({
        "context": format_docs(docs),
        "input": user_question
    })

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

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
    st.markdown("<p style='text-align: center;'>Your Intelligent Multi-Document Assistant</p>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("📚 Knowledge Base")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
            else:
                st.warning("Please upload PDF files first.")
        
        st.markdown("---")
        st.subheader("✨ Features")
        st.markdown("""
        - **Multi-PDF Support**: Chat with multiple documents at once.
        - **Chat History**: Keeps track of your conversation.
        - **Smart Embeddings**: Powered by Google Gemini AI.
        - **Dark Mode**: Eye-friendly UI.
        """)

    with col2:
        st.subheader("💬 Chat")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**🧑‍💻 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 DocuMind AI:** {message['content']}")
                
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            user_input(user_question)
            st.rerun()

if __name__ == "__main__":
    main()
