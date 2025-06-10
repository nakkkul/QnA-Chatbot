import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import tempfile

# -- Configuration ------------------------------------------------------------
load_dotenv()
g = st.secrets.get("GROQ", {})
GROQ_API_KEY = g.get("api_key") or os.getenv("GROQ_API_KEY")
GROQ_MODEL   = g.get("model")   or os.getenv("GROQ_MODEL", "Gemma2-9b-It")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing")
    st.stop()

# -- Initialize components -----------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_llm():
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

@st.cache_resource(show_spinner=False)
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -- Streamlit UI --------------------------------------------------------------
st.set_page_config(page_title="📄🔍 Chatbot with Groq", page_icon="🤖", layout="wide")
st.header("🤖 AI Assistant for Conversational PDF Q&A")

# Session state for messages and PDF retriever
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pdf_retriever" not in st.session_state:
    st.session_state["pdf_retriever"] = None

# Instantiate LLM, embeddings, memory
llm = init_llm()
embeddings = init_embeddings()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Sidebar: PDF upload
with st.sidebar:
    st.subheader("Upload a PDF to chat 👇")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if pdf_file:
        # Save uploaded file to a temp location for loader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load_and_split()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state["pdf_retriever"] = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("PDF indexed! Now ask questions about its content.")

# Setup chain if PDF uploaded
if st.session_state["pdf_retriever"]:
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state["pdf_retriever"],
        memory=memory,
        verbose=False
    )
else:
    chain = None

# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    if chain:
        # Use chain with memory and retrieval
        response = chain.run(question=user_input)
    else:
        # Build chat history for LLM fallback
        msgs = []
        for m in st.session_state["messages"]:
            if m["role"] == "user":
                msgs.append(HumanMessage(content=m["content"]))
            else:
                msgs.append(AIMessage(content=m["content"]))
        msgs.append(HumanMessage(content=user_input))
        ai_response = llm(msgs)
        if isinstance(ai_response, list):
            response = ai_response[-1].content
        else:
            response = ai_response.content
    # Append assistant response
    st.session_state["messages"].append({"role": "assistant", "content": response})

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
