# PDF Chatbot 🚀

A Streamlit-based conversational AI assistant that lets you ask questions to your PDFs—and remembers your conversation context. Powered by Groq’s hosted LLM, FAISS for vector retrieval, and LangChain for chaining and memory.

---

## 🔍 Features

- **PDF Upload & Indexing**  
  Upload a PDF and automatically split, embed, and index its contents with FAISS.

- **Conversational Memory**  
  Maintains chat history in session state, so follow-up questions “remember” earlier context.

- **Groq LLM Integration**  
  Uses `langchain-groq`’s `ChatGroq` for inference—no local GPU required.

- **Simple Deployment**  
  Configurable via `.env` (local) and Streamlit’s secrets (Cloud) for secure key management.

---

## 📂 Repository Structure

qna-chatbot/
├── app.py
├── requirements.txt
├── .gitignore
├── .env # NOT committed (local only)
└── .streamlit/
└── secrets.toml # committed with placeholder values

## 📝 Usage
1. Upload a PDF via the sidebar.
2. Type your question in the chat box and hit “Send.”
3. Continue the conversation—the bot retains context.
