import streamlit as st
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide"
)


# -------------------------------------------------------
# MODERN UI STYLING
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#eef2f7,#e3ecf7);
font-family: 'Segoe UI', sans-serif;
}

.header{
text-align:center;
padding-top:20px;
padding-bottom:20px;
}

.title{
font-size:48px;
font-weight:800;
background: linear-gradient(90deg,#1b4f72,#2e86c1);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}

.subtitle{
font-size:20px;
color:#5d6d7e;
}

.card{
background: rgba(255,255,255,0.9);
padding:25px;
border-radius:18px;
box-shadow:0px 10px 25px rgba(0,0,0,0.08);
margin-bottom:25px;
}

.tech-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 8px 20px rgba(0,0,0,0.08);
text-align:center;
font-weight:600;
}

[data-testid="stChatMessageContent"][aria-label="assistant"]{
background:white;
padding:16px;
border-radius:15px;
}

[data-testid="stChatMessageContent"][aria-label="user"]{
background:#27ae60;
color:white;
padding:16px;
border-radius:15px;
}

.footer{
text-align:center;
margin-top:40px;
color:#7b7d7d;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""
<div class="header">
<div class="title">🩺 Medical AI Assistant</div>
<div class="subtitle">
AI Powered Healthcare Knowledge System using Retrieval-Augmented Generation
</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------------

st.markdown("### 📌 Project Overview")

st.markdown("""
<div class="card">

The **Medical AI Assistant** is an intelligent healthcare question-answering system built using **Retrieval-Augmented Generation (RAG)**.

Traditional AI chatbots rely only on training data, which may be outdated. This system improves reliability by retrieving relevant medical information from a **vector database** before generating responses.

The system combines:

• Semantic Search using embeddings  
• Vector database retrieval  
• Large Language Model reasoning  

This allows the assistant to produce **accurate and context-aware healthcare explanations**.

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PROBLEM STATEMENT
# -------------------------------------------------------

st.markdown("### ❗ Problem Statement")

st.markdown("""
<div class="card">

Healthcare information on the internet can often be **confusing, fragmented, or unreliable**.

Users frequently face problems such as:

• Searching multiple websites for medical information  
• Difficulty understanding medical terminology  
• Misinformation from unreliable sources  

This project aims to build an **AI-powered assistant that retrieves trusted information and generates clear explanations.**

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# SOLUTION APPROACH
# -------------------------------------------------------

st.markdown("### 🧠 Solution Approach")

st.markdown("""
<div class="card">

The system uses **Retrieval-Augmented Generation (RAG)**.

Workflow:

1️⃣ User asks a question  
2️⃣ Question converted to vector embeddings  
3️⃣ Pinecone searches similar documents  
4️⃣ Relevant context retrieved  
5️⃣ AI model generates response using the context

This ensures answers are **reliable and grounded in real data**.

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# TECHNOLOGY STACK
# -------------------------------------------------------

st.markdown("### ⚙️ Technology Stack")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="tech-card">🧠 LangChain<br>AI Orchestration</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="tech-card">📚 Pinecone<br>Vector Database</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="tech-card">🤖 Ollama<br>Local LLM</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="tech-card">🔎 Sentence Transformers<br>Embeddings</div>', unsafe_allow_html=True)


# -------------------------------------------------------
# REAL WORLD USE CASES
# -------------------------------------------------------

st.markdown("### 🌍 Real World Use Cases")

st.markdown("""
<div class="card">

👨‍⚕️ Patient Education — Users can ask questions about diseases and treatments.

🏥 Hospital AI Assistants — Can be integrated into hospital websites.

📚 Medical Learning — Helpful for medical students.

💻 Telemedicine Chatbots — Used in healthcare applications.

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# BENEFITS
# -------------------------------------------------------

st.markdown("### ⭐ Key Benefits")

st.markdown("""
<div class="card">

✔ Quick access to healthcare knowledge  
✔ Reduces misinformation  
✔ Demonstrates modern Generative AI architecture  
✔ Scalable AI system using vector search

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# LIMITATIONS
# -------------------------------------------------------

st.markdown("### ⚠ Limitations")

st.markdown("""
<div class="card">

• The system **does not replace professional medical advice**

• Answers depend on the **quality of the dataset**

Users should always consult **medical professionals** for real health decisions.

</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# FUTURE IMPROVEMENTS
# -------------------------------------------------------

st.markdown("### 🚀 Future Improvements")

st.markdown("""
<div class="card">

Future enhancements may include:

📄 Medical PDF document upload  
📚 Source citations for answers  
🧠 Conversation memory  
⚡ Streaming responses  
☁ Cloud deployment

</div>
""", unsafe_allow_html=True)


st.markdown("---")


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.title("⚙️ System Info")

    st.write("Model: **Ollama phi**")
    st.write("Vector DB: **Pinecone**")
    st.write("Retriever: **Top 3 documents**")

    st.markdown("---")

    st.write(
        "This project demonstrates a **RAG-based healthcare AI assistant** "
        "built with LangChain, Pinecone, and Ollama."
    )


# -------------------------------------------------------
# LOAD ENV
# -------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

embeddings = load_embeddings()


# -------------------------------------------------------
# VECTOR STORE
# -------------------------------------------------------

index_name = "healthcarebot"

@st.cache_resource
def load_vector_store():
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore


docsearch = load_vector_store()

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)


# -------------------------------------------------------
# LLM
# -------------------------------------------------------

llm = ChatOllama(
    model="phi",
    temperature=0
)


# -------------------------------------------------------
# PROMPT TEMPLATE
# -------------------------------------------------------

system_prompt = (
"You are a helpful medical assistant.\n"
"Use the context to answer the question.\n"
"If the answer is not in context say you don't know.\n"
"Limit response to three sentences.\n\n"
"{context}"
)

prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{input}")
])


# -------------------------------------------------------
# RAG CHAIN
# -------------------------------------------------------

document_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)


# -------------------------------------------------------
# CHAT MEMORY
# -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------------
# CHAT UI
# -------------------------------------------------------

st.markdown("### 💬 Ask a Medical Question")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -------------------------------------------------------
# USER INPUT
# -------------------------------------------------------

user_prompt = st.chat_input("Type your medical question...")

if user_prompt:

    st.session_state.messages.append(
        {"role":"user","content":user_prompt}
    )

    with st.chat_message("user"):
        st.write(user_prompt)

    with st.spinner("Analyzing medical knowledge..."):

        response = rag_chain.invoke({
            "input": user_prompt
        })

        answer = response["answer"]

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )

    with st.chat_message("assistant"):
        st.write(answer)


# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown("""
<div class="footer">

Built with ❤️ using **LangChain • Pinecone • Ollama • Streamlit**


</div>
""", unsafe_allow_html=True)
