import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(
    page_title="AI Course Intelligence",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    background: linear-gradient(to bottom right, #0f172a, #111827);
}
.big-title {
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("<div class='big-title'>🎓 AI Course Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Semantic Retrieval + Gemini 2.5 + 13,876 Course Embeddings</div>", unsafe_allow_html=True)

# Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='metric-box'>📚 10,000+ Courses</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-box'>⚡ 13,876 Chunks</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-box'>🧠 Gemini 2.5 Flash</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='metric-box'>🔍 Top-K = 6</div>", unsafe_allow_html=True)

st.markdown("---")
# Load DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 6}),
    return_source_documents=True
)

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask about courses, skills, pricing, levels...")

if query:
    with st.spinner("Searching 13,876 course embeddings..."):
        response = qa.invoke({"query": query})

    st.session_state.chat_history.append({
        "role": "user",
        "message": query
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "message": response["result"],
        "sources": response["source_documents"]
    })

# ---- DISPLAY BLOCK ----
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").markdown(chat["message"])
    else:
        st.chat_message("assistant").markdown(
            f"<div class='card'>{chat['message']}</div>",
            unsafe_allow_html=True
        )

        with st.expander("🔎 View Retrieved Context"):
            for doc in chat["sources"]:
                st.write(doc.page_content)