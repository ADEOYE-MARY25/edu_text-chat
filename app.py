"""
Streamlit web UI for the educational RAG chatbot.
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from education_rag_pipeline import (
    CHROMA_DIR,
    PROJECT_DIR,
    get_embeddings,
    retrieve_context,
    build_vector_store,
    load_documents,
    split_documents,
)

st.set_page_config(
    page_title="EduBot - Educational RAG",
    page_icon="book",
    layout="wide",
)

# Load .env for local development
load_dotenv(PROJECT_DIR / ".env")

# Try Streamlit secrets first, then environment
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError, AttributeError):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it in Streamlit secrets, in .env, or as an environment variable.")
    st.stop()


@st.cache_resource
def load_embeddings():
    return get_embeddings()


@st.cache_resource
def load_vector_store():
    from langchain_community.vectorstores import Chroma

    embeddings = load_embeddings()

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        st.info("Building vector database from documents... (this may take a minute)")
        docs = load_documents()
        if not docs:
            st.error("No documents found in data/ folder.")
            st.stop()
        chunks = split_documents(docs)
        if not chunks:
            st.error("No text could be extracted from your documents.")
            st.stop()
        vectordb = build_vector_store(chunks, embeddings, persist_dir=CHROMA_DIR, force_rebuild=True)
    else:
        vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

    if vectordb._collection.count() == 0:
        st.error("The vector store is empty. Add documents to data/ and restart the app.")
        st.stop()

    return vectordb


@st.cache_resource
def get_groq_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        max_tokens=1024,
        request_timeout=30,
    )


def answer_with_retry(qa_chain, query, max_retries=3, base_delay=2):
    for attempt in range(max_retries):
        try:
            return qa_chain.invoke(query)
        except Exception as exc:
            error_msg = str(exc).lower()
            is_rate_limit = "rate limit" in error_msg or "429" in error_msg

            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                st.warning(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                continue

            if is_rate_limit:
                return "The chatbot is experiencing high demand. Please try again in a minute."

            return f"An error occurred: {exc}"

    return "Max retries exceeded. Please try again later."


@st.cache_resource
def build_qa_chain():
    vectordb = load_vector_store()
    llm = get_groq_llm()
    context_retriever = RunnableLambda(lambda question: retrieve_context(vectordb, question, k=8))

    template = """You are an expert educational assistant. Use ONLY the following pieces of context to answer the user's question.
If you don't know the answer based on the context, say "I don't have enough information in the provided materials to answer that."
Keep your answer clear, accurate, and appropriate for a student.

Context:
{context}

Question: {question}

Helpful Educational Answer:"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{question}"),
        ]
    )

    return (
        {"context": context_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


st.markdown(
    "<div style='background: linear-gradient(135deg, #6B82FF 0%, #3DC4A9 100%); "
    "padding: 32px; border-radius: 24px; color: white; text-align: center;'>"
    "<h1 style='margin: 0; font-size: 3rem;'>EduBot</h1>"
    "<p style='margin: 10px 0 0; font-size: 1.05rem;'>A vibrant study assistant built for students.</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='margin-top: 24px; padding: 22px; border-radius: 22px; "
    "background: #f3f7ff; box-shadow: 0 18px 50px rgba(34, 77, 255, 0.08);'>"
    "<p style='margin: 0; font-size: 1.05rem; color: #1f2937;'>Ask questions based on your educational materials and get answers with context.</p>"
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h2 style='color: #3D5AFE;'>About EduBot</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background: white; padding: 18px; border-radius: 18px; "
        "box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);'>"
        "<ul style='margin: 0; padding-left: 20px; color: #334155;'>"
        "<li>Groq-powered answer generation</li>"
        "<li>Local hashing embeddings</li>"
        "<li>Searches your indexed educational documents</li>"
        "<li>Designed for students and classrooms</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### Database")
    st.code(str(CHROMA_DIR), language="text")


try:
    qa_chain = build_qa_chain()
    st.success("RAG system ready. Type your question below.")
except Exception as exc:
    st.error(f"Failed to initialize RAG system: {exc}")
    st.stop()


query = st.text_input(
    "Your question:",
    placeholder="Example: What is the law of conservation of energy?",
)

if query:
    with st.spinner("Thinking..."):
        answer = answer_with_retry(qa_chain, query)

    st.markdown("### Answer")
    st.write(answer)
    st.caption("Each question is answered independently, without conversation memory.")