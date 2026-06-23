# """
# Educational RAG pipeline.

# This script loads educational files from edu_rag_project/data, splits them into
# small text chunks, embeds those chunks locally, and stores them in
# edu_rag_project/chroma_db for app.py to reuse.
# """

# import os
# import argparse
# import hashlib
# import math
# import re
# import shutil
# from pathlib import Path

# from dotenv import load_dotenv
# from langchain_community.embeddings import FastEmbedEmbeddings



# # These paths are anchored to this file, so the script works whether you run it
# # from personal_project/ or from edu_rag_project/.
# PROJECT_DIR = Path(__file__).resolve().parent
# DATA_DIR = PROJECT_DIR / "data"
# CHROMA_DIR = PROJECT_DIR / "chroma_db"

# EMBEDDING_DIMENSIONS = 384
# TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
# STOPWORDS = {
#     "a",
#     "an",
#     "and",
#     "are",
#     "as",
#     "at",
#     "be",
#     "by",
#     "for",
#     "from",
#     "how",
#     "in",
#     "is",
#     "it",
#     "of",
#     "on",
#     "or",
#     "that",
#     "the",
#     "this",
#     "to",
#     "what",
#     "when",
#     "where",
#     "which",
#     "who",
#     "why",
#     "with",
# }


# def load_documents(data_dir=DATA_DIR):
#     """Load .txt, .md, and searchable .pdf files into LangChain documents."""
#     documents = []
#     data_path = Path(data_dir)

#     if not data_path.exists():
#         print(f"Data folder not found. Creating it at: {data_path}")
#         data_path.mkdir(parents=True, exist_ok=True)
#         return documents

#     supported_suffixes = {".txt", ".md", ".pdf"}
#     files = [
#         file_path
#         for file_path in sorted(data_path.iterdir())
#         if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
#     ]

#     if not files:
#         print(f"No supported files found in: {data_path}")
#         print("Add .txt, .md, or searchable .pdf files, then run this script again.")
#         return documents

#     for file_path in files:
#         if file_path.suffix.lower() in {".txt", ".md"}:
#             from langchain_community.document_loaders.text import TextLoader

#             loader = TextLoader(str(file_path), encoding="utf-8")
#         else:
#             from langchain_community.document_loaders.pdf import PyPDFLoader

#             loader = PyPDFLoader(str(file_path))

#         documents.extend(loader.load())
#         print(f"Loaded: {file_path.name}")

#     print(f"Loaded {len(documents)} document page(s) from: {data_path}")
#     return documents


# def split_documents(documents, chunk_size=500, chunk_overlap=50):
#     """Split documents into overlapping chunks for better retrieval."""
#     from langchain_text_splitters import RecursiveCharacterTextSplitter

#     if not documents:
#         return []

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""],
#         length_function=len,
#     )

#     chunks = text_splitter.split_documents(documents)
#     chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

#     print(f"Split into {len(chunks)} non-empty text chunk(s)")
#     return chunks


# def get_embeddings():
#     return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


# class HashingEmbeddings:
#     """Small local embedding function compatible with LangChain and Chroma.

#     The embedder converts words into a fixed-size normalized vector using stable
#     hashes. It is not as semantically rich as a transformer model, but it is fast,
#     deterministic, offline, and good enough to retrieve matching textbook chunks.
#     """

#     def __init__(self, dimensions=384):
#         self.dimensions = dimensions
#         self._token_pattern = re.compile(r"[a-zA-Z0-9]+")

#     def embed_documents(self, texts):
#         return [self._embed(text) for text in texts]

#     def embed_query(self, text):
#         return self._embed(text)

#     def _embed(self, text):
#         vector = [0.0] * self.dimensions
#         tokens = self._token_pattern.findall(text.lower())

#         for token in tokens:
#             digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
#             value = int.from_bytes(digest, byteorder="big", signed=False)
#             index = value % self.dimensions
#             sign = 1.0 if (value >> 1) % 2 == 0 else -1.0
#             vector[index] += sign

#         norm = math.sqrt(sum(value * value for value in vector))
#         if norm == 0:
#             return vector

#         return [value / norm for value in vector]


# def build_vector_store(
#     chunks,
#     embeddings,
#     persist_dir=CHROMA_DIR,
#     batch_size=500,
#     force_rebuild=False,
# ):
#     """Create or load a persistent Chroma database.

#     Chroma cannot initialize a collection from an empty list, so this function
#     raises a clear error if the files did not produce readable text chunks.
#     """
#     from langchain_community.vectorstores import Chroma

#     persist_path = Path(persist_dir)

#     if force_rebuild and persist_path.exists():
#         print(f"Removing existing vector store at: {persist_path}")
#         shutil.rmtree(persist_path)

#     if persist_path.exists() and any(persist_path.iterdir()):
#         print(f"Loading existing vector store from: {persist_path}")
#         vectordb = Chroma(
#             persist_directory=str(persist_path),
#             embedding_function=embeddings,
#         )
#     else:
#         if not chunks:
#             raise ValueError(
#                 "No text chunks were created. Make sure data/ contains readable text, "
#                 "Markdown, or searchable PDF files."
#             )

#         print(f"Creating new vector store at: {persist_path}")
#         persist_path.mkdir(parents=True, exist_ok=True)

#         # Seed the collection with the first chunk, then add the rest in batches
#         # so large PDF sets do not need to be embedded all at once.
#         vectordb = Chroma.from_documents(
#             documents=chunks[:1],
#             embedding=embeddings,
#             persist_directory=str(persist_path),
#         )

#         total = len(chunks)
#         for start in range(1, total, batch_size):
#             end = min(start + batch_size, total)
#             vectordb.add_documents(chunks[start:end])
#             print(f"Added chunks {start + 1}-{end} of {total}")

#     print(f"Vector store contains {vectordb._collection.count()} vector(s)")
#     return vectordb


# def get_groq_llm():
#     """Create the Groq chat model used for the command-line QA loop."""
#     from langchain_groq import ChatGroq

#     load_dotenv(PROJECT_DIR / ".env")
#     api_key = os.getenv("GROQ_API_KEY")

#     if not api_key:
#         raise ValueError("GROQ_API_KEY not found. Set it in .env or environment variables.")

#     return ChatGroq(
#         temperature=0.3,
#         groq_api_key=api_key,
#         model_name="llama-3.3-70b-versatile",
#         max_tokens=1024,
#     )


# def format_docs(docs):
#     """Join retrieved chunks into one context string for the prompt."""
#     return "\n\n".join(doc.page_content for doc in docs)


# def tokenize_for_search(text):
#     """Extract useful lowercase tokens for keyword retrieval."""
#     tokens = TOKEN_PATTERN.findall(text.lower())
#     return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


# def load_all_chroma_docs(vectordb):
#     """Read stored Chroma documents once and cache them on the vector store."""
#     if hasattr(vectordb, "_all_docs_cache"):
#         return vectordb._all_docs_cache

#     collection_data = vectordb._collection.get(include=["documents", "metadatas"])
#     documents = collection_data.get("documents") or []
#     metadatas = collection_data.get("metadatas") or []

#     vectordb._all_docs_cache = list(zip(documents, metadatas))
#     return vectordb._all_docs_cache


# def lexical_search(vectordb, query, k=8):
#     """Find chunks that contain the strongest keyword overlap with the query."""
#     from langchain_core.documents import Document

#     query_tokens = tokenize_for_search(query)
#     if not query_tokens:
#         return []

#     query_phrase = " ".join(query_tokens)
#     scored_docs = []

#     for content, metadata in load_all_chroma_docs(vectordb):
#         if not content:
#             continue

#         content_lower = content.lower()
#         score = 0

#         for token in query_tokens:
#             # Count repeated topic words because textbook chunks often repeat
#             # key terms in definitions and explanations.
#             score += content_lower.count(token)

#         if query_phrase and query_phrase in content_lower:
#             score += 5

#         if score > 0:
#             scored_docs.append((score, Document(page_content=content, metadata=metadata or {})))

#     scored_docs.sort(key=lambda item: item[0], reverse=True)
#     return [doc for _, doc in scored_docs[:k]]


# def retrieve_relevant_docs(vectordb, query, k=8):
#     """Combine vector search with keyword search for more reliable textbook QA."""
#     vector_docs = vectordb.similarity_search(query, k=k)
#     keyword_docs = lexical_search(vectordb, query, k=k)

#     combined_docs = []
#     seen = set()

#     # Put keyword results first because the offline hashing vectors are simple;
#     # exact textbook term matches are often the strongest signal for school notes.
#     for doc in keyword_docs + vector_docs:
#         key = (doc.page_content[:200], doc.metadata.get("source"), doc.metadata.get("page"))
#         if key in seen:
#             continue
#         seen.add(key)
#         combined_docs.append(doc)

#         if len(combined_docs) >= k:
#             break

#     return combined_docs


# def retrieve_context(vectordb, query, k=8):
#     """Return formatted context for the LLM, including source hints."""
#     docs = retrieve_relevant_docs(vectordb, query, k=k)
#     context_parts = []

#     for index, doc in enumerate(docs, start=1):
#         source = Path(doc.metadata.get("source", "unknown")).name
#         page = doc.metadata.get("page")
#         page_label = f", page {page + 1}" if isinstance(page, int) else ""
#         context_parts.append(
#             f"[Source {index}: {source}{page_label}]\n{doc.page_content}"
#         )

#     return "\n\n".join(context_parts)


# def build_qa_chain(vectordb):
#     """Connect Chroma retrieval, the prompt, Groq, and output parsing."""
#     from langchain_core.output_parsers import StrOutputParser
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.runnables import RunnableLambda, RunnablePassthrough

#     llm = get_groq_llm()
#     context_retriever = RunnableLambda(lambda question: retrieve_context(vectordb, question, k=8))

#     template = """You are an expert educational assistant. Use ONLY the following pieces of context to answer the user's question.
# If you don't know the answer based on the context, say "I don't have enough information in the provided materials to answer that."
# Keep your answer clear, accurate, and appropriate for a student.

# Context:
# {context}

# Question: {question}

# Helpful Educational Answer:"""

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", template),
#             ("human", "{question}"),
#         ]
#     )

#     return (
#         {"context": context_retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )


# def parse_args():
#     """Read command-line options for indexing and optional terminal chat."""
#     parser = argparse.ArgumentParser(description="Build the educational ChromaDB index.")
#     parser.add_argument(
#         "--chat",
#         action="store_true",
#         help="Start the terminal question-answer loop after indexing.",
#     )
#     parser.add_argument(
#         "--rebuild",
#         action="store_true",
#         help="Delete the existing chroma_db folder and rebuild it from data/.",
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     print("\nStarting RAG Pipeline for Education")
#     print("=" * 50)

#     embeddings = get_embeddings()

#     if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) and not args.rebuild:
#         print(f"Existing ChromaDB found at: {CHROMA_DIR}")
#         vectordb = build_vector_store([], embeddings, persist_dir=CHROMA_DIR)
#     else:
#         docs = load_documents(DATA_DIR)
#         if not docs:
#             print(f"No documents loaded. Add files to: {DATA_DIR}")
#             return

#         chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
#         if not chunks:
#             print("No text chunks were created. Your PDFs may be scanned images without text.")
#             return

#         vectordb = build_vector_store(
#             chunks,
#             embeddings,
#             persist_dir=CHROMA_DIR,
#             force_rebuild=args.rebuild,
#         )

#     if not args.chat:
#         print(f"\nChromaDB is ready at: {CHROMA_DIR}")
#         print("Run this script with --chat if you also want the terminal QA loop.")
#         return

#     try:
#         qa_chain = build_qa_chain(vectordb)
#     except ValueError as exc:
#         print(f"\n{exc}")
#         print("ChromaDB was generated successfully, so app.py can use it after the key is configured.")
#         return

#     print("\nRAG system ready. Type your questions, or type 'exit' to stop.")
#     print("-" * 50)

#     while True:
#         query = input("\nYour question: ").strip()
#         if query.lower() in ("exit", "quit", "q"):
#             print("Goodbye.")
#             break
#         if not query:
#             continue

#         try:
#             answer = qa_chain.invoke(query)
#             print(f"\nAnswer:\n{answer}")
#         except Exception as exc:
#             print(f"Error: {exc}. Check your Groq API key or network.")


# if __name__ == "__main__":
#     main()








# """
# Educational RAG pipeline.

# This script loads educational files from edu_rag_project/data, splits them into
# small text chunks, embeds those chunks locally, and stores them in
# edu_rag_project/chroma_db for app.py to reuse.
# """

import os
import argparse
import hashlib
import math
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv

# These paths are anchored to this file, so the script works whether you run it
# from personal_project/ or from edu_rag_project/.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

EMBEDDING_DIMENSIONS = 384
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "what", "when", "where", "which", "who", "why", "with",
}


class HashingEmbeddings:
    """Small local embedding function compatible with LangChain and Chroma.

    The embedder converts words into a fixed-size normalized vector using stable
    hashes. It is not as semantically rich as a transformer model, but it is fast,
    deterministic, offline, and good enough to retrieve matching textbook chunks.
    """

    def __init__(self, dimensions=384):
        self.dimensions = dimensions
        self._token_pattern = re.compile(r"[a-zA-Z0-9]+")

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        tokens = self._token_pattern.findall(text.lower())

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            index = value % self.dimensions
            sign = 1.0 if (value >> 1) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def get_embeddings():
    """Return the local hashing embedder (no internet required)."""
    return HashingEmbeddings(dimensions=EMBEDDING_DIMENSIONS)


def load_documents(data_dir=DATA_DIR):
    """Load .txt, .md, and searchable .pdf files into LangChain documents."""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Data folder not found. Creating it at: {data_path}")
        data_path.mkdir(parents=True, exist_ok=True)
        return documents

    supported_suffixes = {".txt", ".md", ".pdf"}
    files = [
        file_path
        for file_path in sorted(data_path.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    ]

    if not files:
        print(f"No supported files found in: {data_path}")
        print("Add .txt, .md, or searchable .pdf files, then run this script again.")
        return documents

    for file_path in files:
        if file_path.suffix.lower() in {".txt", ".md"}:
            from langchain_community.document_loaders.text import TextLoader
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            from langchain_community.document_loaders.pdf import PyPDFLoader
            loader = PyPDFLoader(str(file_path))

        documents.extend(loader.load())
        print(f"Loaded: {file_path.name}")

    print(f"Loaded {len(documents)} document page(s) from: {data_path}")
    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into overlapping chunks for better retrieval."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    print(f"Split into {len(chunks)} non-empty text chunk(s)")
    return chunks


def build_vector_store(
    chunks,
    embeddings,
    persist_dir=CHROMA_DIR,
    batch_size=500,
    force_rebuild=False,
):
    """Create or load a persistent Chroma database."""
    from langchain_community.vectorstores import Chroma

    persist_path = Path(persist_dir)

    if force_rebuild and persist_path.exists():
        print(f"Removing existing vector store at: {persist_path}")
        shutil.rmtree(persist_path)

    if persist_path.exists() and any(persist_path.iterdir()):
        print(f"Loading existing vector store from: {persist_path}")
        vectordb = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
    else:
        if not chunks:
            raise ValueError(
                "No text chunks were created. Make sure data/ contains readable text, "
                "Markdown, or searchable PDF files."
            )

        print(f"Creating new vector store at: {persist_path}")
        persist_path.mkdir(parents=True, exist_ok=True)

        # Seed with the first chunk, then add the rest in batches
        vectordb = Chroma.from_documents(
            documents=chunks[:1],
            embedding=embeddings,
            persist_directory=str(persist_path),
        )

        total = len(chunks)
        for start in range(1, total, batch_size):
            end = min(start + batch_size, total)
            vectordb.add_documents(chunks[start:end])
            print(f"Added chunks {start + 1}-{end} of {total}")

    print(f"Vector store contains {vectordb._collection.count()} vector(s)")
    return vectordb


def get_groq_llm():
    """Create the Groq chat model used for the command-line QA loop."""
    from langchain_groq import ChatGroq

    load_dotenv(PROJECT_DIR / ".env")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Set it in .env or environment variables.")

    return ChatGroq(
        temperature=0.3,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        max_tokens=1024,
    )


def format_docs(docs):
    """Join retrieved chunks into one context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def tokenize_for_search(text):
    """Extract useful lowercase tokens for keyword retrieval."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def load_all_chroma_docs(vectordb):
    """Read stored Chroma documents once and cache them on the vector store."""
    if hasattr(vectordb, "_all_docs_cache"):
        return vectordb._all_docs_cache

    collection_data = vectordb._collection.get(include=["documents", "metadatas"])
    documents = collection_data.get("documents") or []
    metadatas = collection_data.get("metadatas") or []

    vectordb._all_docs_cache = list(zip(documents, metadatas))
    return vectordb._all_docs_cache


def lexical_search(vectordb, query, k=8):
    """Find chunks that contain the strongest keyword overlap with the query."""
    from langchain_core.documents import Document

    query_tokens = tokenize_for_search(query)
    if not query_tokens:
        return []

    query_phrase = " ".join(query_tokens)
    scored_docs = []

    for content, metadata in load_all_chroma_docs(vectordb):
        if not content:
            continue

        content_lower = content.lower()
        score = 0

        for token in query_tokens:
            score += content_lower.count(token)

        if query_phrase and query_phrase in content_lower:
            score += 5

        if score > 0:
            scored_docs.append((score, Document(page_content=content, metadata=metadata or {})))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:k]]


def retrieve_relevant_docs(vectordb, query, k=8):
    """Combine vector search with keyword search for more reliable textbook QA."""
    vector_docs = vectordb.similarity_search(query, k=k)
    keyword_docs = lexical_search(vectordb, query, k=k)

    combined_docs = []
    seen = set()

    for doc in keyword_docs + vector_docs:
        key = (doc.page_content[:200], doc.metadata.get("source"), doc.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        combined_docs.append(doc)

        if len(combined_docs) >= k:
            break

    return combined_docs


def retrieve_context(vectordb, query, k=8):
    """Return formatted context for the LLM, including source hints."""
    docs = retrieve_relevant_docs(vectordb, query, k=k)
    context_parts = []

    for index, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        context_parts.append(
            f"[Source {index}: {source}{page_label}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def build_qa_chain(vectordb):
    """Connect Chroma retrieval, the prompt, Groq, and output parsing."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

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


def parse_args():
    """Read command-line options for indexing and optional terminal chat."""
    parser = argparse.ArgumentParser(description="Build the educational ChromaDB index.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the terminal question-answer loop after indexing.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing chroma_db folder and rebuild it from data/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\nStarting RAG Pipeline for Education")
    print("=" * 50)

    embeddings = get_embeddings()

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) and not args.rebuild:
        print(f"Existing ChromaDB found at: {CHROMA_DIR}")
        vectordb = build_vector_store([], embeddings, persist_dir=CHROMA_DIR)
    else:
        docs = load_documents(DATA_DIR)
        if not docs:
            print(f"No documents loaded. Add files to: {DATA_DIR}")
            return

        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        if not chunks:
            print("No text chunks were created. Your PDFs may be scanned images without text.")
            return

        vectordb = build_vector_store(
            chunks,
            embeddings,
            persist_dir=CHROMA_DIR,
            force_rebuild=args.rebuild,
        )

    if not args.chat:
        print(f"\nChromaDB is ready at: {CHROMA_DIR}")
        print("Run this script with --chat if you also want the terminal QA loop.")
        return

    try:
        qa_chain = build_qa_chain(vectordb)
    except ValueError as exc:
        print(f"\n{exc}")
        print("ChromaDB was generated successfully, so app.py can use it after the key is configured.")
        return

    print("\nRAG system ready. Type your questions, or type 'exit' to stop.")
    print("-" * 50)

    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if not query:
            continue

        try:
            answer = qa_chain.invoke(query)
            print(f"\nAnswer:\n{answer}")
        except Exception as exc:
            print(f"Error: {exc}. Check your Groq API key or network.")


if __name__ == "__main__":
    main()
