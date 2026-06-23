from education_rag_pipeline import CHROMA_DIR, get_embeddings, retrieve_context
from langchain_community.vectorstores import Chroma

embeddings = get_embeddings()
vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)

query = "What is the law of conservation of energy?"  # change to a question from your textbook
context = retrieve_context(vectordb, query, k=4)
print("Retrieved context:\n", context)