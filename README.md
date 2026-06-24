# Educational RAG Project

This project is a local Retrieval-Augmented Generation (RAG) system for asking
questions about educational documents. The documents are stored in `data/`,
converted into searchable chunks, saved into a local ChromaDB database, and then
used by a Streamlit chatbot in `app.py`.

## What Was Fixed

The main problem was that `chroma_db` was not being generated reliably, which
also made `app.py` fail because the Streamlit app depends on that database.

The important corrections were:

- Project paths are now anchored to the `edu_rag_project/` folder.
- `data/` and `chroma_db/` are read from the correct location no matter where the
  command is run from.
- ChromaDB generation is now the default pipeline behavior.
- The terminal chat loop is optional and only runs when `--chat` is passed.
- The embedding step now uses a local hashing embedder so the database can be
  generated offline without waiting for a Hugging Face model download.
- `app.py` now imports the same `CHROMA_DIR` and embedding function used by the
  pipeline, so both files agree on the database location and embedding method.
- `requirements.txt` was updated to match the packages used by the project.

## Folder Structure

```text
edu_rag_project/
  app.py
  education_rag_pipeline.py
  requirements.txt
  test_pdfs.py
  .env
  data/
  chroma_db/
  venv/
```

## Important Files

### `education_rag_pipeline.py`

This is the main indexing pipeline. Its job is to create or load the local
ChromaDB vector database.

Main responsibilities:

- Defines stable project paths:
  - `PROJECT_DIR`
  - `DATA_DIR`
  - `CHROMA_DIR`
- Loads supported documents from `data/`.
- Supports `.pdf`, `.txt`, and `.md` files.
- Splits loaded documents into smaller chunks.
- Converts chunks into vectors using `HashingEmbeddings`.
- Saves the vectors into `chroma_db/`.
- Can optionally start a terminal question-answer loop with `--chat`.

Important functions:

- `load_documents()`: reads files from `data/`.
- `split_documents()`: breaks long documents into smaller overlapping chunks.
- `HashingEmbeddings`: creates local deterministic vectors without internet.
- `build_vector_store()`: creates or loads the ChromaDB database.
- `get_groq_llm()`: loads the Groq model for answering questions.
- `build_qa_chain()`: connects retrieval, prompt formatting, Groq, and output.
- `main()`: controls the full process.

Why this file is important:

`app.py` cannot answer document-based questions unless this file has already
created `chroma_db/`. This is the file that prepares the knowledge base.

### `app.py`

This is the Streamlit web application. It gives the user a simple browser
interface for asking questions.

Main responsibilities:

- Loads the `.env` file and reads `GROQ_API_KEY`.
- Loads the same embedding function used by the pipeline.
- Opens the existing `chroma_db/` database.
- Builds a retrieval chain.
- Sends retrieved context and the user question to Groq.
- Displays the answer in the browser.
- Handles Groq rate-limit errors with retries.

Why this file is important:

This is the user-facing chatbot. It does not create the database itself; it
expects `education_rag_pipeline.py` to create `chroma_db/` first.

### `requirements.txt`

This file lists the Python packages needed by the project.

Important packages:

- `langchain`: chain and prompt building.
- `langchain-community`: document loaders and Chroma wrapper.
- `langchain-groq`: Groq LLM integration.
- `langchain-text-splitters`: document chunking.
- `chromadb`: local vector database.
- `pypdf`: PDF text extraction.
- `python-dotenv`: loads `.env` values.
- `streamlit`: web UI.

Why this file is important:

It helps another environment install the same dependencies needed to run the
pipeline and app.

### `test_pdfs.py`

This helper checks whether PDF files contain extractable text.

Why this file is important:

Some PDFs are scanned images. If a PDF is only an image, normal PDF text
extraction will not work, and the pipeline may create few or no useful chunks.
This file helps confirm whether the PDFs are readable before indexing.

### `data/`

This folder stores the educational documents that should be indexed.

Supported file types:

- `.pdf`
- `.txt`
- `.md`

Why this folder is important:

The RAG system can only answer from materials placed here.

### `chroma_db/`

This folder stores the generated Chroma vector database.

Why this folder is important:

It is the searchable memory of the project. `app.py` loads this database to find
the most relevant document chunks for each question.

### `.env`

This file stores private environment variables such as:

```text
GROQ_API_KEY=your_key_here
```

To configure the app locally:

1. Copy `.env.example` to `.env`.
2. Replace `your_key_here` with your real Groq API key.
3. Optionally, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` if you want to use Streamlit secrets.

Why this file is important:

Groq needs an API key before the app can generate answers. Do not share this file
publicly.

## Full Process

### 1. Add Documents

Put educational materials into:

```text
edu_rag_project/data/
```

The documents should be searchable PDFs, text files, or Markdown files.

### 2. Generate ChromaDB

From the `personal_project` folder, run:

```powershell
.\edu_rag_project\venv\Scripts\python.exe -u edu_rag_project\education_rag_pipeline.py
```

This command:

- Loads documents from `edu_rag_project/data/`.
- Splits them into chunks.
- Embeds each chunk locally.
- Creates or loads `edu_rag_project/chroma_db/`.
- Prints the number of vectors in the database.

Expected result:

```text
ChromaDB is ready at: ...\edu_rag_project\chroma_db
```

### 3. Rebuild ChromaDB

If documents change and you want to recreate the database from scratch, run:

```powershell
.\edu_rag_project\venv\Scripts\python.exe -u edu_rag_project\education_rag_pipeline.py --rebuild
```

Use this when:

- New documents were added.
- Old documents were removed.
- The database seems outdated.
- You want a clean index.

### 4. Optional Terminal Chat

To build or load the database and then chat in the terminal, run:

```powershell
.\edu_rag_project\venv\Scripts\python.exe -u edu_rag_project\education_rag_pipeline.py --chat
```

This requires `GROQ_API_KEY` to be set in `.env`.

### 5. Run the Streamlit App

After `chroma_db/` exists, start the web app:

```powershell
.\edu_rag_project\venv\Scripts\streamlit.exe run edu_rag_project\app.py
```

The app will open in the browser or show a local URL in the terminal.

## How RAG Works Here

RAG means Retrieval-Augmented Generation.

In this project, the process is:

1. Documents are loaded from `data/`.
2. Documents are split into smaller chunks.
3. Each chunk is converted into a vector.
4. Vectors are stored in ChromaDB.
5. A user asks a question in Streamlit.
6. ChromaDB retrieves the most relevant chunks.
7. The retrieved chunks are placed into a prompt.
8. Groq generates an answer using only that context.

This makes the chatbot answer from your educational materials instead of relying
only on the model's general knowledge.

## Why Local Hashing Embeddings Are Used

The project uses `HashingEmbeddings` instead of a Hugging Face transformer model
by default.

Benefits:

- Works offline.
- Does not require downloading model files.
- Starts faster on low-resource machines.
- Produces deterministic vectors, so the same text always gets the same vector.
- Avoids the issue where ChromaDB generation hangs while trying to initialize or
  download an embedding model.

Tradeoff:

Hashing embeddings are simpler than transformer embeddings. They are good for
matching words and related textbook chunks, but they may be less semantically
powerful than models such as `sentence-transformers/all-MiniLM-L6-v2`.

## Troubleshooting

### `Vector store not found`

Run the pipeline first:

```powershell
.\edu_rag_project\venv\Scripts\python.exe -u edu_rag_project\education_rag_pipeline.py
```

### `GROQ_API_KEY not found`

Create or update `.env` from `.env.example` and add:

```text
GROQ_API_KEY=your_key_here
```

Or create `.streamlit/secrets.toml` from
`.streamlit/secrets.toml.example` and add:

```toml
GROQ_API_KEY = "your_key_here"
```

### No Useful Answers

Check that:

- Documents are inside `data/`.
- PDFs contain selectable/searchable text.
- `chroma_db/` was rebuilt after adding new documents.
- The question is related to the indexed documents.

### Scanned PDFs

If a PDF is scanned, the text extractor may not read it. Run:

```powershell
.\edu_rag_project\venv\Scripts\python.exe edu_rag_project\test_pdfs.py
```

If the PDF has no extractable text, it needs OCR before it can be indexed well.

## Recommended Workflow

1. Add or update files in `data/`.
2. Run `test_pdfs.py` if the files are PDFs.
3. Run the pipeline with `--rebuild`.
4. Confirm that `chroma_db/` exists.
5. Start `app.py` with Streamlit.
6. Ask questions in the browser.

