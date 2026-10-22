# QnA Chatbot 

**Upload a PDF and ask questions in everyday language.** The chatbot can help you summarize a document, find details, explore important dates, and ask follow-up questions. Answers include page references and a way to inspect the passages used to prepare them.

You do not need to know any technical terms to use the app: choose a PDF, select an example question or type your own, and read the answer. The app currently supports PDFs with selectable text up to 20 MB and 300 pages; scanned pages and image-only PDFs are not supported yet.

## Using the app

1. Choose a PDF in the left-hand panel.
2. Wait while the app prepares the document. Larger PDFs may take longer.
3. Select one of the example questions or type a question in your own words.
4. Open **View passages used for this answer** to see the PDF text that supported the response.

The chatbot can make mistakes or miss relevant details. Check important information against the displayed PDF passages. The app sends your question and selected relevant passages to an online AI service to draft an answer, so avoid uploading confidential or sensitive documents.

This is a showcase project built with Streamlit, local embeddings, FAISS hybrid retrieval, and a Groq-hosted language model. Technical setup and implementation details follow.

## Features

- **PDF ingestion:** extracts text by page, splits it into overlapping chunks, and retains page/source metadata.
- **Local embeddings:** uses `sentence-transformers/all-MiniLM-L6-v2` on CPU.
- **Hybrid retrieval:** combines semantic similarity with BM25-style keyword relevance using Reciprocal Rank Fusion (RRF); returns up to four excerpts.
- **Conversational follow-ups:** rewrites follow-up questions into standalone search queries using the conversation history.
- **Grounded answer prompt:** instructs the language model to answer from retrieved excerpts and use `[Page N]` citations.
- **Source inspection:** displays the retrieved PDF excerpts under each answer.
- **Basic safeguards:** 20 MB upload limit, 300-page limit, duplicate-upload detection, and clear errors for PDFs without extractable text.
- **Regression tests:** covers lexical tokenization, BM25 ranking, hybrid ranking, result limits, and empty-index handling.

## Architecture

```text
PDF upload
    ↓
Page-wise text extraction (pypdf)
    ↓
Overlapping text chunks + page/source metadata
    ↓
Local sentence-transformer embeddings
    ↓
FAISS inner-product index ─────────────┐
                                       ├─→ RRF hybrid ranking → Top 4 excerpts
BM25-style keyword ranking ────────────┘                         ↓
                                                   Groq-hosted LLM
                                                   + conversation history
                                                              ↓
                                                Answer + page citations
                                                + inspectable excerpts
```

PDF extraction and embedding/index construction run locally. The question and retrieved excerpts are sent to the configured Groq model to generate the answer, so this is **not a fully offline or fully local LLM application**.

## Requirements

- Python compatible with the versions pinned in `requirements.txt`
- A Groq API key
- Internet access for Groq requests and the initial download of the embedding model

## Run locally on Windows

Open Anaconda Prompt or a terminal in the project folder:

```bat
conda create -n qna-v2 python=3.13 -y
conda activate qna-v2
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root (use `.env.example` as a template) and set your key:

```dotenv
GROQ_API_KEY=put_your_own_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

Keep `.env` private. Do not commit it or share screenshots/logs that expose the key. The app also supports Streamlit secrets for hosted deployments.

Start the app:

```bat
streamlit run app.py --server.fileWatcherType=none
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Tests and checks

Run the retrieval regression suite and basic project checks from the project root:

```bat
python -m py_compile app.py retrieval.py tests\test_retrieval.py
python -m unittest discover -s tests -v
python -m pip check
git diff --check
```

The retrieval unit tests do not call Groq or measure end-to-end answer accuracy. Use a small set of representative questions against a known PDF for a manual smoke test as well.

## Current limitations

- Scanned/image-only PDFs are not OCR-processed; the file must contain extractable text.
- Retrieval can miss relevant passages, and a language model can still produce an incomplete answer or an imperfect page citation. Verify important answers against the displayed excerpts.
- The four-excerpt retrieval limit favors concise context but can omit evidence spread across many pages.
- Exact, current information that is absent from the uploaded document cannot be established from that document alone.
- Answer generation requires access to the configured Groq model and is subject to API availability, quota, and model changes.

## Demo checklist

1. Start the app and upload a text-based PDF.
2. Confirm that the page/chunk indexing status appears.
3. Ask one direct factual question and confirm that retrieved excerpts are available.
4. Ask a follow-up such as “What is the minimum duration?” after asking about programme duration.
5. Ask something not specified in the PDF and check that the response acknowledges the gap instead of inventing a precise fact.
6. Keep the source excerpt expander available when presenting an answer so the evidence can be inspected live.

## Tech stack

Python · Streamlit · LangChain Core · ChatGroq · Sentence Transformers · FAISS CPU · NumPy · pypdf · python-dotenv
