# AI Chatbot with RAG (Retrieval-Augmented Generation) 🤖

A production-style **Retrieval-Augmented Generation** chatbot that answers questions from a private knowledge base. Includes both a command-line interface and a **Flask web app** you can demo in the browser.

This is an advanced, interview-worthy project — RAG is the technology behind every modern AI assistant (support bots, document Q&A, internal knowledge tools).

## 🎯 What makes this "advanced"

Unlike a plain LLM wrapper, this system **grounds its answers in real documents**:
1. **INDEX** - loads documents, splits them into chunks, embeds each chunk
2. **RETRIEVE** - finds the most relevant chunks for a user's question
3. **GENERATE** - produces an answer grounded in the retrieved context (with source citations)

This is the exact architecture used in production RAG systems, just scaled down.

## 🧠 How it works

```
                 ┌────────────────────────────────────────────┐
                 │            KNOWLEDGE BASE                  │
                 │  company.txt  ·  rag.txt  ·  (add your own)│
                 └────────────────────────────────────────────┘
                                  │  INDEX (chunk + embed)
                                  ▼
                 ┌────────────────────────────────────────────┐
                 │              VECTOR STORE                   │
                 │   TF-IDF vectors for every chunk            │
                 └────────────────────────────────────────────┘
   User question ──► RETRIEVE (cosine similarity) ──► top-k chunks
                                  │
                                  ▼
                 ┌────────────────────────────────────────────┐
                 │            GENERATION                      │
                 │   LLM (or offline extractive fallback)     │
                 └────────────────────────────────────────────┘
                                  │
                                  ▼
                          Grounded answer + sources
```

## ✨ Features

- **Chunking** - paragraph-aware splitting with heading merging
- **Retrieval** - TF-IDF + cosine similarity (runs offline, no heavy deps)
- **Generation** - two modes:
  - **LLM mode** (best quality): set `OPENAI_API_KEY`, uses any OpenAI-compatible API
  - **Offline mode** (zero setup): extractive answer, works anywhere
- **Source citations** - shows which document each answer came from
- **Two interfaces** - CLI + Flask web app

## 🚀 How to run

**Web app (browser demo):**
```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000
```

**Command line:**
```bash
python rag_engine.py
```

**Enable LLM mode (optional, best answers):**
```bash
set OPENAI_API_KEY=your_key_here
python app.py
```

## 📊 Example

```
Q: What does SupportBot do?
A: Our flagship product is SupportBot, an AI assistant that answers customer
   questions automatically using retrieval-augmented generation (RAG).
Sources: company.txt

Q: Why use RAG?
A: RAG is used for three main reasons. First, it keeps answers up to date
   without retraining the model... Second, it reduces hallucination...
Sources: rag.txt
```

## 🏗️ Project Structure

```
06-ai-chatbot-rag/
├── app.py              # Flask web server
├── rag_engine.py       # Core RAG pipeline (index, retrieve, generate)
├── knowledge_base/     # Drop your documents here (.txt)
│   ├── company.txt
│   └── rag.txt
├── templates/
│   └── index.html      # Browser chat UI
├── requirements.txt
└── README.md
```

## 📚 Concepts Covered

- Retrieval-Augmented Generation (RAG) end-to-end
- Document chunking strategies & their tradeoffs
- Text embedding & vector similarity search
- LLM API integration with graceful fallback
- Web app deployment (Flask)
- Prompt engineering (system prompt for grounded answers)

## 💡 To Extend (great hackathon ideas)

- Swap TF-IDF for real embeddings (sentence-transformers, OpenAI, or a vector DB like Chroma/FAISS)
- Add document upload through the web UI
- Add reranking to improve retrieval quality
- Add chat history/memory for multi-turn conversations
