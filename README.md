# RAG chatbot

Answers questions from a set of documents, with citations back to the source. Browser UI included.

```
   user question
        │
        ▼
   RETRIEVE ──── top-k chunks by cosine similarity
        │
        ▼
   GENERATE ──── grounded answer + source citations
        │
        ▼
   "Answer. Sources: company.txt"
```

Documents are split on paragraph boundaries with headings kept attached, embedded with TF-IDF, and stored as vectors. Retrieval is cosine similarity over those vectors.

Two generation modes:

- **LLM** — set `OPENAI_API_KEY`, works with any OpenAI-compatible endpoint. Better answers.
- **Offline** — no key, no network. Pulls the most relevant sentences from the retrieved chunks directly.

The offline mode is not a stub. It's what runs in CI, and it makes the retrieval half of the pipeline demonstrable without an API key, which is the half that's actually worth reading.

## Run it

```bash
pip install -r requirements.txt

python app.py     # browser UI at http://127.0.0.1:5000
python rag_engine.py   # command line
```

Drop `.txt` files into `knowledge_base/` to change what it knows.

## Files

```
app.py              # Flask server
rag_engine.py       # index, retrieve, generate
knowledge_base/     # your documents here
templates/          # browser UI
requirements.txt
```