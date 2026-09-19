"""
RAG Engine - Retrieval-Augmented Generation
============================================
A complete RAG pipeline that answers questions from a private knowledge
base:

  1. INDEX  - load documents, split into chunks, embed them
  2. RETRIEVE - given a question, find the most relevant chunks
  3. GENERATE - produce a grounded answer from the retrieved chunks

Design: clean, modular, and runs fully offline for demos. For higher
quality answers, set OPENAI_API_KEY to use a real LLM. Without a key it
falls back to an extractive answer from the retrieved text.

Libraries: scikit-learn (TF-IDF retrieval), standard library
"""

import os
import glob
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self, knowledge_base_dir: str, chunk_size: int = 300):
        self.knowledge_base_dir = knowledge_base_dir
        self.chunk_size = chunk_size
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.vectorizer = None
        self.tfidf_matrix = None

    # ------------------------------------------------------------------
    # INDEXING
    # ------------------------------------------------------------------
    def load_and_chunk(self):
        """Read all documents, split into overlapping chunks, and embed."""
        print("Indexing knowledge base...")
        for path in glob.glob(os.path.join(self.knowledge_base_dir, "*.txt")):
            text = self._read_file(path)
            source = os.path.basename(path)
            for chunk in self._split_into_chunks(text):
                self.chunks.append(chunk)
                self.sources.append(source)
        self._embed()
        print(f"  Indexed {len(self.chunks)} chunks from {len(glob.glob(os.path.join(self.knowledge_base_dir, '*.txt')))} documents")

    def _read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into ~chunk_size-word chunks with overlap, paragraph-aware.

        Markdown headings (#, ##) are attached to the paragraph that follows
        them so each chunk carries meaningful content.
        """
        lines = text.splitlines()
        # Merge heading lines into the following content paragraph
        merged = []
        pending_header = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                pending_header = stripped
            else:
                if pending_header and stripped:
                    merged.append(f"{pending_header}. {stripped}")
                    pending_header = ""
                elif stripped:
                    merged.append(stripped)
        paragraphs = [p for p in merged if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        chunks = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= self.chunk_size:
                chunks.append(para)
            else:
                step = self.chunk_size // 2
                for i in range(0, len(words), step):
                    chunk = " ".join(words[i:i + self.chunk_size])
                    if chunk:
                        chunks.append(chunk)
        return chunks

    def _embed(self):
        """Build TF-IDF vectors for all chunks (works offline, no heavy deps)."""
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """Return the top_k most relevant (chunk, score, source) for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], round(score, 3), self.sources[idx]))
        return results

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------
    def answer(self, query: str, top_k: int = 3) -> dict:
        """End-to-end RAG: retrieve relevant context, then generate an answer."""
        context = self.retrieve(query, top_k)
        if not context:
            return {"answer": "I couldn't find relevant information in the knowledge base.", "sources": []}

        # Combine retrieved chunks into context for the LLM
        context_text = "\n\n".join(c for c, _, _ in context)
        sources = list(dict.fromkeys(src for _, _, src in context))

        answer = self._generate(query, context_text)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [c[:120] + "..." for c, _, _ in context],
        }

    def _generate(self, query: str, context: str) -> str:
        """Generate an answer.

        With OPENAI_API_KEY -> call a real LLM for a fluent grounded answer.
        Without a key -> extractive answer: return the most relevant sentence.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return self._generate_with_llm(query, context, api_key)
        return self._generate_extractive(query, context)

    def _generate_with_llm(self, query: str, context: str, api_key: str) -> str:
        """Call an OpenAI-compatible chat completion API."""
        try:
            import urllib.request
            import json

            system = "You are a helpful assistant. Answer ONLY using the provided context. If the context doesn't contain the answer, say you don't know. Be concise."
            payload = {
                "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                "temperature": 0.2,
            }
            req = urllib.request.Request(
                os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(LLM call failed: {e}. Falling back to extractive answer.)\n\n" + self._generate_extractive(query, context)

    def _generate_extractive(self, query: str, context: str) -> str:
        """Offline fallback: pull out the most relevant sentence from context."""
        # Remove markdown headings and bold markers
        clean_lines = []
        for line in context.splitlines():
            s = line.strip()
            if s.startswith("#"):
                s = s.lstrip("#").strip()  # keep heading text but drop '##'
            clean_lines.append(s)
        clean = " ".join(clean_lines).replace("**", "")
        sentences = [s.strip() for s in clean.split(".") if len(s.strip()) > 25]
        if not sentences:
            return clean[:500]
        # Relevance heuristic: prefer sentences containing query keywords
        query_words = set(w.lower() for w in query.split() if len(w) > 3)
        scored = sorted(
            sentences,
            key=lambda s: sum(1 for w in query_words if w in s.lower()),
            reverse=True,
        )
        top = scored[:2]
        return ". ".join(top) + "." if top else ". ".join(sentences[:2]) + "."


def main():
    engine = RAGEngine("knowledge_base")
    engine.load_and_chunk()

    print("\n" + "=" * 60)
    print("AIHive Knowledge Assistant (RAG)")
    print("=" * 60)
    print("Type a question, or 'quit' to exit.")

    while True:
        try:
            query = input("\nYou: ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break

        result = engine.answer(query)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
