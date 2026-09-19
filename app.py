"""
Flask Web App for the RAG Chatbot
==================================
Serves a browser chat interface powered by the RAGEngine.
Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.

For higher quality answers, set OPENAI_API_KEY (and optionally
OPENAI_BASE_URL / LLM_MODEL) before running. Without a key it uses the
offline extractive answerer, so the demo works with zero setup.
"""

import os

from flask import Flask, render_template, request, jsonify

from rag_engine import RAGEngine

app = Flask(__name__)

# Load knowledge base once at startup
engine = RAGEngine("knowledge_base")
engine.load_and_chunk()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = (data or {}).get("message", "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    result = engine.answer(query)
    return jsonify(
        {
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved": result["retrieved_chunks"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
