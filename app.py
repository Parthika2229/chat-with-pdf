"""
app.py — Flask Web Server
=========================
New features in this version:
  - Auto-summary: When a PDF is uploaded, we immediately generate a summary
  - Chat history: Every session keeps a running conversation history so
    the LLM understands follow-up questions like "tell me more about that"
"""

import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from core.ml_engine import (
    process_pdf,
    load_index,
    search_chunks,
    generate_answer,
    generate_summary
)

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
INDEX_FOLDER  = Path("indexes")
UPLOAD_FOLDER.mkdir(exist_ok=True)
INDEX_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Sessions now also store chat_history per session
# chat_history is a list of {role, content} dicts — exactly what Groq expects
sessions: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
# ROUTE: Serve main page
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────
# ROUTE: Upload & Process PDF  (now also auto-summarizes)
# ─────────────────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["pdf"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    session_id = str(uuid.uuid4())[:8]
    pdf_path   = UPLOAD_FOLDER / f"{session_id}.pdf"
    index_path = INDEX_FOLDER  / f"{session_id}.idx"

    file.save(pdf_path)

    try:
        chunks, idf, vocabulary = process_pdf(str(pdf_path), str(index_path))

        # ── Auto-summary ──────────────────────────────────────
        # Take the first ~2000 words of the document as context for the summary.
        # We don't need vector search here — we just want an overview of the whole doc.
        full_text = " ".join(chunk["text"] for chunk in chunks[:8])
        summary   = generate_summary(full_text, file.filename, client)

        # Seed the chat history with the summary as the first assistant message.
        # This means when the user asks follow-ups, the LLM already "knows"
        # what was summarized and can refer back to it naturally.
        initial_history = [
            {
                "role":    "assistant",
                "content": summary
            }
        ]

        sessions[session_id] = {
            "chunks":       chunks,
            "idf":          idf,
            "vocabulary":   vocabulary,
            "filename":     file.filename,
            "num_chunks":   len(chunks),
            "vocab_size":   len(vocabulary),
            "chat_history": initial_history   # ← NEW: conversation memory
        }

        return jsonify({
            "session_id": session_id,
            "filename":   file.filename,
            "num_chunks": len(chunks),
            "vocab_size": len(vocabulary),
            "summary":    summary            # ← sent to frontend to display immediately
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────
# ROUTE: Chat  (now sends full history to the LLM)
# ─────────────────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json()
    session_id = data.get("session_id")
    question   = data.get("question", "").strip()

    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400

    session = sessions.get(session_id)
    if not session:
        index_path = INDEX_FOLDER / f"{session_id}.idx"
        if not index_path.exists():
            return jsonify({"error": "Session not found. Please re-upload your PDF."}), 404
        chunks, idf, vocabulary = load_index(str(index_path))
        session = {
            "chunks": chunks, "idf": idf, "vocabulary": vocabulary,
            "chat_history": []
        }
        sessions[session_id] = session

    try:
        # STEP 1: Vector search for relevant chunks
        relevant_chunks = search_chunks(
            query=question,
            chunks=session["chunks"],
            idf=session["idf"],
            vocabulary=session["vocabulary"],
            top_k=4
        )

        # STEP 2: Generate answer — passing chat history so it has memory
        answer, updated_history = generate_answer(
            query=question,
            relevant_chunks=relevant_chunks,
            client=client,
            chat_history=session["chat_history"]  # ← pass full history
        )

        # STEP 3: Save updated history back to session
        session["chat_history"] = updated_history

        sources = [
            {
                "text":  chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
                "score": chunk["score"],
                "id":    chunk["id"]
            }
            for chunk in relevant_chunks
            if chunk["score"] > 0
        ]

        return jsonify({
            "answer":  answer,
            "sources": sources,
            "query":   question
        })

    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────
# ROUTE: Clear chat history for a session
# ─────────────────────────────────────────────────────────────

@app.route("/clear/<session_id>", methods=["POST"])
def clear_history(session_id):
    """Reset the conversation but keep the document index."""
    if session_id in sessions:
        sessions[session_id]["chat_history"] = []
    return jsonify({"status": "cleared"})


@app.route("/debug/<session_id>")
def debug_session(session_id):
    session = sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    top_idf_words = sorted(session["idf"].items(), key=lambda x: x[1], reverse=True)[:10]
    return jsonify({
        "filename":        session.get("filename", "unknown"),
        "num_chunks":      session["num_chunks"],
        "vocab_size":      session["vocab_size"],
        "history_length":  len(session.get("chat_history", [])),
        "top_idf_words":   top_idf_words,
    })


if __name__ == "__main__":
    print("\n🚀 Chat with PDF is running!")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
