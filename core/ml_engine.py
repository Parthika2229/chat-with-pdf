"""
ml_engine.py — The Brain of Chat with PDF
==========================================
This file contains all the ML/AI logic, built from scratch so you can
understand every step. No magic black boxes here.

Key Concepts Covered:
  1. PDF Text Extraction
  2. Text Chunking (with overlap)
  3. TF-IDF Embeddings (vectors that represent meaning)
  4. Cosine Similarity (measuring how "close" two vectors are)
  5. Retrieval-Augmented Generation (RAG) pattern
"""

import re
import math
import json
import pickle
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# STEP 1: PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF (fitz).

    PyMuPDF reads the PDF's internal structure and pulls out all
    text content, preserving the reading order of the document.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    full_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" mode preserves layout
        if text.strip():
            full_text.append(f"[Page {page_num + 1}]\n{text}")

    doc.close()
    return "\n\n".join(full_text)


# ─────────────────────────────────────────────────────────────
# STEP 2: TEXT CHUNKING
# ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Split a large text into overlapping chunks.

    WHY CHUNK?
    LLMs have a "context window" — a limit on how much text they can process
    at once. A 100-page PDF might be 80,000 words; we can only send ~2,000
    relevant words at a time. So we split the text into small pieces.

    WHY OVERLAP?
    If a sentence is split across two chunks, we'd lose the context. By
    overlapping chunks by 'overlap' words, we ensure no important idea
    is cut in half.

    Example with chunk_size=5, overlap=2:
      Text: [A B C D E F G H I J]
      Chunk 1: [A B C D E]
      Chunk 2:       [D E F G H]   ← overlaps by 2 words
      Chunk 3:             [G H I J]
    """
    # Split text into sentences first for cleaner boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk_words = []
    current_word_count = 0
    chunk_index = 0

    for sentence in sentences:
        words = sentence.split()
        
        # If adding this sentence exceeds our limit, save the current chunk
        if current_word_count + len(words) > chunk_size and current_chunk_words:
            chunk_text_str = " ".join(current_chunk_words)
            chunks.append({
                "id": chunk_index,
                "text": chunk_text_str,
                "word_count": current_word_count
            })
            chunk_index += 1

            # Keep the last 'overlap' words for the next chunk
            overlap_words = current_chunk_words[-overlap:] if len(current_chunk_words) > overlap else current_chunk_words[:]
            current_chunk_words = overlap_words + words
            current_word_count = len(current_chunk_words)
        else:
            current_chunk_words.extend(words)
            current_word_count += len(words)

    # Don't forget the last chunk
    if current_chunk_words:
        chunks.append({
            "id": chunk_index,
            "text": " ".join(current_chunk_words),
            "word_count": current_word_count
        })

    return chunks


# ─────────────────────────────────────────────────────────────
# STEP 3: TF-IDF EMBEDDINGS (built from scratch!)
# ─────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Convert text into a list of clean tokens (words).
    We lowercase and remove punctuation so 'Python' == 'python'.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove punctuation
    tokens = text.split()
    
    # Remove very common "stop words" that carry little meaning
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'this',
        'that', 'these', 'those', 'it', 'its', 'from', 'as', 'not',
        'also', 'can', 'which', 'there', 'their', 'they', 'what', 'how'
    }
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Term Frequency (TF): How often does each word appear in THIS document?

    TF(word) = count(word in doc) / total words in doc

    A word that appears 10 times in a 100-word doc has TF = 0.1
    A word that appears 1 time  in a 100-word doc has TF = 0.01

    This tells us: "Is this word IMPORTANT to this specific chunk?"
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def compute_idf(all_chunks_tokens: list[list[str]]) -> dict[str, float]:
    """
    Inverse Document Frequency (IDF): How RARE is this word across ALL chunks?

    IDF(word) = log(total_docs / docs_containing_word)

    If 'machine' appears in 5 out of 100 chunks:  IDF = log(100/5)  = 3.0
    If 'the'     appears in 99 out of 100 chunks:  IDF = log(100/99) = 0.01

    KEY INSIGHT: Rare words are MORE meaningful than common ones.
    'quantum' appearing in a chunk is much more informative than 'the'.
    IDF punishes common words and rewards rare, specific ones.
    """
    num_docs = len(all_chunks_tokens)
    idf_scores = {}
    
    # Count how many chunks contain each word
    doc_frequency = Counter()
    for tokens in all_chunks_tokens:
        unique_tokens = set(tokens)  # each word counted once per chunk
        for token in unique_tokens:
            doc_frequency[token] += 1

    # IDF formula: log(N / df) — higher for rarer words
    for word, df in doc_frequency.items():
        idf_scores[word] = math.log(num_docs / df)

    return idf_scores


def build_tfidf_vectors(chunks: list[dict]) -> tuple[list[dict], dict[str, float], list[str]]:
    """
    Build TF-IDF vectors for all chunks.

    TF-IDF = TF × IDF
    This gives each word a score that is:
      - HIGH if the word appears often IN this chunk (high TF)
      - HIGH if the word is rare ACROSS all chunks  (high IDF)
      - LOW  if the word is common everywhere        (low IDF)

    The result is a VECTOR — a list of numbers, one per word in the vocabulary.
    Similar chunks will have similar vectors. This is how we find relevant text!

    Example vector for a chunk about "neural networks":
      vocabulary: ['neural', 'network', 'layer', 'weight', 'cat', 'dog', ...]
      vector:     [0.42,     0.38,      0.21,    0.19,     0.0,   0.0,  ...]
    """
    # Tokenize all chunks
    all_tokens = [tokenize(chunk["text"]) for chunk in chunks]
    
    # Build vocabulary (all unique words across all chunks)
    vocabulary = sorted(set(token for tokens in all_tokens for token in tokens))
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    
    # Compute IDF across all chunks
    idf = compute_idf(all_tokens)
    
    # Compute TF-IDF vector for each chunk
    enriched_chunks = []
    for chunk, tokens in zip(chunks, all_tokens):
        tf = compute_tf(tokens)
        
        # Build sparse vector: {word_index: tfidf_score}
        # We use sparse representation (only non-zero values) to save memory
        vector = {}
        for word, tf_score in tf.items():
            if word in vocab_index and word in idf:
                tfidf_score = tf_score * idf[word]
                if tfidf_score > 0:
                    vector[vocab_index[word]] = tfidf_score
        
        enriched_chunks.append({
            **chunk,
            "vector": vector,
            "tokens": tokens
        })
    
    return enriched_chunks, idf, vocabulary


# ─────────────────────────────────────────────────────────────
# STEP 4: COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    Measure the similarity between two vectors using cosine similarity.

    INTUITION: Think of vectors as arrows pointing in space.
    - Two chunks about the same topic point in similar directions → high similarity
    - Two unrelated chunks point in different directions → low similarity

    Cosine similarity measures the ANGLE between two arrows:
      - cos(0°)  = 1.0  → identical direction (very similar)
      - cos(90°) = 0.0  → perpendicular    (unrelated)
      - cos(180°) = -1.0 → opposite         (antonyms, rare in practice)

    Formula: cos(θ) = (A · B) / (|A| × |B|)
    Where A · B is the dot product and |A| is the magnitude (length) of A.

    WHY NOT EUCLIDEAN DISTANCE?
    Cosine similarity ignores vector magnitude (length), focusing only on
    direction. This means a short chunk and a long chunk about the same
    topic will still score high — the LENGTH of the document doesn't matter,
    only its CONTENT direction.
    """
    if not vec_a or not vec_b:
        return 0.0

    # Dot product: sum of (a_i × b_i) for all shared dimensions
    dot_product = sum(
        vec_a[idx] * vec_b[idx]
        for idx in vec_a
        if idx in vec_b
    )

    # Magnitudes: square root of sum of squares
    magnitude_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def search_chunks(query: str, chunks: list[dict], idf: dict, vocabulary: list[str], top_k: int = 4) -> list[dict]:
    """
    Find the most relevant chunks for a given query.

    This is the RETRIEVAL part of RAG (Retrieval-Augmented Generation).

    Steps:
      1. Convert the query into a TF-IDF vector (same process as chunks)
      2. Compare query vector to every chunk vector using cosine similarity
      3. Return the top-k most similar chunks

    These top chunks become the "context" we pass to the LLM.
    """
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}
    
    # Vectorize the query using the SAME vocabulary and IDF scores
    query_tokens = tokenize(query)
    query_tf = compute_tf(query_tokens)
    
    query_vector = {}
    for word, tf_score in query_tf.items():
        if word in vocab_index and word in idf:
            tfidf_score = tf_score * idf[word]
            if tfidf_score > 0:
                query_vector[vocab_index[word]] = tfidf_score

    # Score every chunk against the query
    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        scored_chunks.append({**chunk, "score": round(score, 4)})

    # Sort by score descending and return top-k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]


# ─────────────────────────────────────────────────────────────
# STEP 5: LLM ANSWER GENERATION (using Anthropic API)
# ─────────────────────────────────────────────────────────────

def generate_answer(query: str, relevant_chunks: list[dict], client) -> str:
    """
    Generate a natural language answer using the Groq API.

    This is the GENERATION part of RAG (Retrieval-Augmented Generation).

    The pattern:
      1. Retrieve relevant chunks (done above)
      2. Build a prompt that includes those chunks as context
      3. Ask the LLM to answer the question ONLY based on that context

    WHY THIS PATTERN?
    Without RAG, LLMs can hallucinate (make things up). By explicitly
    providing the relevant text and saying "only use this", we ground
    the answer in real document content.
    """
    if not relevant_chunks or all(c["score"] == 0 for c in relevant_chunks):
        return "I couldn't find relevant information in the document to answer your question."

    # Build context string from top chunks
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(f"[Excerpt {i+1} — relevance score: {chunk['score']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # The prompt is carefully crafted:
    # - Give the LLM a clear role
    # - Provide the document context
    # - Ask it to cite its sources
    # - Tell it to admit uncertainty (prevents hallucination)
    prompt = f"""You are a helpful assistant answering questions about a PDF document.

Here are the most relevant excerpts from the document:

{context}

Based ONLY on the excerpts above, answer this question:
{query}

Rules:
- Only use information from the excerpts provided
- If the excerpts don't contain enough information, say so clearly
- Be concise and direct
- If you quote directly, mention it's from the document"""

    # Groq uses an OpenAI-compatible format: chat completions with a messages array
    # We use llama-3.3-70b — fast, free on Groq's free tier, and very capable
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# PERSISTENCE: Save & Load the Index
# ─────────────────────────────────────────────────────────────

def save_index(chunks: list[dict], idf: dict, vocabulary: list[str], path: str):
    """
    Save the processed index to disk so we don't re-process the PDF on every query.
    This is analogous to how a real vector database (like Pinecone, ChromaDB) works.
    """
    index = {
        "chunks": [{k: v for k, v in c.items() if k != "tokens"} for c in chunks],
        "idf": idf,
        "vocabulary": vocabulary
    }
    with open(path, 'wb') as f:
        pickle.dump(index, f)


def load_index(path: str) -> tuple[list[dict], dict, list[str]]:
    """Load a previously saved index from disk."""
    with open(path, 'rb') as f:
        index = pickle.load(f)
    return index["chunks"], index["idf"], index["vocabulary"]


def process_pdf(pdf_path: str, index_path: str) -> tuple[list[dict], dict, list[str]]:
    """
    Full pipeline: PDF → Text → Chunks → TF-IDF Vectors → Saved Index

    This runs once per PDF and saves the result. Subsequent queries
    just load the index instead of reprocessing.
    """
    print(f"📄 Extracting text from {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    
    print(f"✂️  Chunking text (length: {len(text.split())} words)...")
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    print(f"   Created {len(chunks)} chunks")
    
    print(f"🔢 Building TF-IDF vectors...")
    chunks_with_vectors, idf, vocabulary = build_tfidf_vectors(chunks)
    print(f"   Vocabulary size: {len(vocabulary)} unique terms")
    
    print(f"💾 Saving index to {index_path}...")
    save_index(chunks_with_vectors, idf, vocabulary, index_path)
    
    print(f"✅ Processing complete!")
    return chunks_with_vectors, idf, vocabulary


# ─────────────────────────────────────────────────────────────
# NEW: AUTO-SUMMARY
# ─────────────────────────────────────────────────────────────

def generate_summary(full_text: str, filename: str, client) -> str:
    """
    Generate an automatic summary the moment a PDF is uploaded.

    We take the first chunk of text (a broad sample of the doc) and
    ask the LLM to produce a structured overview with key points.
    This gives the user an instant understanding of what's in the doc
    before they even ask a question.
    """
    # Limit text to ~3000 words to stay within token limits
    words = full_text.split()
    sample = " ".join(words[:3000])

    prompt = f"""You are analyzing a document called "{filename}".

Here is the content:
{sample}

Please provide:
1. **What this document is about** (2-3 sentences)
2. **Key topics covered** (bullet points, max 6)
3. **Most important findings or facts** (bullet points, max 5)
4. **Who this document is for** (1 sentence)

Be concise and use simple language."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# UPDATED: GENERATE ANSWER WITH CHAT HISTORY
# ─────────────────────────────────────────────────────────────

def generate_answer(query: str, relevant_chunks: list[dict], client, chat_history: list = None) -> tuple:
    """
    Generate an answer using Groq, now with full conversation memory.

    HOW CHAT HISTORY WORKS:
    Instead of sending a single message, we send the ENTIRE conversation
    so far as a list of {role, content} pairs. This is how ChatGPT and
    every modern chatbot maintains context across turns.

    Example history sent to the LLM:
      [
        {"role": "assistant", "content": "This document is about..."},  ← auto-summary
        {"role": "user",      "content": "What are the main risks?"},
        {"role": "assistant", "content": "The main risks are..."},
        {"role": "user",      "content": "Tell me more about the first one"}  ← follow-up works!
      ]

    Without history, "tell me more about the first one" would make no sense.
    With history, the LLM understands exactly what "the first one" refers to.

    Returns: (answer_string, updated_history_list)
    """
    if chat_history is None:
        chat_history = []

    if not relevant_chunks or all(c["score"] == 0 for c in relevant_chunks):
        return "I couldn't find relevant information in the document to answer your question.", chat_history

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(f"[Excerpt {i+1} — relevance: {chunk['score']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    # The system prompt sets the LLM's role for the WHOLE conversation
    system_prompt = """You are a helpful assistant that answers questions about a PDF document.
You have access to relevant excerpts retrieved from the document.
Always base your answers on the provided excerpts.
If a follow-up question refers to something from earlier in the conversation, use that context.
If the excerpts don't contain enough information, say so clearly."""

    # Build the new user message — includes the retrieved context + the question
    new_user_message = f"""Relevant excerpts from the document:
{context}

Question: {query}"""

    # Append the new user message to history
    messages_to_send = chat_history + [
        {"role": "user", "content": new_user_message}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + messages_to_send,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    # Update history: add both the user question AND the assistant answer
    # We store the clean question (not the context-padded version) for readability
    updated_history = chat_history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer}
    ]

    # Keep history to last 10 exchanges (20 messages) to avoid token limit issues
    if len(updated_history) > 20:
        updated_history = updated_history[-20:]

    return answer, updated_history
