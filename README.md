# 📄 Chat with PDF — Learn ML by Building It

A complete "Chat with PDF" application that teaches you the core concepts
behind modern AI systems — built from scratch, with every algorithm explained.

---

## 🧠 What You'll Learn

| Concept | Where it's implemented |
|---|---|
| Text extraction & preprocessing | `core/ml_engine.py` → `extract_text_from_pdf()` |
| Text chunking with overlap | `core/ml_engine.py` → `chunk_text()` |
| TF-IDF embeddings (from scratch) | `core/ml_engine.py` → `compute_tf()`, `compute_idf()`, `build_tfidf_vectors()` |
| Cosine similarity | `core/ml_engine.py` → `cosine_similarity()` |
| Vector search / retrieval | `core/ml_engine.py` → `search_chunks()` |
| RAG (Retrieval-Augmented Generation) | `core/ml_engine.py` → `generate_answer()` |
| REST API design | `app.py` |

---

## 🏗️ Project Structure

```
chat-with-pdf/
│
├── app.py                  # Flask web server & API routes
│
├── core/
│   └── ml_engine.py        # ALL the ML logic (heavily commented)
│
├── templates/
│   └── index.html          # Frontend chat UI
│
├── uploads/                # Uploaded PDFs are stored here
├── indexes/                # Processed indexes saved here
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / download this project

```bash
cd chat-with-pdf
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key

```bash
cp .env.example .env
# Edit .env and paste your key from https://console.anthropic.com/
```

Or set it directly:
```bash
export ANTHROPIC_API_KEY=your_key_here   # Mac/Linux
set ANTHROPIC_API_KEY=your_key_here      # Windows
```

### 5. Run the app

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔬 How It Works (The ML Pipeline)

```
PDF File
   │
   ▼
┌─────────────────────────────────────────────┐
│  STEP 1: Text Extraction                    │
│  PyMuPDF reads the PDF's internal structure  │
│  and outputs raw text with page markers.     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 2: Chunking                           │
│  Split into ~500 word segments with 100     │
│  word overlap so no idea is cut in half.    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 3: TF-IDF Vectorization               │
│  Each chunk → a sparse vector of numbers.   │
│                                             │
│  TF  = how often a word appears here        │
│  IDF = how rare this word is everywhere     │
│  TF-IDF = TF × IDF → rewards rare, specific │
│  words. Common words ("the") get near-zero. │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  SAVED INDEX (chunks + IDF + vocabulary)    │
│  Persisted to disk — only built ONCE        │
└──────────────────┬──────────────────────────┘
                   │
            User asks question
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 4: Query Vectorization                │
│  The question goes through the same TF-IDF  │
│  process using the saved vocabulary & IDF.  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 5: Cosine Similarity Search           │
│  Compare question vector to every chunk.    │
│                                             │
│  cos(θ) = (A·B) / (|A| × |B|)              │
│                                             │
│  → Returns top-4 most similar chunks        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  STEP 6: RAG Generation                     │
│  Top chunks + question → Claude API         │
│  "Answer ONLY based on these excerpts"      │
│  → Grounded, citation-backed answer         │
└─────────────────────────────────────────────┘
```

---

## 💡 Understanding TF-IDF

TF-IDF is a classic NLP technique that turns text into numbers:

**Term Frequency (TF):** How often does a word appear *in this chunk*?
```
TF("machine", chunk) = occurrences("machine") / total_words_in_chunk
```

**Inverse Document Frequency (IDF):** How rare is this word *across all chunks*?
```
IDF("machine") = log(total_chunks / chunks_containing_"machine")
```

The key insight: **rare words are more informative.**
- "machine" appearing in 3 of 200 chunks → IDF ≈ 4.2 (very meaningful)  
- "the" appearing in 198 of 200 chunks   → IDF ≈ 0.01 (almost meaningless)

---

## 📐 Understanding Cosine Similarity

Think of each chunk as an arrow pointing in high-dimensional space.
Two chunks about "neural networks" point in similar directions.
A chunk about "cooking recipes" points a completely different way.

```
similarity = cos(angle between vectors)
           = 1.0  → identical topics
           = 0.5  → somewhat related  
           = 0.0  → completely unrelated
```

The formula:
```python
cos(θ) = dot_product(A, B) / (magnitude(A) * magnitude(B))
```

We use sparse vectors (only storing non-zero values) for efficiency —
a vocabulary of 10,000 words where only 50 appear in a chunk means
9,950 zeros we don't need to store.

---

## 🔧 Extending the Project

Once you've built and understood this project, try these upgrades:

### 1. Better Embeddings
Replace TF-IDF with neural embeddings using `sentence-transformers`:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([chunk["text"] for chunk in chunks])
```
These capture *semantic meaning* not just word overlap.

### 2. Proper Vector Database
Replace the in-memory search with ChromaDB:
```bash
pip install chromadb
```

### 3. Streaming Responses
Add streaming so the answer appears word-by-word (like ChatGPT):
```python
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        yield text
```

### 4. Chat History
Pass previous messages to the LLM so it understands follow-up questions.

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'`**
```bash
pip install PyMuPDF
```

**`AuthenticationError`**
Make sure `ANTHROPIC_API_KEY` is set in your `.env` file.

**Answer quality is poor**
Try increasing `top_k` in `search_chunks()` from 4 to 6, or reduce `chunk_size` for denser documents.

---

## 📚 Further Reading

- [TF-IDF explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)  
- [RAG paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/) — the next step after TF-IDF
- [ChromaDB](https://docs.trychroma.com/) — a real vector database
