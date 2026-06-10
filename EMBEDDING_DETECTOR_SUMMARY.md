# Embedding Detector — Summary

## What It Does

Detects poisoned/anomalous documents in a RAG corpus using embedding-based anomaly detection. Clean documents are ingested into ChromaDB, and incoming documents are screened against them using a **kNN z-score** approach.

## How It Works

### Pipeline

```
1. INGEST (one-time)
   data/clean/*.txt → load → chunk → embed → store in ChromaDB

2. SCREEN (per document)
   new document → chunk → embed → avg embedding
                                       ↓
                              query ChromaDB for 200 nearest neighbors (O(log n))
                                       ↓
                              compute centroid of those 200 neighbors
                              compute z-score against that local neighborhood
                                       ↓
                              z ≤ 2.5  → ACCEPT (also stored in ChromaDB)
                              z > 2.5  → FLAG_FOR_REVIEW
                              z > 3.5  → AUTO_REJECT
```

### Chunking

Documents are split based on the embedding model's token limit (`max_seq_length - 2` for special tokens). For `all-MiniLM-L6-v2`, that's 254 tokens per chunk with 10% overlap between consecutive chunks. Small documents that fit within the limit are kept as-is.

Each chunk is embedded individually. The **document-level average embedding** (mean of all chunk embeddings) is stored in ChromaDB metadata as `doc_avg_embedding` for efficient retrieval.

### Scoring (kNN approach)

Instead of computing distances against the entire corpus (O(n)), scoring uses ChromaDB's HNSW index to find the K nearest neighbors (O(log n)), then computes the z-score against just those neighbors:

1. Query ChromaDB for K=200 nearest chunk embeddings
2. Compute centroid of those K neighbors
3. Compute cosine distances of each neighbor to the centroid → mean_dist, std_dist
4. Compute cosine distance of the new document to the centroid → new_doc_dist
5. z_score = (new_doc_dist - mean_dist) / std_dist

This scales to millions of documents with constant-time scoring (~200 operations regardless of corpus size).

## Functions

| Function | Purpose |
|---|---|
| `load_documents(data_dir)` | Reads .txt, .pdf, .docx, .csv, .json, .py, .md files from a directory |
| `get_model(model_name)` | Loads and caches SentenceTransformer model |
| `chunk_text(text)` | Splits text into token-limited chunks with 10% overlap |
| `generate_embeddings(texts)` | Encodes texts into 384-dim vectors using SentenceTransformer |
| `store_in_chromadb(texts, filenames)` | Chunks, embeds, and stores documents with avg embedding in metadata |
| `scoring(new_doc_embedding)` | kNN-based z-score anomaly scoring via ChromaDB query |
| `screen_document(text, filename)` | Full screening pipeline for a single document |
| `ingest_clean_pipeline(data_dir)` | Batch ingest clean baseline documents |
| `screen_poisoned_pipeline(poison_dir)` | Screen all documents in a directory |

## ChromaDB Storage Format

Each chunk is stored as a separate entry:

| Field | Content |
|---|---|
| `id` | `doc{i}_chunk{j}` |
| `document` | Chunk text |
| `embedding` | 384-dim chunk embedding (used for HNSW queries) |
| `metadata.filename` | Source document name |
| `metadata.chunk_index` | Position of chunk within document |
| `metadata.total_chunks` | Total chunks for this document |
| `metadata.doc_avg_embedding` | JSON-serialized average of all chunk embeddings |

## Constants

| Constant | Value | Purpose |
|---|---|---|
| `FLAG_THRESHOLD` | 2.5 | Z-score above this → flag for human review |
| `REJECT_THRESHOLD` | 3.5 | Z-score above this → auto-reject |
| `K_NEIGHBORS` | 200 | Number of nearest neighbors for scoring |
| `default_model` | `all-MiniLM-L6-v2` | Embedding model (384-dim, 256 max tokens) |
