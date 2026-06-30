# Neural Embedding Detector — Summary

## What It Does

Detects poisoned or anomalous documents using a trained autoencoder that learns the embedding patterns of clean documents. Documents that the autoencoder fails to reconstruct well are flagged as suspicious.

## How It Works

### Core Idea

An autoencoder is trained to compress and reconstruct the embeddings of clean documents. Since it only sees clean data during training, it learns what "normal" embeddings look like. When a new document arrives, its embedding is passed through the autoencoder — if the reconstruction error is unusually high, the document is likely poisoned or off-topic.

### Architecture

```
Input (384-dim) → Linear(384→128) → ReLU → Linear(128→16) → ReLU    [Encoder]
                                                    ↓
                                          Latent Space (16-dim)
                                                    ↓
                  Linear(16→128) → ReLU → Linear(128→384)            [Decoder]
                                                    ↓
                                            Output (384-dim)
```

The 16-dimensional bottleneck forces the network to learn only the essential structure of clean embeddings. Poisoned documents fall outside these learned patterns and reconstruct poorly.

### Embedding Model

All text is embedded using `all-MiniLM-L6-v2` from SentenceTransformers, producing 384-dimensional vectors. This model has a max token limit of 256 tokens.

### Chunking

Documents longer than 254 tokens (256 minus 2 special tokens) are split into overlapping chunks:
- Each chunk is 254 tokens max
- Consecutive chunks overlap by 10% (25 tokens) to preserve boundary context
- Short documents that fit within the limit are kept as-is

At screening time, all chunk embeddings are averaged into a single document-level vector before scoring.

## Training

Training is handled by `neural_embedding_training.py`. There are two modes:

### Mode 1: Train from Files (Default)

Reads clean documents from disk, ingests them into ChromaDB, and trains the autoencoder on the resulting chunk embeddings.

```bash
python -m pds.neural_embedding_training
```

What happens:
1. Loads all documents from `data/clean/`
2. Stores them in ChromaDB (for use by other detectors)
3. Chunks every document and generates embeddings for each chunk
4. Trains the autoencoder on all individual chunk embeddings for 100 epochs
5. Computes mean and std of reconstruction losses across all training chunks
6. Saves `results/autoencoder.pth` (model weights) and `results/autoencoder_metadata.json` (loss statistics)

Use this mode when starting fresh or when you have new clean documents to add.

### Mode 2: Train from Database (`--from-db`)

Skips file loading entirely and trains on embeddings already stored in ChromaDB.

```bash
python -m pds.neural_embedding_training --from-db
```

What happens:
1. Pulls all existing embeddings from the ChromaDB `clean_corpus` collection
2. Trains the autoencoder on those embeddings for 100 epochs
3. Saves updated model weights and metadata

Use this mode when:
- Clean documents have already been ingested into ChromaDB (from a previous training run or from accepted screened documents)
- You want to retrain the autoencoder to incorporate newly accepted documents without re-reading files from disk

### Training Flags

| Flag | Default | Description |
|---|---|---|
| `--clean-dir` | `data/clean` | Directory containing clean training documents |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 0.005 | Learning rate for Adam optimizer |
| `--encoding-dim` | 16 | Bottleneck dimension of the autoencoder |
| `--reset` | off | Deletes existing ChromaDB, model weights, and metadata before training |
| `--from-db` | off | Trains from ChromaDB embeddings instead of reading files from disk |

### Training Output

Two files are saved to `results/`:

**autoencoder.pth** — The trained model weights.

**autoencoder_metadata.json** — Baseline loss statistics:
```json
{
    "mean_loss": 0.001128,
    "std_loss": 0.000224,
    "epochs": 100,
    "num_chunks": 559
}
```

These statistics define what "normal" reconstruction error looks like. New documents are compared against these values.

## Screening

### How Scoring Works

1. The new document is chunked and each chunk is embedded
2. All chunk embeddings are averaged into a single 384-dim vector
3. This vector is passed through the trained autoencoder
4. The reconstruction MSE loss is computed
5. A z-score is calculated: `z = (loss - mean_loss) / std_loss`
6. Decision is made based on the z-score:

| Z-Score | Decision | Meaning |
|---|---|---|
| ≤ 2.5 | ACCEPT | Normal document, also added to ChromaDB |
| 2.5 – 3.5 | FLAG_FOR_REVIEW | Suspicious, needs human review |
| > 3.5 | AUTO_REJECT | Very anomalous, rejected automatically |

### Screening Pipeline

```
New document → chunk → embed each chunk → average embeddings
                                                ↓
                                    Feed through autoencoder
                                                ↓
                                    Compute reconstruction MSE
                                                ↓
                                    Calculate z-score against baseline
                                                ↓
                                    ACCEPT / FLAG / REJECT
```

### Run Screening

Screen all documents in `data/poisoned/` against the trained model:

```bash
python -m pds.neural_embedding_detector
```

This first ingests clean documents into ChromaDB (if not already present), then screens each document in the poisoned directory.

### Use as a Module

```python
from pds.neural_embedding_detector import screen_document, ingest_clean_pipeline

# Ingest clean baseline (one-time)
ingest_clean_pipeline("data/clean")

# Screen a new document
result = screen_document("Some document text...", "new_doc.txt")
print(result["decision"])  # ACCEPT / FLAG_FOR_REVIEW / AUTO_REJECT
```

## ChromaDB Storage

Each chunk is stored as a separate entry in the `clean_corpus` collection:

| Field | Content |
|---|---|
| `id` | `{filename}_chunk{index}` |
| `document` | Chunk text |
| `embedding` | 384-dim chunk embedding |
| `metadata.filename` | Source document name |
| `metadata.chunk_index` | Position of chunk within document |
| `metadata.total_chunks` | Total chunks for this document |
| `metadata.doc_avg_embedding` | JSON-serialized average of all chunk embeddings |

Duplicate handling: IDs are based on filename, so screening the same document twice will upsert (overwrite, not duplicate). Different documents get different IDs.

## Supported File Types

| Extension | How It's Read |
|---|---|
| `.txt`, `.py`, `.md`, `.csv`, `.json` | Raw text (`f.read()`) |
| `.pdf` | Text extraction via pypdf |
| `.docx` | Paragraph extraction via python-docx |
| Images (`.jpg`, `.png`, etc.) | Skipped |

## Limitations

- The autoencoder only knows topics it was trained on. Clean documents on entirely new topics may be flagged as anomalous. This is mitigated by using the neural detector alongside other PDS detectors (instruction detector, perplexity scorer, hubness detector) via the aggregator.
- CSV and JSON files are read as raw text including formatting characters, not parsed into structured data.

## Constants

| Constant | Value | Purpose |
|---|---|---|
| `FLAG_THRESHOLD` | 2.5 | Z-score above this → flag for human review |
| `REJECT_THRESHOLD` | 3.5 | Z-score above this → auto-reject |
| `default_model` | `all-MiniLM-L6-v2` | Embedding model (384-dim, 256 max tokens) |
