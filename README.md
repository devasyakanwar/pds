# Poison Detection System (PDS)

A system to detect poisoned documents in RAG (Retrieval-Augmented Generation) datasets and corpora using embedding-based anomaly detection.

## How It Works

1. **Ingest** clean baseline documents → chunk → embed → store in ChromaDB
2. **Screen** incoming documents → chunk → embed → compare against clean corpus using kNN z-score
3. **Decision**: ACCEPT / FLAG_FOR_REVIEW / AUTO_REJECT based on z-score thresholds

## Project Structure

```
pds/
├── data/
│   ├── clean/                    # Clean baseline documents
│   └── poisoned/                 # Documents to screen
├── pds/
│   ├── __init__.py
│   ├── embeddingdetector.py      # Embedding + kNN anomaly detection
│   ├── perplexity_scorer.py      # LLM perplexity based detection
│   ├── instruction_detector.py   # Instruction injection detection
│   ├── hubness_detector.py       # Hubness (nearest neighbor) detection
│   └── aggregator.py             # Composite scoring
├── experiments/
│   ├── 01_baseline_corpus.ipynb
│   ├── 02_poison_crafting.ipynb
│   ├── 03_pds_evaluation.ipynb
│   └── 04_rag_integration.ipynb
├── results/                      # ChromaDB storage, evaluation outputs
├── EMBEDDING_DETECTOR_SUMMARY.md # Detailed summary of embeddingdetector.py
├── README.md
└── requirements.txt
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/macOS**: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the full pipeline (ingest + screen)

From the project root directory:

```bash
python pds/embeddingdetector.py
```

This will:
1. Load and ingest documents from `data/clean/` into ChromaDB
2. Screen documents from `data/poisoned/` against the clean corpus
3. Print results for each screened document

### Example output

```
{'filename': 'clean_test_doc.txt', 'chunks': 3, 'z_score': 1.12, 'distance': 0.30, 'decision': 'ACCEPT'}
{'filename': 'poisoned_doc_01.txt', 'chunks': 2, 'z_score': 6.55, 'distance': 0.74, 'decision': 'AUTO_REJECT'}
```

### Use as a module

```python
from pds.embeddingdetector import ingest_clean_pipeline, screen_document

# Step 1: Ingest clean docs (one-time)
ingest_clean_pipeline("data/clean")

# Step 2: Screen a new document
result = screen_document("Some document text...", "new_doc.txt")
print(result["decision"])  # ACCEPT / FLAG_FOR_REVIEW / AUTO_REJECT
```

## Configuration

Edit the constants at the top of `pds/embeddingdetector.py`:

| Constant | Default | Description |
|---|---|---|
| `FLAG_THRESHOLD` | 2.5 | Z-score above this → flag for review |
| `REJECT_THRESHOLD` | 3.5 | Z-score above this → auto-reject |
| `K_NEIGHBORS` | 200 | Nearest neighbors for kNN scoring |
| `default_model` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model |

## Adding Your Own Documents

- Place clean baseline documents in `data/clean/`
- Place documents to screen in `data/poisoned/`
- Supported formats: `.txt`, `.pdf`, `.docx`, `.csv`, `.json`, `.py`, `.md`
- Re-run the pipeline after adding new clean documents (clears and re-ingests)
