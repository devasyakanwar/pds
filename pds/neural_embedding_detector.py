import os
import json
import numpy as np
import chromadb
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


default_model = "all-MiniLM-L6-v2"
default_data_clean = os.path.join("data", "clean")
default_data_poisoned = os.path.join("data", "poisoned")
default_chroma_db = os.path.join("results", "chroma_db")
default_collection = "clean_corpus"
default_autoencoder_path = os.path.join("results", "autoencoder.pth")
default_metadata_path = os.path.join("results", "autoencoder_metadata.json")

FLAG_THRESHOLD = 2.5
REJECT_THRESHOLD = 3.5


class Autoencoder(nn.Module):
    def __init__(self, input_dim=384, encoding_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def load_documents(data_dir):
    texts = []
    filenames = []

    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        if not os.path.isfile(path):
            continue

        if file.endswith(image_extensions):
            continue

        if file.endswith((".txt", ".py", ".md", ".csv", ".json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    text = f.read()

        elif file.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)

        elif file.endswith(".docx"):
            from docx import Document
            doc = Document(path)
            text = "\n".join(p.text for p in doc.paragraphs)

        else:
            continue 

        texts.append(text)
        filenames.append(file)

    return texts, filenames


model_cache = {}
network_cache = {}
metadata_cache = {}

def get_model(model_name=default_model):
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

def get_autoencoder(model_path=default_autoencoder_path, metadata_path=default_metadata_path):
    if model_path not in network_cache:
        model = Autoencoder(input_dim=384, encoding_dim=16)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        model.eval()
        network_cache[model_path] = model
    
    if metadata_path not in metadata_cache:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata_cache[metadata_path] = json.load(f)
        else:
            metadata_cache[metadata_path] = {"mean_loss": 0.0, "std_loss": 1.0}
            
    return network_cache[model_path], metadata_cache[metadata_path]

def chunk_text(text, model_name=default_model):
    model = get_model(model_name)
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length - 2

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]

    overlap = max_tokens // 10
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks

def generate_embeddings(texts, model_name=default_model):
    model = get_model(model_name)
    embeddings = model.encode(texts)
    return [embedding.tolist() for embedding in embeddings]


def store_in_chromadb(texts, filenames, chroma_path=default_chroma_db, collection_name=default_collection):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name)

    all_ids = []
    all_chunks = []
    all_embeddings = []
    all_metadatas = []

    for text, filename in zip(texts, filenames):
        chunks = chunk_text(text)
        chunk_embeddings = generate_embeddings(chunks)
        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()

        for chunk_idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
            all_ids.append(f"{filename}_chunk{chunk_idx}")
            all_chunks.append(chunk)
            all_embeddings.append(emb)
            all_metadatas.append({
                "filename": filename,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "doc_avg_embedding": json.dumps(avg_embedding),
            })

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadatas,  # type: ignore[arg-type]
    )

def scoring(new_doc_embedding, model_path=default_autoencoder_path, metadata_path=default_metadata_path):
    model, metadata = get_autoencoder(model_path, metadata_path)
    
    mean_loss = metadata["mean_loss"]
    std_loss = metadata["std_loss"]

    x = torch.tensor(new_doc_embedding, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        reconstructed = model(x)
        loss = float(torch.mean((reconstructed - x) ** 2).item())

    # Regularization to prevent dividing by extremely small standard deviations
    min_std = 0.0001
    effective_std = max(std_loss, min_std)
    
    z_score = (loss - mean_loss) / effective_std

    if z_score > REJECT_THRESHOLD:
        decision = "AUTO_REJECT"
    elif z_score > FLAG_THRESHOLD:
        decision = "FLAG_FOR_REVIEW"
    else:
        decision = "ACCEPT"
    
    return z_score, loss, decision

def screen_document(text, filename, model_path=default_autoencoder_path, metadata_path=default_metadata_path):
    chunks = chunk_text(text)
    chunk_embeddings = generate_embeddings(chunks)
    avg_embedding = np.mean(chunk_embeddings, axis=0)

    z_score, loss, decision = scoring(avg_embedding, model_path=model_path, metadata_path=metadata_path)

    if decision == "ACCEPT":
        store_in_chromadb([text], [filename])

    return {
        "filename": filename,
        "chunks": len(chunks),
        "z_score": z_score,
        "distance": loss,
        "decision": decision
    }

def ingest_clean_pipeline(data_dir=default_data_clean):

    texts, filenames = load_documents(data_dir)

    if not texts:
        return

    store_in_chromadb(texts, filenames)

def screen_poisoned_pipeline(poison_dir=default_data_poisoned, model_path=default_autoencoder_path, metadata_path=default_metadata_path):
    texts, filenames = load_documents(poison_dir)

    if not texts:
        return []

    results = []

    for text, filename in zip(texts, filenames):
        result = screen_document(text, filename, model_path=model_path, metadata_path=metadata_path)
        results.append(result)

    return results

def main():
    ingest_clean_pipeline()
    results = screen_poisoned_pipeline()
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
