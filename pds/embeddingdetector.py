import os
import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_distance


default_model="all-MiniLM-L6-v2"
default_data_clean=os.path.join("data", "clean")
default_data_poisoned=os.path.join("data", "poisoned")
default_chroma_db=os.path.join("results", "chroma_db")
default_collection = "clean_corpus"

FLAG_THRESHOLD = 2.5
REJECT_THRESHOLD = 3.5
K_NEIGHBORS = 200

import os

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
            with open(path, "r", encoding="utf-8") as f:
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

def get_model(model_name=default_model):
    if model_name not in model_cache:
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

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

    client=chromadb.PersistentClient(path=chroma_path)
    collection=client.get_or_create_collection(name=collection_name)

    all_ids = []
    all_chunks = []
    all_embeddings = []
    all_metadatas = []

    for doc_idx, (text, filename) in enumerate(zip(texts, filenames)):
        chunks = chunk_text(text)
        chunk_embeddings = generate_embeddings(chunks)
        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()

        for chunk_idx, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):
            all_ids.append(f"doc{doc_idx}_chunk{chunk_idx}")
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

def scoring(new_doc_embedding, chroma_path=default_chroma_db, collection_name=default_collection):
    client=chromadb.PersistentClient(path=chroma_path)
    collection=client.get_collection(name=collection_name)

    n_docs = collection.count()
    k = min(K_NEIGHBORS, n_docs)

    results = collection.query(
        query_embeddings=[new_doc_embedding.tolist()],
        n_results=k,
        include=["embeddings"]
    )

    neighbor_embeddings = np.array(results["embeddings"][0])
    centroid = np.mean(neighbor_embeddings, axis=0)
    distances = [cosine_distance(emb, centroid) for emb in neighbor_embeddings]

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    new_doc_dist = cosine_distance(new_doc_embedding, centroid)
    if std_dist < 0.01:
        z_score = new_doc_dist * 5.0
    else:
        z_score = (new_doc_dist - mean_dist) / std_dist

    if z_score > REJECT_THRESHOLD:
        decision = "AUTO_REJECT"
    elif z_score > FLAG_THRESHOLD:
        decision = "FLAG_FOR_REVIEW"
    else:
        decision = "ACCEPT"
    
    return z_score, new_doc_dist, decision

def screen_document(text, filename):
    chunks = chunk_text(text)
    chunk_embeddings = generate_embeddings(chunks)
    avg_embedding = np.mean(chunk_embeddings, axis=0)

    z_score, dist, decision = scoring(avg_embedding)

    if decision == "ACCEPT":
        store_in_chromadb([text], [filename])

    return {
        "filename": filename,
        "chunks": len(chunks),
        "z_score": z_score,
        "distance": dist,
        "decision": decision
    }

def ingest_clean_pipeline(data_dir=default_data_clean):

    texts, filenames = load_documents(data_dir)

    if not texts:
        return

    store_in_chromadb(texts, filenames)

def screen_poisoned_pipeline(poison_dir=default_data_poisoned):

    texts, filenames = load_documents(poison_dir)

    if not texts:
        return []

    results = []

    for text, filename in zip(texts, filenames):
        result = screen_document(text, filename)
        results.append(result)

    return results

def main():
    ingest_clean_pipeline()
    results = screen_poisoned_pipeline()
    for r in results:
        print(r)

if __name__ == "__main__":
    main()