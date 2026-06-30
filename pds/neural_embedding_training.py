import os
import sys
import shutil
import json
import argparse
import numpy as np
import chromadb
import torch
import torch.nn as nn
import torch.optim as optim

from pds.neural_embedding_detector import (
    Autoencoder,
    load_documents,
    chunk_text,
    generate_embeddings,
    store_in_chromadb,
    default_data_clean,
    default_chroma_db,
    default_collection,
    default_autoencoder_path,
    default_metadata_path,
)


def get_embeddings_from_db(chroma_path=default_chroma_db, collection_name=default_collection):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)

    results = collection.get(include=["embeddings"])
    if results["embeddings"] is None or len(results["embeddings"]) == 0:
        return None

    return np.array(results["embeddings"])


def train(clean_dir=default_data_clean, epochs=100, lr=0.005, encoding_dim=16, reset=False, from_db=False):

    if reset:
        if os.path.isdir(default_chroma_db):
            shutil.rmtree(default_chroma_db)
        for p in (default_autoencoder_path, default_metadata_path):
            if os.path.exists(p):
                os.remove(p)

    if from_db:
        X_train = get_embeddings_from_db()
        if X_train is None:
            sys.exit(1)
    else:
        texts, filenames = load_documents(clean_dir)
        if not texts:
            sys.exit(1)

        store_in_chromadb(texts, filenames)

        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text))

        if not all_chunks:
            sys.exit(1)

        chunk_embeddings = generate_embeddings(all_chunks)
        X_train = np.array(chunk_embeddings)

    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        losses = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()

    mean_loss = float(np.mean(losses))
    std_loss = float(np.std(losses))

    os.makedirs(os.path.dirname(default_autoencoder_path), exist_ok=True)
    torch.save(model.state_dict(), default_autoencoder_path)

    metadata = {
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "epochs": epochs,
        "num_chunks": len(X_train),
    }
    with open(default_metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dir", default=default_data_clean)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--encoding-dim", type=int, default=16)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--from-db", action="store_true")
    args = parser.parse_args()

    train(
        clean_dir=args.clean_dir,
        epochs=args.epochs,
        lr=args.lr,
        encoding_dim=args.encoding_dim,
        reset=args.reset,
        from_db=args.from_db,
    )


if __name__ == "__main__":
    main()
