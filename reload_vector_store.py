#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

DOCS_DIR = "exports"
PERSIST_DIR = "vector-store"
COLLECTION_NAME = "collection_test"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def load_docs(folder):
    paths = [p for p in Path(folder).rglob("*") if p.is_file()]
    docs = []
    errors = 0
    for p in tqdm(paths, desc="Chargement des fichiers"):
        try:
            if p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
            elif p.suffix.lower() == ".docx":
                docs.extend(Docx2txtLoader(str(p)).load())
            elif p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
        except Exception as e:
            print(f"  Erreur sur {p.name}: {e}")
            errors += 1
    return docs, errors


def main():
    load_dotenv()

    emb_model = os.getenv("MODEL_EMBEDDING")
    if not emb_model:
        print("Variable MODEL_EMBEDDING manquante dans .env", file=sys.stderr)
        return 1

    embeddings = OpenAIEmbeddings(model=emb_model)

    # Supprimer l'ancienne collection
    print(f"Suppression de la collection '{COLLECTION_NAME}'...")
    old_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    old_store.reset_collection()

    # Charger les documents
    print(f"\nChargement des documents depuis '{DOCS_DIR}/'...")
    raw_docs, errors = load_docs(DOCS_DIR)
    print(f"  {len(raw_docs)} documents chargés, {errors} erreur(s)")

    if not raw_docs:
        print("Aucun document trouvé, arrêt.", file=sys.stderr)
        return 1

    # Découper en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  {len(chunks)} chunks créés")

    # Indexer dans ChromaDB
    print("\nIndexation dans ChromaDB...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vector_store.add_documents(documents=chunks)

    print(f"\nTerminé. {len(chunks)} chunks indexés dans '{PERSIST_DIR}/'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
