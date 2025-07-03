"""
chunk_and_embed.py

Task 2: Text Chunking, Embedding, and Vector Store Indexing
- Chunks cleaned complaint narratives
- Embeds each chunk using sentence-transformers/all-MiniLM-L6-v2
- Stores embeddings and metadata in a FAISS or ChromaDB vector store
- Persists the vector store in vector_store/
"""
import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

# Parameters
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 50  # characters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "../vector_store"

# Load cleaned data
filtered = pd.read_csv("data/filtered_complaints.csv")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[". ", "\n", " "]
)

chunks = []
metadata = []
for idx, row in tqdm(filtered.iterrows(), total=len(filtered)):
    text = str(row["cleaned_narrative"])
    chunked = splitter.split_text(text)
    for chunk in chunked:
        chunks.append(chunk)
        metadata.append({
            "complaint_id": row["Complaint ID"] if "Complaint ID" in row else idx,
            "product": row["Product"]
        })

# Embedding
model = SentenceTransformer(f"sentence-transformers/{EMBEDDING_MODEL}")
embeddings = model.encode(chunks, show_progress_bar=True)

# Vector store (ChromaDB)
client = chromadb.Client(Settings(
    persist_directory=VECTOR_STORE_DIR
))
collection = client.get_or_create_collection("complaints")

for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
    collection.add(
        embeddings=[emb.tolist()],
        documents=[chunks[i]],
        metadatas=[meta],
        ids=[str(i)]
    )

client.persist()
print(f"Vector store saved to {VECTOR_STORE_DIR}")
