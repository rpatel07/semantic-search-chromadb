import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# initialize the embedding function
ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="semantic_search",
    embedding_function=ef,
    configuration={"hnsw": {"space": "cosine"}},
)

data_path = Path(__file__).parent / "documents.json"

with open(data_path, "r", encoding="utf-8") as file:
    data = json.load(file)

records = data["documents"]

ids = [record["id"] for record in records]
documents = [record["document"] for record in records]
metadatas = [record["metadata"] for record in records]

# Store the data
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)

# Insert or add the documents
def update_insert(id, document, metadata):
    return collection.upsert(
        ids=[id],
        documents=[document],
        metadatas=[metadata]
    )

# Get one document
def get_document(doc_id):
    return collection.get(
        ids=[doc_id],
        include=["documents", "metadatas"]
    )

# Ask the question
def ask_question(query, n_result=3):
    print(collection.query(
        query_texts=[query],
        n_results=n_result
    ))

# Delete the document
def delete_document(ids):
    collection.delete(ids=[ids])

# Test the function
# print(get_document("doc_003"))
# ask_question("How many provinces does canada have?", 2)
# update_insert("doc_011", "I love playing football", {"game": "football"})
# print(collection.count())
# delete_document("doc_011")
