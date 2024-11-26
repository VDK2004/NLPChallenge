import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        return self.embedding_model.encode(texts).tolist()

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store
        """
        # Prepare data for ChromaDB
        ids = [doc['id'] for doc in documents]
        texts = [doc['content'] for doc in documents]
        metadatas = [
            {
                'filename': doc['filename'],
                'num_pages': doc['num_pages']
            } for doc in documents
        ]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return len(documents)

    def search_similar(self, query: str, n_results: int = 5):
        """
        Search for similar documents
        """
        query_embedding = self.generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results
