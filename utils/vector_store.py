import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChromaVectorStore:
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


class PineconeVectorStore:
    def __init__(self, index_name: str = "documents", external_indexes: List[str] = None):
        self.index_name = index_name
        self.external_indexes = external_indexes or []
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # dimension of all-MiniLM-L6-v2 embeddings
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Free tier region
                )
            )
        
        # Get the main index and external indexes
        self.index = self.pc.Index(self.index_name)
        self.external_index_clients = {
            idx: self.pc.Index(idx) for idx in self.external_indexes
        }

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        """
        return self.embedding_model.encode(texts).tolist()

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store
        """
        vectors_to_upsert = []
        
        for doc in documents:
            # Generate embedding for the document
            embedding = self.generate_embeddings([doc['content']])[0]
            
            # Prepare metadata
            metadata = {
                'content': doc['content'],
                'filename': doc['filename'],
                'num_pages': doc['num_pages'],
                'source_index': self.index_name
            }
            
            # Add additional metadata if present
            for key in ['source_type', 'url', 'video_id']:
                if key in doc:
                    metadata[key] = doc[key]
            
            # Create vector tuple (id, embedding, metadata)
            vectors_to_upsert.append((
                doc['id'],
                embedding,
                metadata
            ))
        
        # Upsert to Pinecone in batches
        self.index.upsert(vectors=vectors_to_upsert)
        
        return len(documents)

    def search_similar(self, query: str, n_results: int = 5, include_external: bool = True):
        """
        Search for similar documents across all indexes
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]
        
        # Query main index
        main_results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        all_matches = list(main_results.matches)
        
        # Query external indexes if enabled
        if include_external and self.external_index_clients:
            for idx_name, idx_client in self.external_index_clients.items():
                try:
                    ext_results = idx_client.query(
                        vector=query_embedding,
                        top_k=n_results,
                        include_metadata=True
                    )
                    all_matches.extend(ext_results.matches)
                except Exception as e:
                    print(f"Error querying external index {idx_name}: {str(e)}")
        
        # Sort all matches by score and take top n_results
        all_matches.sort(key=lambda x: x.score, reverse=True)
        all_matches = all_matches[:n_results]
        
        # Format results to match the expected structure
        formatted_results = {
            'ids': [match.id for match in all_matches],
            'distances': [match.score for match in all_matches],
            'documents': [match.metadata['content'] for match in all_matches],
            'metadatas': [{
                'filename': match.metadata['filename'],
                'num_pages': match.metadata['num_pages'],
                'source_index': match.metadata.get('source_index', 'unknown'),
                'source_type': match.metadata.get('source_type', 'document'),
                'url': match.metadata.get('url', ''),
            } for match in all_matches]
        }
        
        return formatted_results

    def count(self, include_external: bool = True):
        """
        Get the number of documents in all indexes
        """
        total = self.index.describe_index_stats().total_vector_count
        
        if include_external and self.external_index_clients:
            for idx_client in self.external_index_clients.values():
                try:
                    total += idx_client.describe_index_stats().total_vector_count
                except Exception:
                    pass
        
        return total

# Set VectorStore to the implementation we want to use
VectorStore = PineconeVectorStore
