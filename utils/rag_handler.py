import os
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGHandler:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"  # Using Mixtral for better context handling
        
    def generate_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate a prompt for the LLM using the query and retrieved documents
        """
        context_str = "\n\n".join([
            f"Document: {doc['metadata']['filename']}\nContent: {doc['document']}"
            for doc in context_docs
        ])
        
        return f"""You are a helpful assistant that answers questions based on the provided documents. 
Your task is to answer the question using ONLY the information from the provided documents.
If the answer cannot be found in the documents, say so clearly.

Context Documents:
{context_str}

Question: {query}

Answer: Let me help you with that based on the provided documents."""

    async def get_answer(self, query: str, n_results: int = 3) -> str:
        """
        Get an answer for the query using RAG:
        1. Retrieve relevant documents
        2. Generate a prompt with context
        3. Get response from LLM
        """
        try:
            # Retrieve relevant documents
            results = self.vector_store.search_similar(query, n_results)
            
            if not results['documents']:
                return "I couldn't find any relevant information in the uploaded documents to answer your question."
            
            # Prepare documents with metadata
            context_docs = [
                {
                    'document': doc,
                    'metadata': meta
                }
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]
            
            # Generate prompt with context
            prompt = self.generate_prompt(query, context_docs)
            
            # Get response from LLM
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides accurate answers based on the given context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.3,
                max_tokens=1000,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"
