import os
from groq import Groq
from typing import List, Dict
import json

class ChatHandler:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv('GROQ_API_KEY')
        )
        self.model = "mixtral-8x7b-32768"  # Using Mixtral model for better performance
        
    def generate_prompt(self, query: str, context_docs: Dict) -> str:
        """
        Generate a prompt for the LLM using the query and retrieved documents
        """
        # ChromaDB returns a dict with 'documents' as a list of documents
        documents = context_docs.get('documents', [[]])[0]  # Get first batch of documents
        context = "\n\n".join(documents)
        
        return f"""You are a helpful assistant that answers questions based on the provided context. 
        Use the following pieces of context to answer the question. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {query}

        Answer:"""

    def get_response(self, query: str, context_docs: Dict) -> str:
        """
        Generate a response using the LLM based on the query and context documents
        """
        prompt = self.generate_prompt(query, context_docs)
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
        )
        
        return completion.choices[0].message.content
