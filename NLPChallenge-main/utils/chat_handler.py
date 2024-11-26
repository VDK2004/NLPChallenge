import os
from groq import Groq
from typing import List, Dict
import json
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from .llm_config import get_model_config, LLMConfig

class ChatHandler:
    def __init__(self):
        self.groq_client = None
        self.hf_client = None
        self.current_model = None
        self.hf_tokenizer = None
        self.hf_model = None
        self.setup_groq()
        
    def setup_groq(self):
        """Initialize Groq client"""
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            self.groq_client = Groq(api_key=api_key)
            
    def setup_huggingface(self, model_id: str):
        """Initialize HuggingFace model and tokenizer"""
        try:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                raise ValueError("HuggingFace API key not found in environment variables")
                
            # Load tokenizer and model
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load appropriate model based on architecture
            if "t5" in model_id.lower():
                self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            else:
                self.hf_model = AutoModelForCausalLM.from_pretrained(model_id)
                
            if torch.cuda.is_available():
                self.hf_model = self.hf_model.to('cuda')
                
        except Exception as e:
            raise Exception(f"Error setting up HuggingFace model: {str(e)}")
            
    def set_model(self, model_name: str):
        """Set the current model configuration"""
        self.current_model = get_model_config(model_name)
        
        # Initialize appropriate client if needed
        if self.current_model.provider == "huggingface":
            self.setup_huggingface(self.current_model.model_id)
        elif self.current_model.provider == "groq" and not self.groq_client:
            self.setup_groq()
        
    def generate_prompt(self, query: str, context_docs: Dict) -> str:
        """Generate a prompt for the LLM using the query and retrieved documents"""
        documents = context_docs.get('documents', [[]])[0]
        context = "\n\n".join(documents)
        
        if "t5" in self.current_model.model_id.lower():
            return f"answer question: {query} based on context: {context}"
        
        return f"""You are a helpful assistant that answers questions based on the provided context. 
        Use the following pieces of context to answer the question. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {query}

        Answer:"""

    def get_huggingface_response(self, prompt: str) -> str:
        """Generate response using local HuggingFace model"""
        try:
            inputs = self.hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.current_model.max_tokens)
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_length=self.current_model.max_tokens,
                    num_return_sequences=1,
                    temperature=self.current_model.temperature
                )
                
            response = self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            raise Exception(f"Error generating HuggingFace response: {str(e)}")

    def get_response(self, query: str, context_docs: Dict) -> str:
        """Generate a response using the selected LLM based on the query and context documents"""
        if not self.current_model:
            raise ValueError("No LLM model selected. Please select a model first.")
            
        prompt = self.generate_prompt(query, context_docs)
        
        try:
            if self.current_model.provider == "groq":
                if not self.groq_client:
                    raise ValueError("Groq API key not configured")
                    
                completion = self.groq_client.chat.completions.create(
                    model=self.current_model.model_id,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.current_model.temperature,
                    max_tokens=self.current_model.max_tokens,
                )
                return completion.choices[0].message.content
                
            elif self.current_model.provider == "huggingface":
                return self.get_huggingface_response(prompt)
                
            else:
                raise ValueError(f"Unsupported LLM provider: {self.current_model.provider}")
                
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
