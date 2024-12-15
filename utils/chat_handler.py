from groq import Groq
from openai import OpenAI

import os
from dotenv import load_dotenv
from typing import Dict, List
from llama_index.prompts import PromptTemplate
from .llm_config import get_model_config
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import httpx

# Load environment variables
load_dotenv()

class ChatHandler:
    def __init__(self, model_name: str = "Groq - Mixtral 8x7B"):
        self.model_config = get_model_config(model_name)
        if not self.model_config:
            raise ValueError(f"Invalid model name: {model_name}")
            
        # Initialize the appropriate client based on provider
        self._init_client()
        
        # Initialize chat history
        self.chat_history = []
        
        # Custom prompt for better source attribution
        self.qa_prompt = """You are a helpful AI assistant that provides clear and accurate answers based on the provided context.
        Always cite your sources using the format [Source: filename] at the end of relevant statements.

        Context: {context}
        
        Previous conversation:
        {chat_history}

        Question: {query}

        Answer: Let me help you with that.
        """

    def _init_client(self):
        """Initialize the appropriate client based on provider"""
        
        if self.model_config.provider == "Groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
            http_client = httpx.Client(
                base_url="https://api.groq.com/openai/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60.0
            )
            self.client = Groq(http_client=http_client)  # Let it use environment variables
       
        else:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")

    def get_response(self, query: str, context_docs: Dict) -> str:
        """Get a response from the model based on the query and context"""
        context = context_docs.get('documents', [[]])[0]
        
        # Format chat history
        chat_history_text = ""
        if self.chat_history:
            chat_history_text = "\n".join([
                f"User: {exchange['user']}\nAssistant: {exchange['assistant']}"
                for exchange in self.chat_history
            ])
        
        # Format the prompt with context and chat history
        prompt = self.qa_prompt.format(
            context=context, 
            chat_history=chat_history_text,
            query=query
        )

        try:
            response_text = ""          
            if self.model_config.provider == "Groq":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides clear and accurate answers based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                response = self.client.chat.completions.create(
                    model=self.model_config.model_id,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                    messages=messages
                )
                response_text = response.choices[0].message.content
            
            
            else:
                raise ValueError(f"Unsupported provider: {self.model_config.provider}")

            # Add the exchange to chat history
            self.chat_history.append({
                "user": query,
                "assistant": response_text
            })
            
            return response_text

        except Exception as e:
            print(f"Error getting response: {str(e)}")
            return f"Error: Unable to get response from {self.model_config.provider} model."

    def set_model(self, model_name: str):
        """Change the model being used"""
        self.model_config = get_model_config(model_name)
        if not self.model_config:
            raise ValueError(f"Invalid model name: {model_name}")
        self._init_client()

    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = []
