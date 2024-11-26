import PyPDF2
from typing import List, Dict
import uuid
import os
import tiktoken

class DocumentProcessor:
    def __init__(self):
        self.processed_docs = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's GPT-4 tokenizer
        self.max_tokens = 500  # Maximum tokens per chunk

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize and chunk text using tiktoken
        """
        # Encode the text into tokens
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Split into chunks of max_tokens
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """
        Extract text from a PDF file and return a list of document chunks
        """
        try:
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())

            # Join all text and tokenize into chunks
            full_text = '\n'.join(text_content)
            text_chunks = self.tokenize_text(full_text)

            # Create document metadata
            doc_info = []
            for i, chunk in enumerate(text_chunks):
                doc_info.append({
                    'id': f"{str(uuid.uuid4())}_{i}",
                    'filename': pdf_file.name,
                    'num_pages': len(pdf_reader.pages),
                    'content': chunk
                })
            
            return doc_info
        
        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_file.name}: {str(e)}")

    def process_documents(self, files: List) -> List[Dict]:
        """
        Process multiple documents and return their content with metadata
        """
        processed_docs = []
        
        for file in files:
            if file.name.lower().endswith('.pdf'):
                doc_info = self.extract_text_from_pdf(file)
                processed_docs.extend(doc_info)
            
        return processed_docs
