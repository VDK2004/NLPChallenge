import PyPDF2
from typing import List, Dict
import uuid
import os

class DocumentProcessor:
    def __init__(self):
        self.processed_docs = []

    def extract_text_from_pdf(self, pdf_file) -> Dict:
        """
        Extract text from a PDF file and return a dictionary with metadata
        """
        try:
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())

            # Create document metadata
            doc_info = {
                'id': str(uuid.uuid4()),
                'filename': pdf_file.name,
                'num_pages': len(pdf_reader.pages),
                'content': '\n'.join(text_content)
            }
            
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
                processed_docs.append(doc_info)
            
        return processed_docs
