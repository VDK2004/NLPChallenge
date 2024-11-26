import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.chat_handler import ChatHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize processor and vector store
@st.cache_resource
def init_processors():
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    chat_handler = ChatHandler()
    return doc_processor, vector_store, chat_handler

def main():
    st.set_page_config(
        page_title="Document Chatbot",
        page_icon="",
        layout="wide"
    )

    # Initialize processors
    doc_processor, vector_store, chat_handler = init_processors()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload", "Chat"])

    if page == "Home":
        st.title("Welcome to Document Chatbot! ")
        st.write("""
        This application allows you to:
        * Upload PDF documents
        * Process and analyze document content
        * Chat with an AI about your documents
        * Get instant answers to your questions
        
        Start by uploading your documents in the Upload section!
        """)

    elif page == "Upload":
        st.title("Upload Documents ")
        uploaded_files = st.file_uploader(
            "Choose your PDF documents",
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if uploaded_files:
            with st.spinner('Processing documents...'):
                try:
                    # Process documents
                    processed_docs = doc_processor.process_documents(uploaded_files)
                    
                    if processed_docs:
                        # Add to vector store
                        num_docs = vector_store.add_documents(processed_docs)
                        st.success(f"Successfully processed and stored {num_docs} documents!")
                        
                        # Display document info
                        st.subheader("Processed Documents:")
                        for doc in processed_docs:
                            st.write(f" {doc['filename']} ({doc['num_pages']} pages)")
                    else:
                        st.warning("No valid PDF documents were uploaded.")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

    elif page == "Chat":
        st.title("Chat with your Documents 💬")
        
        # Check if there are documents in the vector store
        if vector_store.collection.count() == 0:
            st.warning("Please upload some documents first in the Upload section!")
            return
            
        # Initialize chat history in session state if it doesn't exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about your documents:")
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Search for relevant documents
                        search_results = vector_store.search_similar(user_input, n_results=3)
                        
                        # Generate response using RAG
                        response = chat_handler.get_response(user_input, search_results)
                        
                        # Display response
                        st.write(response)
                        
                        # Add assistant message to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.chat_history.append({"role": "assistant", "content": "I apologize, but I encountered an error while processing your request. Please try again."})

if __name__ == "__main__":
    main()
