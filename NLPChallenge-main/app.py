import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.chat_handler import ChatHandler
from utils.learning_agents import LearningAgent
from utils.llm_config import get_available_models
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
    learning_agent = LearningAgent(chat_handler)
    return doc_processor, vector_store, chat_handler, learning_agent

def main():
    st.set_page_config(
        page_title="Document Chatbot",
        page_icon="",
        layout="wide"
    )

    # Initialize processors
    doc_processor, vector_store, chat_handler, learning_agent = init_processors()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload", "Chat", "Study Tools"])

    # LLM Selection in sidebar
    st.sidebar.title("Model Settings")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox(
        "Select Language Model",
        available_models,
        index=0,
        help="Choose which language model to use for answering questions"
    )
    
    # Update the chat handler with selected model
    chat_handler.set_model(selected_model)

    if page == "Home":
        st.title("Welcome to Document Chatbot! 📚")
        st.write("""
        This application allows you to:
        * Upload PDF documents
        * Process and analyze document content
        * Chat with an AI about your documents
        * Generate study materials (cheatsheets and quizzes)
        * Get instant answers to your questions
        
        Start by uploading your documents in the Upload section!
        """)

    elif page == "Upload":
        st.title("Upload Documents 📄")
        uploaded_files = st.file_uploader(
            "Choose your PDF documents",
            accept_multiple_files=True,
            type=['pdf']
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Process documents
                    processed_docs = doc_processor.process_documents(uploaded_files)
                    
                    # Store in vector store
                    for doc in processed_docs:
                        vector_store.add_texts(doc)
                    
                    st.success(f"Successfully processed {len(processed_docs)} documents!")

    elif page == "Chat":
        st.title("Chat with your Documents 💬")
        
        # Query input
        query = st.text_input("Ask a question about your documents")
        
        if query:
            with st.spinner("Searching for answer..."):
                # Get relevant documents
                context = vector_store.similarity_search(query)
                
                # Generate response
                response = chat_handler.get_response(query, context)
                
                st.write("Answer:")
                st.write(response)

    elif page == "Study Tools":
        st.title("Study Tools 📚")
        
        tool_type = st.radio(
            "Select Tool",
            ["CheatSheet Generator", "Quiz Generator"]
        )
        
        if tool_type == "CheatSheet Generator":
            st.header("Generate Document Summary")
            if st.button("Generate CheatSheet"):
                with st.spinner("Generating summary..."):
                    # Get all documents from vector store
                    all_docs = vector_store.get_all_documents()
                    if all_docs:
                        summary = learning_agent.generate_cheatsheet({"documents": [all_docs]})
                        st.markdown(summary)
                    else:
                        st.warning("Please upload some documents first!")
                        
        else:  # Quiz Generator
            st.header("Generate Quiz")
            col1, col2 = st.columns(2)
            with col1:
                num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=5)
            with col2:
                question_type = st.selectbox("Question Type", ["multiple_choice", "open_ended"])
                
            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    # Get all documents from vector store
                    all_docs = vector_store.get_all_documents()
                    if all_docs:
                        questions = learning_agent.generate_quiz(
                            {"documents": [all_docs]},
                            num_questions=num_questions,
                            question_type=question_type
                        )
                        formatted_quiz = learning_agent.format_quiz_for_display(questions, question_type)
                        st.markdown(formatted_quiz)
                    else:
                        st.warning("Please upload some documents first!")

if __name__ == "__main__":
    main()
