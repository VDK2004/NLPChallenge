import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.chat_handler import ChatHandler
from utils.llm_config import get_available_models, get_model_config
from utils.summary_handler import SummaryHandler
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize processor and vector store
@st.cache_resource
def init_processors():
    doc_processor = DocumentProcessor()
    # Initialize vector store with external indexes
    external_indexes = os.getenv('EXTERNAL_INDEXES', '').split(',')
    external_indexes = [idx.strip() for idx in external_indexes if idx.strip()]
    vector_store = VectorStore(external_indexes=external_indexes)
    chat_handler = ChatHandler()
    return doc_processor, vector_store, chat_handler

def display_quiz(quiz_data: dict):
    """Display an interactive quiz in Streamlit"""
    # Initialize quiz data in session state if not exists
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = quiz_data
        st.session_state.quiz_answers = {}
        st.session_state.show_explanations = False
    
    # Create a container for the quiz
    quiz_container = st.container()
    
    with quiz_container:
        quiz_data = st.session_state.current_quiz
        
        if "error" in quiz_data:
            st.error(f"Error generating quiz: {quiz_data.get('error')}")
            return
            
        metadata = quiz_data.get("quiz_metadata", {})
        st.subheader("📝 Quiz")
        
        # Display quiz metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Topic: {metadata.get('topic', 'N/A')}")
        with col2:
            st.info(f"Difficulty: {metadata.get('difficulty', 'N/A')}")
        with col3:
            st.info(f"Estimated Time: {metadata.get('estimated_time', 'N/A')}")
        
        st.write(f"Total Points: {metadata.get('total_points', 0)}")
        st.write("---")
        
        # Display questions
        questions = quiz_data.get("questions", [])
        for q in questions:
            q_id = q.get("id")
            q_type = q.get("type")
            
            st.markdown(f"**Question {q_id}** ({q.get('points', 1)} points)")
            st.write(q.get("question"))
            
            # Get current answer from session state
            current_answer = st.session_state.quiz_answers.get(q_id, "")
            
            # Different input types based on question type
            if q_type == "multiple_choice":
                options = q.get("options", [])
                selected = st.radio(
                    label=f"Select answer for question {q_id}:",
                    options=options,
                    key=f"q_{q_id}",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[q_id] = selected
                
            elif q_type == "true_false":
                selected = st.radio(
                    label=f"Select answer for question {q_id}:",
                    options=["True", "False"],
                    key=f"q_{q_id}",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[q_id] = selected
                
            elif q_type == "open_ended":
                text_input = st.text_area(
                    label=f"Your answer for question {q_id}:",
                    value=current_answer,
                    key=f"q_{q_id}",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[q_id] = text_input
            
            st.write("---")
        
        # Submit button and show explanations
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Quiz", key="submit_quiz"):
                st.session_state.show_explanations = True
        with col2:
            if st.button("Reset Quiz", key="reset_quiz"):
                st.session_state.quiz_answers = {}
                st.session_state.show_explanations = False
                st.experimental_rerun()
        
        # Show explanations and correct answers
        if st.session_state.show_explanations:
            st.write("### Quiz Results")
            correct_count = 0
            total_points = 0
            earned_points = 0
            
            for q in questions:
                q_id = q.get("id")
                points = q.get("points", 1)
                total_points += points
                
                user_answer = st.session_state.quiz_answers.get(q_id, "")
                correct_answer = q.get("correct_answer")
                
                if user_answer.lower() == correct_answer.lower():
                    correct_count += 1
                    earned_points += points
                    st.success(f"Question {q_id}: Correct! (+{points} points)")
                else:
                    st.error(f"Question {q_id}: Incorrect")
                    st.write(f"Your answer: {user_answer}")
                    st.write(f"Correct answer: {correct_answer}")
                
                st.info(f"Explanation: {q.get('explanation', 'No explanation provided.')}")
                st.write("---")
            
            # Display final score
            score_percentage = (earned_points / total_points) * 100 if total_points > 0 else 0
            st.write(f"### Final Score: {earned_points}/{total_points} points ({score_percentage:.1f}%)")
            st.write(f"Correct Answers: {correct_count}/{len(questions)}")

def main():
    st.set_page_config(
        page_title="Document Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize processors
    doc_processor, vector_store, chat_handler = init_processors()
    summary_handler = SummaryHandler(chat_handler)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload", "Chat", "Study Materials"])
    
    # Model selection in sidebar
    st.sidebar.title("Model Settings")
    available_models = get_available_models()
    current_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0,
        help="Choose which AI model to use for chat responses"
    )
    
    # Show model info
    if current_model:
        model_config = get_model_config(current_model)
        with st.sidebar.expander("Model Information"):
            st.write(f"**Provider:** {model_config.provider}")
            st.write(f"**Context Window:** {model_config.context_window:,} tokens")
            st.write(f"**Description:** {model_config.description}")
    
    # Update chat handler model if changed
    if "current_model" not in st.session_state or st.session_state.current_model != current_model:
        st.session_state.current_model = current_model
        chat_handler.set_model(current_model)

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
        
        # File upload section
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose your documents",
            accept_multiple_files=True,
            type=['pdf', 'pptx', 'ppt']
        )
        
        # URL input section
        st.subheader("Add URLs")
        url_input = st.text_area(
            "Enter URLs (one per line) - Supports YouTube videos and webpages",
            height=100,
            help="Enter URLs to YouTube videos or webpages, one per line"
        )
        
        # Process button
        if st.button("Process Documents"):
            with st.spinner('Processing documents...'):
                try:
                    # Prepare list of all sources
                    all_sources = []
                    
                    # Add uploaded files
                    if uploaded_files:
                        all_sources.extend(uploaded_files)
                    
                    # Add URLs
                    if url_input:
                        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                        all_sources.extend(urls)
                    
                    if all_sources:
                        # Process documents
                        processed_docs = doc_processor.process_documents(all_sources)
                        
                        if processed_docs:
                            # Add to vector store
                            num_docs = vector_store.add_documents(processed_docs)
                            st.success(f"Successfully processed and stored {num_docs} documents!")
                            
                            # Display document info
                            st.subheader("Processed Documents:")
                            for doc in processed_docs:
                                source_type = doc.get('source_type', 'document')
                                if source_type == 'youtube':
                                    st.write(f"📺 YouTube: {doc['filename']} ({doc['url']})")
                                elif source_type == 'webpage':
                                    st.write(f"🌐 Webpage: {doc['url']}")
                                elif source_type == 'powerpoint':
                                    st.write(f"📊 PowerPoint: {doc['filename']} ({doc['num_pages']} slides)")
                                else:
                                    st.write(f"📄 {doc['filename']} ({doc['num_pages']} pages)")
                        else:
                            st.warning("No valid documents were found to process.")
                    else:
                        st.warning("Please upload files or enter URLs to process.")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

    elif page == "Chat":
        st.title("Chat with your Documents 💬")
        
        # Check if there are documents in the vector store
        if vector_store.count() == 0:
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

    elif page == "Study Materials":
        st.title("Study Materials Generator 📚")
        
        if vector_store.count() == 0:
            st.warning("Please upload some documents first in the Upload section!")
            return
            
        # Topic input
        topic = st.text_input("Enter the topic or subject:", 
                            help="Specify the main topic or subject for the study materials")
        
        # Get document content
        if topic:
            with st.spinner("Retrieving relevant content..."):
                search_results = vector_store.search_similar(topic, n_results=3)
                content = search_results.get('documents', [[]])[0][0]
                
                # Study material options
                study_material_type = st.radio(
                    "Choose study material type:",
                    ["Summary", "Quiz"]
                )
                
                # Show specific options based on selection
                if study_material_type == "Summary":
                    summary_type = st.selectbox(
                        "Select summary type:",
                        ["comprehensive", "brief", "technical"]
                    )
                elif study_material_type == "Quiz":
                    col1, col2 = st.columns(2)
                    with col1:
                        num_questions = st.number_input("Number of questions:", min_value=3, max_value=20, value=5)
                    with col2:
                        difficulty = st.selectbox("Difficulty level:", ["easy", "medium", "hard"])
                
                if st.button("Generate Study Materials"):
                    with st.spinner("Generating study materials..."):
                        try:
                            if study_material_type == "Summary":
                                result = summary_handler.create_summary(topic, content, summary_type)
                            else:  # Quiz
                                if 'current_quiz' not in st.session_state:
                                    result = summary_handler.generate_quiz(topic, content, num_questions, difficulty)
                                    st.session_state.current_quiz = result
                                    st.session_state.quiz_answers = {}
                                    st.session_state.show_explanations = False
                                display_quiz(result)
                            
                            if "error" in result:
                                st.error(f"Error generating study materials: {result['error']}")
                                if "raw_response" in result:
                                    with st.expander("Show raw response"):
                                        st.text(result["raw_response"])
                            else:
                                # Display the structured content
                                if study_material_type == "Summary":
                                    # Metadata section
                                    if "metadata" in result:
                                        st.markdown("### 📊 Summary Information")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.info(f"📚 Reading Time: {result['metadata'].get('estimated_reading_time', 'N/A')}")
                                        with col2:
                                            st.info(f"📈 Complexity: {result['metadata'].get('complexity_level', 'N/A')}")
                                        with col3:
                                            if prereqs := result['metadata'].get('prerequisites'):
                                                st.info("🎯 Prerequisites: " + ", ".join(prereqs))
                                    
                                    # Executive Summary
                                    if "executive_summary" in result:
                                        st.markdown("### 📝 Executive Summary")
                                        st.write(result["executive_summary"])
                                    
                                    # Key Concepts
                                    if "key_concepts" in result:
                                        st.markdown("### 🔑 Key Concepts")
                                        for concept in result["key_concepts"]:
                                            st.markdown(f"• {concept}")
                                    
                                    # Detailed Analysis
                                    if "detailed_analysis" in result:
                                        st.markdown("### 🔍 Detailed Analysis")
                                        
                                        if "main_arguments" in result["detailed_analysis"]:
                                            st.markdown("**Main Arguments:**")
                                            for arg in result["detailed_analysis"]["main_arguments"]:
                                                st.markdown(f"• {arg}")
                                        
                                        if "supporting_evidence" in result["detailed_analysis"]:
                                            st.markdown("**Supporting Evidence:**")
                                            for evidence in result["detailed_analysis"]["supporting_evidence"]:
                                                st.markdown(f"• {evidence}")
                                        
                                        if "relationships" in result["detailed_analysis"]:
                                            st.markdown("**Key Relationships:**")
                                            for rel in result["detailed_analysis"]["relationships"]:
                                                st.markdown(f"• {rel}")
                                    
                                    # Technical Details
                                    if "technical_details" in result:
                                        st.markdown("### ⚙️ Technical Details")
                                        for detail in result["technical_details"]:
                                            st.markdown(f"• {detail}")
                                    
                                    # Practical Applications
                                    if "practical_applications" in result:
                                        st.markdown("### 🛠️ Practical Applications")
                                        for app in result["practical_applications"]:
                                            st.markdown(f"• {app}")
                                    
                                    # Key Terms
                                    if "key_terms" in result:
                                        st.markdown("### 📚 Key Terms")
                                        for term, definition in result["key_terms"].items():
                                            with st.expander(f"📌 {term}"):
                                                st.write(definition)
                                
                                else:  # Quiz
                                    display_quiz(result)
                                
                                # Add a button to view the raw JSON
                                with st.expander("View Raw JSON"):
                                    st.json(result)
                                
                        except Exception as e:
                            st.error(f"Error generating study materials: {str(e)}")

if __name__ == "__main__":
    main()
