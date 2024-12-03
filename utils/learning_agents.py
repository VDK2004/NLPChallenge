from .vector_store import VectorStore
from .web_search import WebSearcher
from typing import List, Dict, Optional
import json

class LearningAgent:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        self.vector_store = VectorStore()
        self.web_searcher = WebSearcher()

    def _get_relevant_information(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Get relevant information from both vector store and web search
        """
        # Search vector store first
        vector_results = self.vector_store.search_similar(query, n_results)
        
        # If not enough relevant results, search the web
        if len(vector_results.matches) < n_results:
            web_results = self.web_searcher.search(
                query,
                num_results=n_results - len(vector_results.matches)
            )
            
            # Combine results with source information
            all_results = []
            
            # Add vector store results
            for match in vector_results.matches:
                all_results.append({
                    "content": match.metadata["content"],
                    "source": f"Document: {match.metadata.get('filename', 'Unknown')}",
                    "score": match.score
                })
            
            # Add web results
            for result in web_results:
                all_results.append({
                    "content": result["content"],
                    "source": f"Web: {result['url']}",
                    "score": 0.5  # Default score for web results
                })
            
            return all_results
        
        return [{
            "content": match.metadata["content"],
            "source": f"Document: {match.metadata.get('filename', 'Unknown')}",
            "score": match.score
        } for match in vector_results.matches]

    def generate_cheatsheet(self, topic: str, content: str) -> Dict:
        """
        Generate a comprehensive cheatsheet for a given topic
        """
        try:
            # Get relevant information
            relevant_info = self._get_relevant_information(topic)
            
            # Prepare context with relevant information
            context = "\n\n".join([
                f"Source: {info['source']}\n{info['content']}"
                for info in relevant_info
            ])
            
            # Generate cheatsheet with references
            system_prompt = """You are an expert educator tasked with creating comprehensive cheatsheets.
            Use the provided context and your knowledge to create a detailed cheatsheet.
            Include references to sources where information was obtained.
            Format the output as a JSON with the following structure:
            {
                "title": "Topic title",
                "sections": [
                    {
                        "heading": "Section heading",
                        "content": "Section content",
                        "references": ["Source references"]
                    }
                ]
            }"""
            
            user_prompt = f"""Create a comprehensive cheatsheet for the topic: {topic}
            
            Use this content as primary source:
            {content}
            
            Additional context from other sources:
            {context}
            
            Make sure to:
            1. Include key concepts, definitions, and examples
            2. Use clear and concise language
            3. Organize information logically
            4. Reference sources for each section
            5. Keep information accurate and relevant"""
            
            response = self.llm_handler.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Parse and validate response
            try:
                cheatsheet = json.loads(response)
                return cheatsheet
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse cheatsheet response",
                    "raw_response": response
                }
                
        except Exception as e:
            return {
                "error": f"Error generating cheatsheet: {str(e)}"
            }

    def generate_quiz(self, topic: str, content: str, num_questions: int = 5, question_type: str = "multiple_choice") -> List[Dict]:
        """Generate quiz questions based on the document content"""
        if question_type == "multiple_choice":
            prompt = f"""As an expert educator, create {num_questions} high-quality multiple-choice questions based on the following content.

            Guidelines:
            1. Questions should test understanding, not just memorization
            2. Include a mix of difficulty levels
            3. Make all options plausible
            4. Avoid obvious incorrect answers
            5. Include questions that test different cognitive levels (recall, application, analysis)
            
            Format each question as a JSON object:
            {{
                "question": "[Question text]",
                "options": {{
                    "A": "[Option A]",
                    "B": "[Option B]",
                    "C": "[Option C]",
                    "D": "[Option D]"
                }},
                "correct_answer": "[A/B/C/D]",
                "explanation": "[Brief explanation of why this is the correct answer]"
            }}
            
            Content for quiz:
            {content}
            
            Return the questions as a JSON array:"""
        else:
            prompt = f"""As an expert educator, create {num_questions} thought-provoking open-ended questions based on the following content.
            
            Guidelines:
            1. Questions should promote critical thinking and deep understanding
            2. Include a mix of question types (explain, compare, analyze, evaluate)
            3. Each question should have a detailed model answer
            4. Questions should test different cognitive levels
            
            Format each question as a JSON object:
            {{
                "question": "[Question text]",
                "model_answer": "[Detailed answer]",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "difficulty": "[Easy/Medium/Hard]"
            }}
            
            Content for quiz:
            {content}
            
            Return the questions as a JSON array:"""

        response = self.llm_handler.get_response("", {"documents": [[prompt]]})
        try:
            # Parse the response as JSON and return
            return json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return [{"error": "Failed to parse response as JSON", "raw_response": response}]

    def format_quiz_for_display(self, questions: List[Dict], question_type: str = "multiple_choice") -> str:
        """Format quiz questions for display in Streamlit"""
        formatted_output = []
        
        for i, q in enumerate(questions, 1):
            formatted_output.append(f"### Question {i}")
            formatted_output.append(q['question'])
            
            if question_type == "multiple_choice":
                formatted_output.extend(q['options'].values())
                formatted_output.append(f"\n**Correct Answer:** {q['correct_answer']}")
            else:
                formatted_output.append("\n**Model Answer:**")
                formatted_output.append(q['model_answer'])
            
            formatted_output.append("\n---\n")
            
        return "\n".join(formatted_output)

    def create_study_material(self, topic: str, content: str, material_type: str = "summary") -> Dict:
        """Create study material for a specific topic
        
        Args:
            topic (str): The topic to create material for
            content (str): The content to base the material on
            material_type (str): Type of material to create ("summary", "quiz", "flashcards")
            
        Returns:
            Dict: The created study material
        """
        if material_type == "summary":
            return self._generate_summary(topic, content)
        elif material_type == "quiz":
            return self._generate_quiz(topic, content)
        elif material_type == "flashcards":
            return self._generate_flashcards(topic, content)
        else:
            raise ValueError(f"Unsupported material type: {material_type}")

    def _generate_summary(self, topic: str, content: str) -> Dict:
        """Generate a structured summary of the content"""
        prompt = f"""Create a comprehensive summary of the following content about {topic}. 
        Structure the summary as follows:
        1. Key Concepts (3-5 main ideas)
        2. Detailed Explanation (2-3 paragraphs)
        3. Examples or Applications
        4. Important Terms (with definitions)
        5. Quick Reference Points
        
        Content to summarize:
        {content}
        
        Format the response as a JSON object with these sections as keys.
        """

        response = self.llm_handler.get_response(prompt, {"documents": [[content]]})
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse summary", "raw_response": response}

    def _generate_quiz(self, topic: str, content: str, num_questions: int = 5) -> Dict:
        """Generate a structured quiz with different question types"""
        prompt = f"""Create a comprehensive quiz about {topic} with the following:
        1. {num_questions} multiple-choice questions
        2. 2 short-answer questions
        3. 1 essay question for deeper understanding
        
        Base the questions on this content:
        {content}
        
        Format the response as a JSON object with:
        - multiple_choice: array of question objects (text, options, correct_answer)
        - short_answer: array of question objects (text, sample_answer)
        - essay: object with question and evaluation_criteria
        """

        response = self.llm_handler.get_response(prompt, {"documents": [[content]]})
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse quiz", "raw_response": response}

    def _generate_flashcards(self, topic: str, content: str, num_cards: int = 10) -> Dict:
        """Generate flashcards for quick review"""
        prompt = f"""Create {num_cards} flashcards about {topic} based on this content:
        {content}
        
        Include a mix of:
        - Key term definitions
        - Concept explanations
        - Problem-solving examples
        
        Format the response as a JSON array of flashcard objects with:
        - front: what appears on the front of the card
        - back: what appears on the back
        - type: "definition", "concept", or "example"
        """

        response = self.llm_handler.get_response(prompt, {"documents": [[content]]})
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse flashcards", "raw_response": response}

    def get_topic_suggestions(self, content: str) -> List[str]:
        """Suggest key topics from the content that would be good for studying"""
        prompt = """Analyze the content and suggest 5-7 key topics that would be 
        important to study. For each topic, provide a brief explanation of why it's important.
        
        Content to analyze:
        {content}
        
        Format the response as a JSON array of objects with 'topic' and 'importance' keys.
        """
        
        response = self.llm_handler.get_response(prompt, {"documents": [[content]]})
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse topic suggestions", "raw_response": response}
