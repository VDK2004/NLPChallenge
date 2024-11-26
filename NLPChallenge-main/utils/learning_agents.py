from typing import List, Dict
import random

class LearningAgent:
    def __init__(self, chat_handler):
        self.chat_handler = chat_handler

    def generate_cheatsheet(self, context_docs: Dict) -> str:
        """Generate a concise summary of the document"""
        prompt = f"""Create a comprehensive but concise cheatsheet or summary of the following content. 
        Include the most important concepts, key points, and main ideas. 
        Format the output with clear sections, bullet points, and highlights.
        
        Content to summarize:
        {context_docs.get('documents', [[]])[0]}
        
        Cheatsheet:"""

        return self.chat_handler.get_response("", {"documents": [[prompt]]})

    def generate_quiz(self, context_docs: Dict, num_questions: int = 5, question_type: str = "multiple_choice") -> List[Dict]:
        """Generate quiz questions based on the document content"""
        if question_type == "multiple_choice":
            prompt = f"""Create {num_questions} multiple-choice questions based on the following content. 
            For each question, provide 4 options (A, B, C, D) with one correct answer.
            Format each question as follows:
            Q1. [Question text]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct: [A/B/C/D]
            
            Make sure the questions test understanding of key concepts and important details.
            
            Content for quiz:
            {context_docs.get('documents', [[]])[0]}
            
            Questions:"""
        else:
            prompt = f"""Create {num_questions} open-ended questions based on the following content. 
            For each question, also provide a model answer.
            Format each question as follows:
            Q1. [Question text]
            Answer: [Detailed answer]
            
            Make sure the questions test deep understanding and critical thinking.
            
            Content for quiz:
            {context_docs.get('documents', [[]])[0]}
            
            Questions:"""

        response = self.chat_handler.get_response("", {"documents": [[prompt]]})
        
        # Parse the response into a structured format
        questions = []
        current_question = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Q'):
                if current_question:
                    questions.append(current_question)
                current_question = {'question': line.split('. ', 1)[1]}
                if question_type == "multiple_choice":
                    current_question['options'] = []
                
            elif question_type == "multiple_choice":
                if line.startswith(('A)', 'B)', 'C)', 'D)')):
                    current_question['options'].append(line)
                elif line.startswith('Correct:'):
                    current_question['correct_answer'] = line.split(': ')[1].strip()
                    
            elif line.startswith('Answer:'):
                current_question['answer'] = line.split(': ')[1]
                
        if current_question:
            questions.append(current_question)
            
        return questions

    def format_quiz_for_display(self, questions: List[Dict], question_type: str = "multiple_choice") -> str:
        """Format quiz questions for display in Streamlit"""
        formatted_output = []
        
        for i, q in enumerate(questions, 1):
            formatted_output.append(f"### Question {i}")
            formatted_output.append(q['question'])
            
            if question_type == "multiple_choice":
                formatted_output.extend(q['options'])
                formatted_output.append(f"\n**Correct Answer:** {q['correct_answer']}")
            else:
                formatted_output.append("\n**Model Answer:**")
                formatted_output.append(q['answer'])
            
            formatted_output.append("\n---\n")
            
        return "\n".join(formatted_output)
