from typing import List, Dict, Optional
import json
import re
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
import yaml
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LearningCrew:
    def __init__(self):
        # Load agent configurations
        self.config = self._load_agent_config()
        self.model = "groq/mixtral-8x7b-32768"
        
        # Initialize tools
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize agents
        self.researcher = Agent(
            role='Research Analyst',
            goal='Analyze content and extract key information',
            backstory="""You are an expert researcher with a deep understanding of various subjects.
                        Your goal is to analyze content thoroughly and identify key concepts and relationships.
                        You excel at breaking down complex topics into understandable components.""",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        self.writer = Agent(
            role='Content Writer',
            goal='Create clear and engaging educational content',
            backstory="""You are a skilled educational content writer.
                        You excel at explaining complex topics in simple terms and creating
                        engaging learning materials that keep students interested.""",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True
        )
        
        self.evaluator = Agent(
            role='Assessment Expert',
            goal='Create effective assessments and quizzes',
            backstory="""You are an expert in educational assessment.
                        You know how to create questions that test understanding rather than just memorization.
                        You excel at creating varied and engaging assessments.""",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True
        )

    def _load_agent_config(self) -> dict:
        """Load agent configurations from YAML file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def create_summary(self, topic: str, content: str, summary_type: str = "comprehensive") -> Dict:
        """Create a summary using the research and writing agents"""
        crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[
                Task(
                    description=f"""Analyze the following content about {topic}.
                                   Identify key concepts, main arguments, and supporting evidence.
                                   Content: {content}""",
                    agent=self.researcher
                ),
                Task(
                    description=f"""Create a {summary_type} summary based on the research analysis.
                                   Include key concepts, practical applications, and technical details if relevant.
                                   Make it engaging and easy to understand.""",
                    agent=self.writer
                )
            ],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return self._format_summary_result(result)

    def create_quiz(self, topic: str, content: str, num_questions: int = 5, difficulty: str = "medium") -> Dict:
        """Create a quiz using the research and evaluation agents"""
        crew = Crew(
            agents=[self.researcher, self.evaluator],
            tasks=[
                Task(
                    description=f"""Analyze this content about {topic} for quiz creation.
                                   Identify key testable concepts and knowledge points.
                                   Content: {content}""",
                    agent=self.researcher
                ),
                Task(
                    description=f"""Create a {difficulty} difficulty quiz with {num_questions} questions.
                                   Include a mix of question types (multiple choice, true/false, open-ended).
                                   Provide detailed explanations for all answers.""",
                    agent=self.evaluator
                )
            ],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return self._format_quiz_result(result)

    def _format_summary_result(self, result: str) -> Dict:
        """Format the summary result into a structured dictionary"""
        try:
            # Assuming result is JSON string or can be parsed as JSON
            return json.loads(result)
        except json.JSONDecodeError:
            # If not JSON, create a basic structure
            return {
                "metadata": {
                    "estimated_reading_time": "5 minutes",
                    "complexity_level": "medium"
                },
                "executive_summary": result,
                "key_concepts": [],
                "detailed_analysis": {
                    "main_arguments": [],
                    "supporting_evidence": []
                }
            }

    def _format_quiz_result(self, result: str) -> Dict:
        """Format the quiz result into a structured dictionary"""
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # If not JSON, return error
            return {
                "error": "Failed to parse quiz result",
                "raw_response": result
            }

class SummaryHandler:
    def __init__(self, chat_handler):
        self.chat_handler = chat_handler
    
    def create_summary(self, topic: str, content: str, summary_type: str = "comprehensive") -> Dict:
        """Create a summary of the content
        
        Args:
            topic (str): Topic of the content
            content (str): Content to summarize
            summary_type (str): Type of summary ("comprehensive", "brief", "technical")
            
        Returns:
            Dict: Structured summary data
        """
        if summary_type == "comprehensive":
            return self._create_comprehensive_summary(topic, content)
        elif summary_type == "brief":
            return self._create_brief_summary(topic, content)
        elif summary_type == "technical":
            return self._create_technical_summary(topic, content)
        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")
    
    def _create_comprehensive_summary(self, topic: str, content: str) -> Dict:
        prompt = f"""You are a specialized AI assistant that creates structured summaries in JSON format.
        Create a comprehensive summary of the content about {topic}.
        
        IMPORTANT: Your response must be ONLY valid JSON, with no additional text or markdown formatting.
        
        Required JSON structure:
        {{
            "metadata": {{
                "estimated_reading_time": "X minutes",
                "complexity_level": "Basic/Intermediate/Advanced",
                "prerequisites": ["prerequisite1", "prerequisite2"]
            }},
            "executive_summary": "2-3 sentence overview",
            "key_concepts": [
                "concept 1 with brief explanation",
                "concept 2 with brief explanation"
            ],
            "detailed_analysis": {{
                "main_arguments": [
                    "argument 1",
                    "argument 2"
                ],
                "supporting_evidence": [
                    "evidence 1",
                    "evidence 2"
                ],
                "relationships": [
                    "relationship 1",
                    "relationship 2"
                ]
            }},
            "technical_details": [
                "detail 1",
                "detail 2"
            ],
            "practical_applications": [
                "application 1",
                "application 2"
            ],
            "key_terms": {{
                "term1": "definition1",
                "term2": "definition2"
            }}
        }}

        Content to summarize:
        {content}
        
        Remember: Return ONLY the JSON object, no other text.
        """
        
        try:
            response = self.chat_handler.get_response(prompt, {"documents": [[content]]})
            # Clean the response to handle potential formatting issues
            response = response.strip()
            
            # Try to find JSON content if wrapped in other text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group()
            
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                # If direct parsing fails, try to clean up common issues
                response = response.replace('\n', ' ').replace('\r', '')
                response = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', response)  # Fix unquoted keys
                response = response.replace("'", '"')  # Replace single quotes with double quotes
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "error": "Failed to parse summary",
                "raw_response": response,
                "error_details": str(e)
            }
        except Exception as e:
            print(f"Error creating summary: {str(e)}")
            return {"error": f"Error creating summary: {str(e)}"}

    def _chunk_content(self, content: str, chunk_size: int = 4000) -> List[str]:
        """Split content into smaller chunks to avoid token limits
        
        Args:
            content (str): Content to split
            chunk_size (int): Approximate size of each chunk in characters
            
        Returns:
            List[str]: List of content chunks
        """
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    def _create_brief_summary(self, topic: str, content: str) -> Dict:
        chunks = self._chunk_content(content)
        chunk_summaries = []
        
        chunk_prompt_template = """You are a specialized AI assistant that creates structured summaries in JSON format.
        Create a brief summary of part {i} of {total} about {topic}.
        
        IMPORTANT: Your response must be ONLY valid JSON, with no additional text or markdown formatting.
        
        Required JSON structure:
        {{
            "summary": "Brief summary text",
            "key_points": [
                "key point 1",
                "key point 2"
            ]
        }}

        Content to summarize:
        {chunk}
        
        Remember: Return ONLY the JSON object, no other text.
        """
        
        for i, chunk in enumerate(chunks, 1):
            try:
                prompt = chunk_prompt_template.format(
                    i=i,
                    total=len(chunks),
                    topic=topic,
                    chunk=chunk
                )
                response = self.chat_handler.get_response(prompt, {"documents": [[chunk]]})
                # Clean the response
                response = response.strip()
                
                # Try to find JSON content if wrapped in other text
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group()
                
                try:
                    summary = json.loads(response)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to clean up common issues
                    response = response.replace('\n', ' ').replace('\r', '')
                    response = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', response)
                    response = response.replace("'", '"')
                    summary = json.loads(response)
                
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue
        
        if not chunk_summaries:
            return {"error": "Failed to generate any summaries"}
        
        combined_prompt = f"""You are a specialized AI assistant that creates structured summaries in JSON format.
        Combine these summaries into a single coherent summary about {topic}.
        
        IMPORTANT: Your response must be ONLY valid JSON, with no additional text or markdown formatting.
        
        Required JSON structure:
        {{
            "overview": "Brief overview of the entire content",
            "key_takeaways": [
                "takeaway 1",
                "takeaway 2"
            ],
            "important_points": [
                "point 1",
                "point 2"
            ]
        }}

        Summaries to combine:
        {json.dumps(chunk_summaries, indent=2)}
        
        Remember: Return ONLY the JSON object, no other text.
        """
        
        try:
            final_response = self.chat_handler.get_response(combined_prompt, {"documents": [[json.dumps(chunk_summaries)]]})
            # Clean the response
            final_response = final_response.strip()
            
            # Try to find JSON content if wrapped in other text
            json_match = re.search(r'\{.*\}', final_response, re.DOTALL)
            if json_match:
                final_response = json_match.group()
            
            try:
                return json.loads(final_response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to clean up common issues
                final_response = final_response.replace('\n', ' ').replace('\r', '')
                final_response = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', final_response)
                final_response = final_response.replace("'", '"')
                return json.loads(final_response)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in final combination: {str(e)}")
            print(f"Raw response: {final_response}")
            return chunk_summaries[0]  # Return the first chunk summary as fallback
        except Exception as e:
            print(f"Error combining summaries: {str(e)}")
            return chunk_summaries[0]  # Return the first chunk summary as fallback

    def _create_technical_summary(self, topic: str, content: str) -> Dict:
        prompt = f"""Create a technical summary of the content about {topic}.
        Focus on:
        1. Technical Concepts
           - Definitions
           - Formulas
           - Algorithms
        2. Implementation Details
        3. Technical Requirements
        4. Best Practices
        5. Common Pitfalls
        
        Content to summarize:
        {content}
        
        Format as JSON with these sections as keys.
        Include code snippets or pseudocode where relevant.
        """
        
        response = self.chat_handler.get_response(prompt, {"documents": [[content]]})
        try:
            return json.loads(response)
        except:
            return {"error": "Failed to parse summary", "raw_response": response}

    def create_study_notes(self, topic: str, content: str) -> Dict:
        """Create structured study notes from the content
        
        Args:
            topic (str): Topic of the content
            content (str): Content to create notes from
            
        Returns:
            Dict: Structured study notes
        """
        prompt = f"""Create detailed study notes about {topic}.
        Return the response in the following JSON format:
        {{
            "main_topics": [
                {{
                    "title": "Topic 1",
                    "subtopics": [
                        "Subtopic 1.1",
                        "Subtopic 1.2"
                    ]
                }}
            ],
            "key_points": [
                {{
                    "topic": "Topic 1",
                    "points": [
                        "Key point 1",
                        "Key point 2"
                    ]
                }}
            ],
            "examples": [
                {{
                    "topic": "Topic 1",
                    "examples": [
                        "Example 1",
                        "Example 2"
                    ]
                }}
            ],
            "practice_problems": [
                {{
                    "question": "Problem 1",
                    "solution": "Solution 1"
                }}
            ],
            "review_questions": [
                "Question 1",
                "Question 2"
            ],
            "additional_resources": [
                {{
                    "title": "Resource 1",
                    "description": "Description 1",
                    "type": "book/article/video"
                }}
            ]
        }}

        Content for notes:
        {content}
        
        Ensure the response is a valid JSON object with the exact structure shown above.
        """
        
        try:
            response = self.chat_handler.get_response(prompt, {"documents": [[content]]})
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "error": "Failed to parse study notes",
                "raw_response": response,
                "error_details": str(e)
            }
        except Exception as e:
            print(f"Error creating study notes: {str(e)}")
            return {"error": f"Error creating study notes: {str(e)}"}

    def create_mind_map(self, topic: str, content: str) -> Dict:
        """Create a mind map structure of the content
        
        Args:
            topic (str): Central topic for the mind map
            content (str): Content to structure
            
        Returns:
            Dict: Mind map structure
        """
        prompt = f"""Create a mind map structure for {topic}.
        Return the response in the following JSON format:
        {{
            "central_topic": {{
                "title": "{topic}",
                "description": "Brief description of the topic"
            }},
            "main_branches": [
                {{
                    "title": "Branch 1",
                    "description": "Description of branch 1",
                    "sub_branches": [
                        {{
                            "title": "Sub-branch 1.1",
                            "description": "Description of sub-branch 1.1",
                            "keywords": ["keyword1", "keyword2"],
                            "connections": ["Branch 2", "Branch 3"]
                        }}
                    ]
                }}
            ],
            "relationships": [
                {{
                    "from": "Branch 1",
                    "to": "Branch 2",
                    "description": "How these branches are related"
                }}
            ]
        }}

        Content to map:
        {content}
        
        Ensure the response is a valid JSON object with the exact structure shown above.
        """
        
        try:
            response = self.chat_handler.get_response(prompt, {"documents": [[content]]})
            # Clean the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "error": "Failed to parse mind map",
                "raw_response": response,
                "error_details": str(e)
            }
        except Exception as e:
            print(f"Error creating mind map: {str(e)}")
            return {"error": f"Error creating mind map: {str(e)}"}

    def generate_quiz(self, topic: str, content: str, num_questions: int = 5, difficulty: str = "medium") -> Dict:
        """
        Generate a quiz based on the content.
        
        Args:
            topic (str): The topic of the content
            content (str): The content to generate questions from
            num_questions (int): Number of questions to generate
            difficulty (str): Difficulty level ("easy", "medium", "hard")
            
        Returns:
            Dict: Quiz data in JSON format
        """
        prompt = f"""You are a specialized AI assistant that creates educational quizzes in JSON format.
        Create a quiz about {topic} with {num_questions} {difficulty}-level questions.
        
        IMPORTANT: Your response must be ONLY valid JSON, with no additional text or markdown formatting.
        
        Required JSON structure:
        {{
            "quiz_metadata": {{
                "topic": "{topic}",
                "difficulty": "{difficulty}",
                "estimated_time": "X minutes",
                "total_points": X
            }},
            "questions": [
                {{
                    "id": 1,
                    "type": "multiple_choice",  # or "true_false" or "open_ended"
                    "question": "Question text",
                    "points": X,
                    "options": [  # Only for multiple_choice
                        "Option A",
                        "Option B",
                        "Option C",
                        "Option D"
                    ],
                    "correct_answer": "Correct answer",
                    "explanation": "Explanation of the correct answer"
                }},
                # More questions...
            ]
        }}

        Content to create quiz from:
        {content}
        
        Guidelines:
        1. Mix different question types (multiple_choice, true_false, open_ended)
        2. Ensure questions test different levels of understanding
        3. Make answers and explanations clear and educational
        4. For {difficulty} difficulty:
           - Easy: Basic recall and understanding
           - Medium: Application and analysis
           - Hard: Analysis, evaluation, and synthesis
        
        Remember: Return ONLY the JSON object, no other text.
        """
        
        try:
            response = self.chat_handler.get_response(prompt, {"documents": [[content]]})
            # Clean the response
            response = response.strip()
            
            # Try to find JSON content if wrapped in other text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group()
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to clean up common issues
                response = response.replace('\n', ' ').replace('\r', '')
                response = re.sub(r'(?<!\\)"(\w+)":', r'"\1":', response)
                response = response.replace("'", '"')
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "error": "Failed to generate quiz",
                "raw_response": response,
                "error_details": str(e)
            }
        except Exception as e:
            print(f"Error generating quiz: {str(e)}")
            return {"error": f"Error generating quiz: {str(e)}"}
