from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from typing import List, Dict, Optional
import yaml
import json
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