from crewai import Agent, Task, Crew, Process
from litellm import completion
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LearningAgent:
    def __init__(self):
        # Initialize Groq through LiteLLM
        self.model = "groq/mixtral-8x7b-32768"
        
        # Create CrewAI agents
        self.researcher = Agent(
            role='Research Analyst',
            goal='Analyze content and extract key information',
            backstory='Expert at analyzing educational content and identifying key concepts',
            tools=[self.analyze_content],
            model=self.model
        )
        
        self.educator = Agent(
            role='Education Specialist',
            goal='Create effective educational materials',
            backstory='Experienced educator skilled in creating learning materials',
            tools=[self.create_educational_content],
            model=self.model
        )
        
        self.evaluator = Agent(
            role='Assessment Expert',
            goal='Create and evaluate assessments',
            backstory='Expert in creating effective quizzes and evaluations',
            tools=[self.create_assessment],
            model=self.model
        )

    def analyze_content(self, content: str) -> Dict:
        """Analyze content using Groq through LiteLLM"""
        response = completion(
            model=self.model,
            messages=[{
                "role": "system",
                "content": "Analyze the following educational content and extract key concepts, terms, and relationships."
            }, {
                "role": "user",
                "content": content
            }]
        )
        return json.loads(response.choices[0].message.content)

    def create_educational_content(self, topic: str, content: str, material_type: str) -> Dict:
        """Create educational content using CrewAI"""
        crew = Crew(
            agents=[self.researcher, self.educator],
            tasks=[
                Task(
                    description=f"Analyze content about {topic} and identify key points",
                    agent=self.researcher
                ),
                Task(
                    description=f"Create {material_type} material about {topic}",
                    agent=self.educator
                )
            ],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return json.loads(result)

    def create_assessment(self, topic: str, content: str, assessment_type: str) -> Dict:
        """Create assessment materials using CrewAI"""
        crew = Crew(
            agents=[self.researcher, self.evaluator],
            tasks=[
                Task(
                    description=f"Analyze content about {topic} for assessment",
                    agent=self.researcher
                ),
                Task(
                    description=f"Create {assessment_type} assessment for {topic}",
                    agent=self.evaluator
                )
            ],
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return json.loads(result)

    # ... rest of the methods can be updated similarly ...
