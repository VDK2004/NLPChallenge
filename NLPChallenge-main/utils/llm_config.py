from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LLMConfig:
    name: str
    provider: str
    model_id: str
    max_tokens: int
    temperature: float
    requires_api_key: bool = True

# Available LLM configurations
LLM_CONFIGS = {
    "Groq - Mixtral 8x7B": LLMConfig(
        name="Groq - Mixtral 8x7B",
        provider="groq",
        model_id="mixtral-8x7b-32768",
        max_tokens=1000,
        temperature=0.5
    ),
    "HuggingFace - BERT Base": LLMConfig(
        name="HuggingFace - BERT Base",
        provider="huggingface",
        model_id="bert-base-uncased",
        max_tokens=512,
        temperature=0.5
    ),
    "HuggingFace - GPT2": LLMConfig(
        name="HuggingFace - GPT2",
        provider="huggingface",
        model_id="gpt2",
        max_tokens=1000,
        temperature=0.5
    ),
    "HuggingFace - T5": LLMConfig(
        name="HuggingFace - T5",
        provider="huggingface",
        model_id="t5-base",
        max_tokens=512,
        temperature=0.5
    )
}

def get_available_models() -> List[str]:
    """Get list of available model names"""
    return list(LLM_CONFIGS.keys())

def get_model_config(model_name: str) -> LLMConfig:
    """Get configuration for a specific model"""
    return LLM_CONFIGS.get(model_name)
