from dataclasses import dataclass
from typing import Dict, List

@dataclass
class LLMConfig:
    model_id: str
    provider: str
    context_window: int
    description: str
    temperature: float = 0.7
    max_tokens: int = 1000

# Available models configuration
AVAILABLE_MODELS = {
    
    "Groq - Mixtral 8x7B": LLMConfig(
        model_id="mixtral-8x7b-32768",
        provider="Groq",
        context_window=32768,
        description="Fast open-source model with good performance.",
        temperature=0.7,
        max_tokens=4096
    ),
    "Groq - LLaMA2 70B": LLMConfig(
        model_id="llama2-70b-4096",
        provider="Groq",
        context_window=4096,
        description="Powerful open-source model for focused tasks.",
        temperature=0.7,
        max_tokens=1024
    ),
    "T5-Large": LLMConfig(
        model_id="t5-large",
        provider="HuggingFace",
        context_window=512,
        description="Efficient and reliable model for text generation tasks",
        temperature=0.7,
        max_tokens=512
    ),
    "MPT-7B-Instruct": LLMConfig(
        model_id="mosaicml/mpt-7b-instruct",
        provider="HuggingFace",
        context_window=2048,
        description="7B parameter instruction-tuned model for chat and text generation",
        temperature=0.7,
        max_tokens=1024
    ),
    "FLAN-T5-XL": LLMConfig(
        model_id="google/flan-t5-xl",
        provider="HuggingFace",
        context_window=512,
        description="Versatile open-source model good for various NLP tasks",
        temperature=0.7,
        max_tokens=512
    ),
    "Mistral-7B": LLMConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        provider="HuggingFace",
        context_window=4096,
        description="Powerful open-source large language model for instruction following",
        temperature=0.7,
        max_tokens=1024
    ),
    "BART-Large-CNN": LLMConfig(
        model_id="facebook/bart-large-cnn",
        provider="HuggingFace",
        context_window=1024,
        description="Specialized model for summarization tasks",
        temperature=0.7,
        max_tokens=1024
    )
}

def get_available_models() -> List[str]:
    """Get list of available model names"""
    return list(AVAILABLE_MODELS.keys())

def get_model_config(model_name: str) -> LLMConfig:
    """Get configuration for a specific model"""
    return AVAILABLE_MODELS.get(model_name)

def get_models_by_provider(provider: str) -> Dict[str, LLMConfig]:
    """Get all models for a specific provider"""
    return {name: config for name, config in AVAILABLE_MODELS.items() 
            if config.provider.lower() == provider.lower()}
