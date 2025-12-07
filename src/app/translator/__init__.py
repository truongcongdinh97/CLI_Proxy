"""
Translator module for CLI Proxy API.
Converts between different AI provider API formats.
"""

from .base import BaseTranslator, TranslationResult
from .openai_to_gemini import OpenAIToGeminiTranslator
from .registry import TranslatorRegistry

__all__ = [
    "BaseTranslator",
    "TranslationResult",
    "OpenAIToGeminiTranslator",
    "TranslatorRegistry",
]
