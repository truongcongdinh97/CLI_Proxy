"""
Translator registry for managing format conversions.
"""

from typing import Dict, Optional, Any
from .base import BaseTranslator, TranslationResult
from .openai_to_gemini import OpenAIToGeminiTranslator


class TranslatorRegistry:
    """Registry for managing translators between different formats."""
    
    def __init__(self):
        """Initialize translator registry."""
        self.translators: Dict[str, BaseTranslator] = {}
        self._initialize_default_translators()
    
    def _initialize_default_translators(self):
        """Initialize default translators."""
        # OpenAI to Gemini
        self.register_translator(OpenAIToGeminiTranslator())
        
        # Add more translators here as they are implemented
        # self.register_translator(OpenAIToClaudeTranslator())
        # self.register_translator(GeminiToOpenAITranslator())
        # self.register_translator(ClaudeToOpenAITranslator())
    
    def register_translator(self, translator: BaseTranslator):
        """
        Register a translator.
        
        Args:
            translator: Translator instance
        """
        key = f"{translator.source_format}:{translator.target_format}"
        self.translators[key] = translator
    
    def get_translator(self, source_format: str, target_format: str) -> Optional[BaseTranslator]:
        """
        Get translator for source to target format.
        
        Args:
            source_format: Source API format
            target_format: Target API format
            
        Returns:
            Translator instance, or None if not found
        """
        key = f"{source_format}:{target_format}"
        return self.translators.get(key)
    
    async def translate_request(
        self,
        source_format: str,
        target_format: str,
        request_data: Dict[str, Any]
    ) -> TranslationResult:
        """
        Translate request from source to target format.
        
        Args:
            source_format: Source API format
            target_format: Target API format
            request_data: Request data in source format
            
        Returns:
            TranslationResult with translated data
        """
        translator = self.get_translator(source_format, target_format)
        if not translator:
            return TranslationResult.error_result(
                error=f"No translator found from {source_format} to {target_format}",
                error_code="translator_not_found",
                source_format=source_format,
                target_format=target_format,
            )
        
        return await translator.translate_request(request_data)
    
    async def translate_response(
        self,
        source_format: str,
        target_format: str,
        response_data: Dict[str, Any]
    ) -> TranslationResult:
        """
        Translate response from target back to source format.
        
        Args:
            source_format: Original source format
            target_format: Target format that was used
            response_data: Response data in target format
            
        Returns:
            TranslationResult with translated data
        """
        # We need to translate from target back to source
        translator = self.get_translator(target_format, source_format)
        if not translator:
            return TranslationResult.error_result(
                error=f"No translator found from {target_format} back to {source_format}",
                error_code="translator_not_found",
                source_format=target_format,
                target_format=source_format,
            )
        
        return await translator.translate_response(response_data)
    
    def list_translators(self) -> Dict[str, str]:
        """
        List all registered translators.
        
        Returns:
            Dictionary of translator keys to descriptions
        """
        result = {}
        for key, translator in self.translators.items():
            result[key] = f"{translator.source_format} -> {translator.target_format}"
        return result
    
    def get_supported_conversions(self) -> Dict[str, list]:
        """
        Get supported format conversions.
        
        Returns:
            Dictionary with source formats as keys and list of target formats as values
        """
        conversions = {}
        for key in self.translators.keys():
            source, target = key.split(":")
            if source not in conversions:
                conversions[source] = []
            conversions[source].append(target)
        return conversions
