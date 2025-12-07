"""
OpenAI to Gemini translator.
Converts OpenAI API requests/responses to Gemini format.
"""

from typing import Dict, Any
from .base import BaseTranslator, TranslationResult


class OpenAIToGeminiTranslator(BaseTranslator):
    """Translator from OpenAI to Gemini format."""
    
    def __init__(self):
        """Initialize OpenAI to Gemini translator."""
        super().__init__("openai", "gemini")
    
    async def translate_request(self, request_data: Dict[str, Any]) -> TranslationResult:
        """
        Translate OpenAI request to Gemini format.
        
        Args:
            request_data: OpenAI request data
            
        Returns:
            TranslationResult with Gemini request data
        """
        try:
            # Extract messages
            messages = self._extract_messages(request_data)
            
            # Create Gemini contents
            contents = self._create_gemini_contents(messages)
            
            # Build Gemini request
            gemini_request = {
                "contents": contents,
            }
            
            # Copy common parameters
            if "model" in request_data:
                # Map OpenAI model names to Gemini model names
                model = request_data["model"]
                gemini_request["model"] = self._map_model_name(model)
            
            if "temperature" in request_data:
                gemini_request["temperature"] = request_data["temperature"]
            
            if "max_tokens" in request_data:
                gemini_request["maxOutputTokens"] = request_data["max_tokens"]
            
            if "top_p" in request_data:
                gemini_request["topP"] = request_data["top_p"]
            
            if "frequency_penalty" in request_data:
                gemini_request["frequencyPenalty"] = request_data["frequency_penalty"]
            
            if "presence_penalty" in request_data:
                gemini_request["presencePenalty"] = request_data["presence_penalty"]
            
            if "stop" in request_data:
                gemini_request["stopSequences"] = request_data["stop"]
            
            return TranslationResult.success_result(
                gemini_request,
                self.source_format,
                self.target_format,
            )
            
        except Exception as e:
            return TranslationResult.error_result(
                error=f"Failed to translate request: {str(e)}",
                error_code="translation_failed",
                source_format=self.source_format,
                target_format=self.target_format,
            )
    
    async def translate_response(self, response_data: Dict[str, Any]) -> TranslationResult:
        """
        Translate Gemini response to OpenAI format.
        
        Args:
            response_data: Gemini response data
            
        Returns:
            TranslationResult with OpenAI response data
        """
        try:
            # Extract text from Gemini response
            choices = []
            
            if "candidates" in response_data:
                for candidate in response_data["candidates"]:
                    if "content" in candidate and "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            if "text" in part:
                                choice = {
                                    "index": len(choices),
                                    "message": {
                                        "role": "assistant",
                                        "content": part["text"],
                                    },
                                    "finish_reason": candidate.get("finishReason", "stop"),
                                }
                                choices.append(choice)
            
            # Build OpenAI response
            openai_response = {
                "id": response_data.get("id", f"gemini-{hash(str(response_data))}"),
                "object": "chat.completion",
                "created": response_data.get("created", 0),
                "model": response_data.get("model", "gemini-pro"),
                "choices": choices,
                "usage": {
                    "prompt_tokens": response_data.get("usageMetadata", {}).get("promptTokenCount", 0),
                    "completion_tokens": response_data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                    "total_tokens": response_data.get("usageMetadata", {}).get("totalTokenCount", 0),
                },
            }
            
            return TranslationResult.success_result(
                openai_response,
                self.target_format,
                self.source_format,
            )
            
        except Exception as e:
            return TranslationResult.error_result(
                error=f"Failed to translate response: {str(e)}",
                error_code="translation_failed",
                source_format=self.target_format,
                target_format=self.source_format,
            )
    
    def _map_model_name(self, openai_model: str) -> str:
        """
        Map OpenAI model name to Gemini model name.
        
        Args:
            openai_model: OpenAI model name
            
        Returns:
            Gemini model name
        """
        model_mapping = {
            "gpt-3.5-turbo": "gemini-pro",
            "gpt-4": "gemini-pro",
            "gpt-4-turbo": "gemini-1.5-pro",
            "gpt-4o": "gemini-1.5-flash",
        }
        
        return model_mapping.get(openai_model, "gemini-pro")
