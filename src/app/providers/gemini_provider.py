"""
Gemini provider implementation.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderStatus
from ..auth.manager import AuthManager
from ..utils.http_client import HTTPClient


class GeminiProvider(BaseProvider):
    """Gemini AI provider implementation."""
    
    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: AuthManager,
        http_client: HTTPClient
    ):
        """
        Initialize Gemini provider.
        
        Args:
            config: Provider configuration
            auth_manager: Authentication manager
            http_client: HTTP client
        """
        super().__init__(config, auth_manager, http_client)
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com"
        self.api_version = "v1beta"
        self.api_key = config.api_key
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        # Set default headers
        self.http_client.set_default_headers({
            "Content-Type": "application/json",
        })
        
        self._set_status(ProviderStatus.HEALTHY)
    
    async def health_check(self) -> bool:
        """
        Perform health check.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - try to list models
            url = f"{self.base_url}/{self.api_version}/models"
            if self.api_key:
                url += f"?key={self.api_key}"
            response = await self.http_client.get(
                url,
                timeout=10.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def can_handle_model(self, model: str) -> bool:
        """
        Check if provider can handle a model.
        
        Args:
            model: Model name
            
        Returns:
            True if can handle, False otherwise
        """
        # Gemini models typically start with "gemini-" or "models/gemini-"
        return model.startswith("gemini-") or "gemini" in model.lower()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            messages: List of messages
            model: Model name
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if gemini_contents and gemini_contents[-1]["role"] == "user":
                    gemini_contents[-1]["parts"][0]["text"] = (
                        content + "\n\n" + gemini_contents[-1]["parts"][0]["text"]
                    )
                else:
                    # Store system message for later
                    pass
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
        
        # Prepare request body
        request_body = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.95),
                "topK": kwargs.get("top_k", 40),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "stopSequences": kwargs.get("stop", []),
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Build URL with API key
        url = f"{self.base_url}/{self.api_version}/models/{model}:generateContent"
        if self.api_key:
            url += f"?key={self.api_key}"
        
        # Make request
        response = await self.http_client.post(
            url,
            json=request_body,
            timeout=kwargs.get("timeout", 30.0)
        )
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Convert Gemini response to OpenAI format
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                text = candidate["content"]["parts"][0].get("text", "")
                
                # Extract token counts
                prompt_tokens = result.get("usageMetadata", {}).get("promptTokenCount", 0)
                completion_tokens = result.get("usageMetadata", {}).get("candidatesTokenCount", 0)
                total_tokens = prompt_tokens + completion_tokens
                
                return {
                    "id": f"chatcmpl-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "object": "chat.completion",
                    "created": int(datetime.utcnow().timestamp()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": candidate.get("finishReason", "stop").lower(),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                }
        
        raise Exception("Invalid response format from Gemini API")
    
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._set_status(ProviderStatus.OFFLINE)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information
        """
        try:
            url = f"{self.base_url}/{self.api_version}/models"
            if self.api_key:
                url += f"?key={self.api_key}"
            
            response = await self.http_client.get(
                url,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                models = []
                
                for model_info in result.get("models", []):
                    models.append({
                        "id": model_info["name"],
                        "object": "model",
                        "created": 0,  # Not provided by Gemini
                        "owned_by": "google",
                        "permission": [],
                        "root": model_info["name"],
                        "parent": None,
                    })
                
                return models
            else:
                return []
                
        except Exception:
            return []
    
    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information, or None if not found
        """
        try:
            models = await self.list_models()
            for m in models:
                if m["id"] == model or m["id"].endswith(f"/{model}"):
                    return m
            return None
        except Exception:
            return None
