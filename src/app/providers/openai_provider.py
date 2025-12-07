"""
OpenAI provider implementation.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderStatus
from ..auth.manager import AuthManager
from ..utils.http_client import HTTPClient


class OpenAIProvider(BaseProvider):
    """OpenAI AI provider implementation."""
    
    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: AuthManager,
        http_client: HTTPClient
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            config: Provider configuration
            auth_manager: Authentication manager
            http_client: HTTP client
        """
        super().__init__(config, auth_manager, http_client)
        self.base_url = config.base_url or "https://api.openai.com"
        self.api_version = "v1"
        self.api_key = config.api_key
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        # Set default headers with API key
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.http_client.set_default_headers(headers)
        
        self._set_status(ProviderStatus.HEALTHY)
    
    async def health_check(self) -> bool:
        """
        Perform health check.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - try to list models
            response = await self.http_client.get(
                f"{self.base_url}/{self.api_version}/models",
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
        # OpenAI models typically start with "gpt-", "text-", "code-", etc.
        openai_prefixes = ["gpt-", "text-", "code-", "davinci-", "curie-", "babbage-", "ada-"]
        return any(model.startswith(prefix) for prefix in openai_prefixes)
    
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
        # Prepare request body (OpenAI format)
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": kwargs.get("stream", False),
        }
        
        # Add optional parameters
        optional_params = ["frequency_penalty", "presence_penalty", "stop", "logit_bias", "user"]
        for param in optional_params:
            if param in kwargs:
                request_body[param] = kwargs[param]
        
        # Make request
        response = await self.http_client.post(
            f"{self.base_url}/{self.api_version}/chat/completions",
            json=request_body,
            timeout=kwargs.get("timeout", 30.0)
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Ensure response has expected format
        if "choices" not in result or not result["choices"]:
            raise Exception("Invalid response format from OpenAI API")
        
        return result
    
    async def models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information
        """
        try:
            response = await self.http_client.get(
                f"{self.base_url}/{self.api_version}/models",
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                models = []
                
                for model_info in result.get("data", []):
                    models.append({
                        "id": model_info["id"],
                        "object": "model",
                        "created": model_info.get("created", 0),
                        "owned_by": model_info.get("owned_by", "openai"),
                        "permission": model_info.get("permission", []),
                        "root": model_info["id"],
                        "parent": None,
                    })
                
                return models
            else:
                return []
                
        except Exception:
            return []

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._set_status(ProviderStatus.OFFLINE)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        return await self.models()
    
    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            models = await self.list_models()
            for m in models:
                if m["id"] == model:
                    return m
            return None
        except Exception:
            return None
