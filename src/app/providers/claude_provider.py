"""
Claude provider implementation.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseProvider, ProviderConfig, ProviderStatus
from ..auth.manager import AuthManager
from ..utils.http_client import HTTPClient


class ClaudeProvider(BaseProvider):
    """Claude AI provider implementation."""
    
    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: AuthManager,
        http_client: HTTPClient
    ):
        """
        Initialize Claude provider.
        
        Args:
            config: Provider configuration
            auth_manager: Authentication manager
            http_client: HTTP client
        """
        super().__init__(config, auth_manager, http_client)
        self.base_url = config.base_url or "https://api.anthropic.com"
        self.api_version = "v1"
        self.api_key = config.api_key
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        # Set default headers with API key
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
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
        # Claude models typically start with "claude-"
        return model.startswith("claude-")
    
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
        # Convert OpenAI-style messages to Claude format
        system_message = None
        claude_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content
            elif role == "user":
                claude_messages.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                claude_messages.append({
                    "role": "assistant",
                    "content": content
                })
        
        # Prepare request body (Claude format)
        request_body = {
            "model": model,
            "messages": claude_messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        }
        
        # Add system message if present
        if system_message:
            request_body["system"] = system_message
        
        # Add optional parameters
        optional_params = ["stop_sequences", "top_k"]
        for param in optional_params:
            if param in kwargs:
                request_body[param] = kwargs[param]
        
        # Make request
        response = await self.http_client.post(
            f"{self.base_url}/{self.api_version}/messages",
            json=request_body,
            timeout=kwargs.get("timeout", 30.0)
        )
        
        if response.status_code != 200:
            raise Exception(f"Claude API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Convert Claude response to OpenAI format
        if "content" in result and len(result["content"]) > 0:
            content = result["content"][0]["text"]
            
            # Extract token counts
            input_tokens = result.get("usage", {}).get("input_tokens", 0)
            output_tokens = result.get("usage", {}).get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
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
                            "content": content,
                        },
                        "finish_reason": result.get("stop_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            }
        
        raise Exception("Invalid response format from Claude API")
    
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
                        "created": 0,  # Not provided by Claude
                        "owned_by": "anthropic",
                        "permission": [],
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
