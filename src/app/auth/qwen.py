"""
Qwen authentication provider.
Supports API key authentication for Alibaba Qwen.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import httpx

from .base import BaseAuthProvider, AuthResult, TokenData, TokenStatus


class QwenAuth(BaseAuthProvider):
    """Qwen authentication provider."""
    
    def __init__(self, config: Any, http_client: Any):
        """
        Initialize Qwen auth provider.
        
        Args:
            config: Application configuration
            http_client: HTTP client for making requests
        """
        super().__init__("qwen", config, http_client)
        
        # Qwen API endpoints
        self.base_url = "https://dashscope.aliyuncs.com"
        
        # Get provider-specific configuration
        self.provider_config = self.get_provider_config()
    
    async def authenticate(self, **kwargs) -> AuthResult:
        """
        Authenticate with Qwen using API key.
        
        Supported kwargs:
            api_key: Qwen API key
            key_id: Optional identifier for the API key
            
        Returns:
            AuthResult with authentication result
        """
        api_key = kwargs.get("api_key")
        key_id = kwargs.get("key_id", api_key[:8] if api_key else "unknown")
        
        if not api_key:
            return AuthResult.error_result(
                error="API key is required for Qwen authentication",
                error_code="missing_api_key",
                provider=self.provider_name,
            )
        
        # Validate API key by making a test request
        try:
            # Try to list models to validate the API key
            test_url = f"{self.base_url}/api/v1/models"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            response = await self.http_client.get(
                test_url,
                headers=headers,
                timeout=10.0,
            )
            
            if response.status_code == 200:
                # API key is valid
                token_data = TokenData(
                    access_token=api_key,
                    token_type="Bearer",
                    issued_at=datetime.utcnow(),
                    # Qwen API keys don't expire (unless revoked)
                    expires_at=None,
                    extra_data={
                        "key_id": key_id,
                        "validation_time": datetime.utcnow().isoformat(),
                    },
                )
                
                return AuthResult.success_result(token_data, self.provider_name)
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                return AuthResult.error_result(
                    error=f"API key validation failed: {response.status_code} - {error_text}",
                    error_code="api_key_validation_failed",
                    provider=self.provider_name,
                )
                
        except httpx.TimeoutException:
            return AuthResult.error_result(
                error="Connection timeout while validating API key",
                error_code="connection_timeout",
                provider=self.provider_name,
            )
        except Exception as e:
            return AuthResult.error_result(
                error=f"Authentication failed: {str(e)}",
                error_code="authentication_failed",
                provider=self.provider_name,
            )
    
    async def refresh_token(self, refresh_token: str) -> TokenData:
        """
        Refresh an expired access token.
        
        Note: Qwen API keys don't expire, so this just returns the same token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New TokenData
        """
        # Qwen API keys don't expire, so we just return the same token
        return TokenData(
            access_token=refresh_token,
            token_type="Bearer",
            issued_at=datetime.utcnow(),
            expires_at=None,
        )
    
    async def validate_token(self, token_data: TokenData) -> TokenStatus:
        """
        Validate a Qwen API key.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            TokenStatus indicating token validity
        """
        api_key = token_data.access_token
        
        if not api_key:
            return TokenStatus.INVALID
        
        # Check if token is expired (Qwen API keys don't expire)
        if token_data.is_expired():
            return TokenStatus.EXPIRED
        
        # Validate API key by making a test request
        try:
            test_url = f"{self.base_url}/api/v1/models"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            response = await self.http_client.get(
                test_url,
                headers=headers,
                timeout=5.0,
            )
            
            if response.status_code == 200:
                return TokenStatus.VALID
            elif response.status_code == 401:
                return TokenStatus.INVALID
            else:
                return TokenStatus.REFRESH_NEEDED
                
        except Exception:
            return TokenStatus.REFRESH_NEEDED
    
    def get_default_headers(self, token_data: TokenData) -> Dict[str, str]:
        """
        Get default headers for Qwen API requests.
        
        Args:
            token_data: Token data to use for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = super().get_default_headers(token_data)
        
        # Add Qwen-specific headers
        headers.update({
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",  # Disable server-sent events by default
        })
        
        return headers
    
    async def make_authenticated_request(
        self,
        method: str,
        url: str,
        token_data: TokenData,
        **kwargs
    ) -> Any:
        """
        Make an authenticated HTTP request to Qwen API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            token_data: Token data for authentication
            **kwargs: Additional arguments for HTTP request
            
        Returns:
            HTTP response
        """
        # Ensure URL is absolute
        if not url.startswith("http"):
            url = f"{self.base_url}{url}"
        
        return await super().make_authenticated_request(method, url, token_data, **kwargs)
