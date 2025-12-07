"""
Claude authentication provider.
Supports API key authentication for Anthropic Claude.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import httpx

from .base import BaseAuthProvider, AuthResult, TokenData, TokenStatus, PKCECodes


class ClaudeAuth(BaseAuthProvider):
    """Claude authentication provider."""
    
    def __init__(self, config: Any, http_client: Any):
        """
        Initialize Claude auth provider.
        
        Args:
            config: Application configuration
            http_client: HTTP client for making requests
        """
        super().__init__("claude", config, http_client)
        
        # Claude API endpoints
        self.base_url = "https://api.anthropic.com"
        self.auth_url = "https://claude.ai/oauth/authorize"
        self.token_url = "https://claude.ai/oauth/token"
        
        # Get provider-specific configuration
        self.provider_config = self.get_provider_config()
    
    async def authenticate(self, **kwargs) -> AuthResult:
        """
        Authenticate with Claude using API key.
        
        Supported kwargs:
            api_key: Claude API key
            key_id: Optional identifier for the API key
            base_url: Optional custom base URL for Claude-compatible APIs
            
        Returns:
            AuthResult with authentication result
        """
        api_key = kwargs.get("api_key")
        key_id = kwargs.get("key_id", api_key[:8] if api_key else "unknown")
        base_url = kwargs.get("base_url", self.base_url)
        
        if not api_key:
            return AuthResult.error_result(
                error="API key is required for Claude authentication",
                error_code="missing_api_key",
                provider=self.provider_name,
            )
        
        # Validate API key by making a test request
        try:
            # Try to get Claude version to validate the API key
            test_url = f"{base_url}/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            
            # Create a minimal test request
            test_data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }
            
            response = await self.http_client.post(
                test_url,
                headers=headers,
                json=test_data,
                timeout=10.0,
            )
            
            # Claude returns 400 for invalid API keys, 200/429 for valid ones
            if response.status_code in [200, 400, 429]:
                # API key is valid (400 is for missing model access, 429 is rate limit)
                token_data = TokenData(
                    access_token=api_key,
                    token_type="Bearer",
                    issued_at=datetime.utcnow(),
                    # Claude API keys don't expire (unless revoked)
                    expires_at=None,
                    extra_data={
                        "key_id": key_id,
                        "base_url": base_url,
                        "validation_time": datetime.utcnow().isoformat(),
                        "anthropic_version": "2023-06-01",
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
        
        Note: Claude API keys don't expire, so this just returns the same token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New TokenData
        """
        # Claude API keys don't expire, so we just return the same token
        # In a real implementation, we might want to re-validate the API key
        return TokenData(
            access_token=refresh_token,  # In this case, refresh_token is the API key
            token_type="Bearer",
            issued_at=datetime.utcnow(),
            expires_at=None,
        )
    
    async def validate_token(self, token_data: TokenData) -> TokenStatus:
        """
        Validate a Claude API key.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            TokenStatus indicating token validity
        """
        api_key = token_data.access_token
        base_url = token_data.extra_data.get("base_url", self.base_url) if token_data.extra_data else self.base_url
        
        if not api_key:
            return TokenStatus.INVALID
        
        # Check if token is expired (Claude API keys don't expire)
        if token_data.is_expired():
            return TokenStatus.EXPIRED
        
        # Validate API key by making a test request
        try:
            test_url = f"{base_url}/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            
            # Create a minimal test request
            test_data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            }
            
            response = await self.http_client.post(
                test_url,
                headers=headers,
                json=test_data,
                timeout=5.0,
            )
            
            if response.status_code in [200, 400, 429]:
                # 200 = success, 400 = model not accessible, 429 = rate limit
                # All indicate valid API key
                return TokenStatus.VALID
            elif response.status_code == 401:
                return TokenStatus.INVALID
            else:
                # Other errors might be temporary
                return TokenStatus.REFRESH_NEEDED
                
        except Exception:
            # Network errors or timeouts
            return TokenStatus.REFRESH_NEEDED
    
    async def get_auth_url(self, **kwargs) -> AuthResult:
        """
        Get OAuth authorization URL for Claude.
        
        Note: Claude primarily uses API keys, but OAuth is also supported
        for some use cases.
        
        Returns:
            AuthResult with auth_url for redirect
        """
        # Check if OAuth is configured
        if not self.provider_config or len(self.provider_config) == 0:
            return AuthResult.error_result(
                error="OAuth not configured for Claude",
                error_code="oauth_not_configured",
                provider=self.provider_name,
            )
        
        # Generate PKCE codes
        pkce = self.create_pkce_codes()
        state = self.generate_state()
        
        # Build authorization URL
        auth_params = {
            "client_id": self._get_client_id(),
            "redirect_uri": self._get_redirect_uri(),
            "response_type": "code",
            "scope": "claude",
            "state": state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
        }
        
        # Add optional parameters
        if "login_hint" in kwargs:
            auth_params["login_hint"] = kwargs["login_hint"]
        
        # Build URL
        from urllib.parse import urlencode
        auth_url = f"{self.auth_url}?{urlencode(auth_params)}"
        
        return AuthResult.oauth_redirect(
            auth_url=auth_url,
            state=state,
            code_verifier=pkce.code_verifier,
            provider=self.provider_name,
        )
    
    async def exchange_code(self, code: str, state: str, code_verifier: str) -> AuthResult:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter for CSRF protection
            code_verifier: PKCE code verifier
            
        Returns:
            AuthResult with token data
        """
        try:
            # Exchange code for tokens
            token_data = {
                "client_id": self._get_client_id(),
                "client_secret": self._get_client_secret(),
                "code": code,
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": self._get_redirect_uri(),
            }
            
            response = await self.http_client.post(
                self.token_url,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            if response.status_code != 200:
                error_text = response.text[:200] if response.text else "Unknown error"
                return AuthResult.error_result(
                    error=f"Token exchange failed: {response.status_code} - {error_text}",
                    error_code="token_exchange_failed",
                    provider=self.provider_name,
                )
            
            token_response = response.json()
            
            # Create TokenData from response
            token_data = TokenData(
                access_token=token_response["access_token"],
                refresh_token=token_response.get("refresh_token"),
                token_type=token_response.get("token_type", "Bearer"),
                expires_at=datetime.utcnow() + timedelta(seconds=token_response.get("expires_in", 3600)),
                issued_at=datetime.utcnow(),
                scope=token_response.get("scope"),
                extra_data={
                    "id_token": token_response.get("id_token"),
                },
            )
            
            return AuthResult.success_result(token_data, self.provider_name)
            
        except Exception as e:
            return AuthResult.error_result(
                error=f"Code exchange failed: {str(e)}",
                error_code="code_exchange_failed",
                provider=self.provider_name,
            )
    
    def get_default_headers(self, token_data: TokenData) -> Dict[str, str]:
        """
        Get default headers for Claude API requests.
        
        Args:
            token_data: Token data to use for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = super().get_default_headers(token_data)
        
        # Claude uses x-api-key header instead of Authorization
        if "x-api-key" not in headers:
            headers["x-api-key"] = token_data.access_token
        
        # Add Claude-specific headers
        anthropic_version = "2023-06-01"
        if token_data.extra_data and "anthropic_version" in token_data.extra_data:
            anthropic_version = token_data.extra_data["anthropic_version"]
        
        headers.update({
            "Content-Type": "application/json",
            "anthropic-version": anthropic_version,
            "anthropic-beta": "max-tokens-2024-07-15",  # Enable beta features
        })
        
        return headers
    
    def _get_client_id(self) -> str:
        """Get OAuth client ID from config."""
        # This would come from configuration
        # For now, return a placeholder
        return "claude-client-id"
    
    def _get_client_secret(self) -> str:
        """Get OAuth client secret from config."""
        # This would come from configuration
        # For now, return a placeholder
        return "claude-client-secret"
    
    def _get_redirect_uri(self) -> str:
        """Get OAuth redirect URI."""
        # This would come from configuration
        return "http://localhost:8317/auth/callback/claude"
    
    async def make_authenticated_request(
        self,
        method: str,
        url: str,
        token_data: TokenData,
        **kwargs
    ) -> Any:
        """
        Make an authenticated HTTP request to Claude API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            token_data: Token data for authentication
            **kwargs: Additional arguments for HTTP request
            
        Returns:
            HTTP response
        """
        # Get base URL from token data or use default
        base_url = self.base_url
        if token_data.extra_data and "base_url" in token_data.extra_data:
            base_url = token_data.extra_data["base_url"]
        
        # Ensure URL is absolute
        if not url.startswith("http"):
            url = f"{base_url}{url}"
        
        return await super().make_authenticated_request(method, url, token_data, **kwargs)
    
    async def create_message(
        self,
        token_data: TokenData,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a message using Claude API.
        
        Args:
            token_data: Token data for authentication
            model: Model to use
            messages: List of message dictionaries
            **kwargs: Additional parameters for message creation
            
        Returns:
            Message response
        """
        url = "/v1/messages"
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs,
        }
        
        response = await self.make_authenticated_request(
            method="POST",
            url=url,
            token_data=token_data,
            json=data,
        )
        
        return response.json()
