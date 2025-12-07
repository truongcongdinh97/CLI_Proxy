"""
Gemini authentication provider.
Supports API key authentication for Google Gemini.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import httpx

from .base import BaseAuthProvider, AuthResult, TokenData, TokenStatus, PKCECodes


class GeminiAuth(BaseAuthProvider):
    """Gemini authentication provider."""
    
    def __init__(self, config: Any, http_client: Any):
        """
        Initialize Gemini auth provider.
        
        Args:
            config: Application configuration
            http_client: HTTP client for making requests
        """
        super().__init__("gemini", config, http_client)
        
        # Gemini API endpoints
        self.base_url = "https://generativelanguage.googleapis.com"
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        
        # Get provider-specific configuration
        self.provider_config = self.get_provider_config()
    
    async def authenticate(self, **kwargs) -> AuthResult:
        """
        Authenticate with Gemini using API key.
        
        Supported kwargs:
            api_key: Gemini API key
            key_id: Optional identifier for the API key
            
        Returns:
            AuthResult with authentication result
        """
        api_key = kwargs.get("api_key")
        key_id = kwargs.get("key_id", api_key[:8] if api_key else "unknown")
        
        if not api_key:
            return AuthResult.error_result(
                error="API key is required for Gemini authentication",
                error_code="missing_api_key",
                provider=self.provider_name,
            )
        
        # Validate API key by making a test request
        try:
            # Try to list models to validate the API key
            test_url = f"{self.base_url}/v1beta/models"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
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
                    # API keys don't expire
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
        
        Note: Gemini API keys don't expire, so this just returns the same token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New TokenData
        """
        # Gemini API keys don't expire, so we just return the same token
        # In a real implementation, we might want to re-validate the API key
        return TokenData(
            access_token=refresh_token,  # In this case, refresh_token is the API key
            token_type="Bearer",
            issued_at=datetime.utcnow(),
            expires_at=None,
        )
    
    async def validate_token(self, token_data: TokenData) -> TokenStatus:
        """
        Validate a Gemini API key.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            TokenStatus indicating token validity
        """
        api_key = token_data.access_token
        
        if not api_key:
            return TokenStatus.INVALID
        
        # Check if token is expired (Gemini API keys don't expire)
        if token_data.is_expired():
            return TokenStatus.EXPIRED
        
        # Validate API key by making a test request
        try:
            test_url = f"{self.base_url}/v1beta/models"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            
            response = await self.http_client.get(
                test_url,
                headers=headers,
                timeout=5.0,
            )
            
            if response.status_code == 200:
                return TokenStatus.VALID
            elif response.status_code == 403:
                return TokenStatus.INVALID
            else:
                # Other errors might be temporary
                return TokenStatus.REFRESH_NEEDED
                
        except Exception:
            # Network errors or timeouts
            return TokenStatus.REFRESH_NEEDED
    
    async def get_auth_url(self, **kwargs) -> AuthResult:
        """
        Get OAuth authorization URL for Gemini.
        
        Note: Gemini primarily uses API keys, but OAuth is also supported
        for some use cases.
        
        Returns:
            AuthResult with auth_url for redirect
        """
        # Check if OAuth is configured
        if not self.provider_config or len(self.provider_config) == 0:
            return AuthResult.error_result(
                error="OAuth not configured for Gemini",
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
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "state": state,
            "code_challenge": pkce.code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
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
        Get default headers for Gemini API requests.
        
        Args:
            token_data: Token data to use for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = super().get_default_headers(token_data)
        
        # Gemini uses x-goog-api-key header for API keys
        if token_data.token_type == "Bearer" and "x-goog-api-key" not in headers:
            headers["x-goog-api-key"] = token_data.access_token
        
        # Add Gemini-specific headers
        headers.update({
            "Content-Type": "application/json",
        })
        
        return headers
    
    def _get_client_id(self) -> str:
        """Get OAuth client ID from config."""
        # This would come from configuration
        # For now, return a placeholder
        return "gemini-client-id"
    
    def _get_client_secret(self) -> str:
        """Get OAuth client secret from config."""
        # This would come from configuration
        # For now, return a placeholder
        return "gemini-client-secret"
    
    def _get_redirect_uri(self) -> str:
        """Get OAuth redirect URI."""
        # This would come from configuration
        return "http://localhost:8317/auth/callback/gemini"
    
    async def make_authenticated_request(
        self,
        method: str,
        url: str,
        token_data: TokenData,
        **kwargs
    ) -> Any:
        """
        Make an authenticated HTTP request to Gemini API.
        
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
