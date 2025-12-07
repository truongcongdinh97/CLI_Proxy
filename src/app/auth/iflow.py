"""
iFlow authentication provider.
Supports cookie-based authentication for iFlow.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import httpx

from .base import BaseAuthProvider, AuthResult, TokenData, TokenStatus


class iFlowAuth(BaseAuthProvider):
    """iFlow authentication provider."""
    
    def __init__(self, config: Any, http_client: Any):
        """
        Initialize iFlow auth provider.
        
        Args:
            config: Application configuration
            http_client: HTTP client for making requests
        """
        super().__init__("iflow", config, http_client)
        
        # iFlow API endpoints
        self.base_url = "https://iflow.team"
        
        # Get provider-specific configuration
        self.provider_config = self.get_provider_config()
    
    async def authenticate(self, **kwargs) -> AuthResult:
        """
        Authenticate with iFlow using cookies.
        
        Supported kwargs:
            cookies: Dictionary of cookies
            cookie_string: Cookie string
            key_id: Optional identifier for the session
            
        Returns:
            AuthResult with authentication result
        """
        cookies = kwargs.get("cookies")
        cookie_string = kwargs.get("cookie_string")
        key_id = kwargs.get("key_id", "iflow_session")
        
        # Parse cookies from string if provided
        if cookie_string and not cookies:
            cookies = self._parse_cookie_string(cookie_string)
        
        if not cookies:
            return AuthResult.error_result(
                error="Cookies are required for iFlow authentication",
                error_code="missing_cookies",
                provider=self.provider_name,
            )
        
        # Validate cookies by making a test request
        try:
            # Try to access user info to validate cookies
            test_url = f"{self.base_url}/api/user/info"
            
            # Create cookie header
            cookie_header = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            
            headers = {
                "Cookie": cookie_header,
                "Content-Type": "application/json",
            }
            
            response = await self.http_client.get(
                test_url,
                headers=headers,
                timeout=10.0,
            )
            
            if response.status_code == 200:
                # Cookies are valid
                token_data = TokenData(
                    access_token=json.dumps(cookies),  # Store cookies as JSON string
                    token_type="Cookie",
                    issued_at=datetime.utcnow(),
                    # Cookie sessions typically expire
                    expires_at=datetime.utcnow() + timedelta(days=7),
                    extra_data={
                        "key_id": key_id,
                        "validation_time": datetime.utcnow().isoformat(),
                        "cookies": cookies,
                    },
                )
                
                return AuthResult.success_result(token_data, self.provider_name)
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                return AuthResult.error_result(
                    error=f"Cookie validation failed: {response.status_code} - {error_text}",
                    error_code="cookie_validation_failed",
                    provider=self.provider_name,
                )
                
        except httpx.TimeoutException:
            return AuthResult.error_result(
                error="Connection timeout while validating cookies",
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
        Refresh an expired session.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New TokenData
        """
        # For iFlow, we need to re-authenticate with cookies
        # This is a simplified implementation
        try:
            # Parse cookies from refresh_token (which is JSON string)
            cookies = json.loads(refresh_token)
            
            # Create new token data
            return TokenData(
                access_token=refresh_token,
                token_type="Cookie",
                issued_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=7),
                extra_data={
                    "cookies": cookies,
                    "refreshed_at": datetime.utcnow().isoformat(),
                },
            )
        except:
            # If we can't parse, return a basic token
            return TokenData(
                access_token=refresh_token,
                token_type="Cookie",
                issued_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=1),
            )
    
    async def validate_token(self, token_data: TokenData) -> TokenStatus:
        """
        Validate iFlow cookies.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            TokenStatus indicating token validity
        """
        if not token_data.access_token:
            return TokenStatus.INVALID
        
        # Check if token is expired
        if token_data.is_expired():
            return TokenStatus.EXPIRED
        
        # Try to parse cookies
        try:
            cookies = json.loads(token_data.access_token)
            
            # Validate cookies by making a test request
            test_url = f"{self.base_url}/api/user/info"
            cookie_header = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            
            headers = {
                "Cookie": cookie_header,
                "Content-Type": "application/json",
            }
            
            response = await self.http_client.get(
                test_url,
                headers=headers,
                timeout=5.0,
            )
            
            if response.status_code == 200:
                return TokenStatus.VALID
            elif response.status_code in [401, 403]:
                return TokenStatus.INVALID
            else:
                return TokenStatus.REFRESH_NEEDED
                
        except Exception:
            return TokenStatus.REFRESH_NEEDED
    
    def get_default_headers(self, token_data: TokenData) -> Dict[str, str]:
        """
        Get default headers for iFlow API requests.
        
        Args:
            token_data: Token data to use for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = super().get_default_headers(token_data)
        
        # Add iFlow-specific headers
        headers.update({
            "Content-Type": "application/json",
        })
        
        # Add cookies if available
        if token_data.extra_data and "cookies" in token_data.extra_data:
            cookies = token_data.extra_data["cookies"]
            cookie_header = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            headers["Cookie"] = cookie_header
        
        return headers
    
    def _parse_cookie_string(self, cookie_string: str) -> Dict[str, str]:
        """
        Parse cookie string into dictionary.
        
        Args:
            cookie_string: Cookie string
            
        Returns:
            Dictionary of cookies
        """
        cookies = {}
        for cookie in cookie_string.split(";"):
            cookie = cookie.strip()
            if "=" in cookie:
                key, value = cookie.split("=", 1)
                cookies[key.strip()] = value.strip()
        return cookies
    
    async def make_authenticated_request(
        self,
        method: str,
        url: str,
        token_data: TokenData,
        **kwargs
    ) -> Any:
        """
        Make an authenticated HTTP request to iFlow API.
        
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
