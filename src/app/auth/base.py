"""
Base authentication provider interface.
Defines the common interface for all authentication providers.
"""

import abc
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

from pydantic import BaseModel


class AuthType(Enum):
    """Authentication type enumeration."""
    OAUTH = "oauth"
    API_KEY = "api_key"
    COOKIE = "cookie"
    TOKEN = "token"


class TokenStatus(Enum):
    """Token status enumeration."""
    VALID = "valid"
    EXPIRED = "expired"
    INVALID = "invalid"
    REFRESH_NEEDED = "refresh_needed"


@dataclass
class TokenData:
    """Token data structure."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    issued_at: Optional[datetime] = None
    scope: Optional[str] = None
    email: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at
    
    def expires_in(self) -> Optional[int]:
        """Get seconds until expiration."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert datetime to ISO format string
        for key in ["expires_at", "issued_at"]:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenData":
        """Create from dictionary."""
        # Convert ISO format strings back to datetime
        for key in ["expires_at", "issued_at"]:
            if key in data and data[key] and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key].replace("Z", "+00:00"))
        return cls(**data)


@dataclass
class AuthResult:
    """Authentication result."""
    success: bool
    token_data: Optional[TokenData] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    provider: Optional[str] = None
    auth_url: Optional[str] = None
    state: Optional[str] = None
    code_verifier: Optional[str] = None
    
    @classmethod
    def success_result(cls, token_data: TokenData, provider: str) -> "AuthResult":
        """Create a successful authentication result."""
        return cls(
            success=True,
            token_data=token_data,
            provider=provider,
        )
    
    @classmethod
    def error_result(cls, error: str, error_code: str, provider: str) -> "AuthResult":
        """Create an error authentication result."""
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            provider=provider,
        )
    
    @classmethod
    def oauth_redirect(cls, auth_url: str, state: str, code_verifier: str, provider: str) -> "AuthResult":
        """Create an OAuth redirect result."""
        return cls(
            success=False,  # Not yet authenticated
            auth_url=auth_url,
            state=state,
            code_verifier=code_verifier,
            provider=provider,
        )


class PKCECodes:
    """PKCE (Proof Key for Code Exchange) codes for OAuth2."""
    
    def __init__(self, code_verifier: Optional[str] = None):
        """Initialize PKCE codes."""
        if code_verifier:
            self.code_verifier = code_verifier
        else:
            self.code_verifier = self.generate_code_verifier()
        
        self.code_challenge = self.generate_code_challenge(self.code_verifier)
    
    @staticmethod
    def generate_code_verifier(length: int = 128) -> str:
        """Generate a code verifier for PKCE."""
        # According to RFC 7636, code verifier must be:
        # - 43-128 characters
        # - A-Z, a-z, 0-9, -, ., _, ~
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def generate_code_challenge(code_verifier: str) -> str:
        """Generate code challenge from code verifier."""
        # SHA256 hash, then base64url encode without padding
        sha256_hash = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(sha256_hash).decode().replace("=", "")
        return code_challenge
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "code_verifier": self.code_verifier,
            "code_challenge": self.code_challenge,
        }


class BaseAuthProvider(abc.ABC):
    """Base class for all authentication providers."""
    
    def __init__(self, provider_name: str, config: Any, http_client: Any):
        """
        Initialize authentication provider.
        
        Args:
            provider_name: Name of the provider (e.g., "gemini", "claude")
            config: Application configuration
            http_client: HTTP client for making requests
        """
        self.provider_name = provider_name
        self.config = config
        self.http_client = http_client
    
    @abc.abstractmethod
    async def authenticate(self, **kwargs) -> AuthResult:
        """
        Authenticate with the provider.
        
        Returns:
            AuthResult with authentication result
        """
        pass
    
    @abc.abstractmethod
    async def refresh_token(self, refresh_token: str) -> TokenData:
        """
        Refresh an expired access token.
        
        Args:
            refresh_token: Refresh token to use
            
        Returns:
            New TokenData
        """
        pass
    
    @abc.abstractmethod
    async def validate_token(self, token_data: TokenData) -> TokenStatus:
        """
        Validate a token.
        
        Args:
            token_data: Token data to validate
            
        Returns:
            TokenStatus indicating token validity
        """
        pass
    
    async def get_auth_url(self, **kwargs) -> AuthResult:
        """
        Get OAuth authorization URL (for OAuth providers).
        
        Returns:
            AuthResult with auth_url for redirect
        """
        raise NotImplementedError("This provider does not support OAuth")
    
    async def exchange_code(self, code: str, state: str, code_verifier: str) -> AuthResult:
        """
        Exchange authorization code for tokens (for OAuth providers).
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter for CSRF protection
            code_verifier: PKCE code verifier
            
        Returns:
            AuthResult with token data
        """
        raise NotImplementedError("This provider does not support OAuth code exchange")
    
    def generate_state(self) -> str:
        """Generate a random state for OAuth."""
        return secrets.token_urlsafe(32)
    
    def create_pkce_codes(self) -> PKCECodes:
        """Create PKCE codes for OAuth."""
        return PKCECodes()
    
    async def logout(self, token_data: TokenData) -> bool:
        """
        Log out (invalidate tokens).
        
        Args:
            token_data: Token data to invalidate
            
        Returns:
            True if logout successful
        """
        # Default implementation just returns True
        # Providers can override to actually invalidate tokens
        return True
    
    def get_provider_config(self) -> List[Any]:
        """Get provider-specific configuration from app config."""
        return self.config.get_provider_config(self.provider_name)
    
    def get_default_headers(self, token_data: TokenData) -> Dict[str, str]:
        """
        Get default headers for API requests.
        
        Args:
            token_data: Token data to use for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = {
            "Authorization": f"{token_data.token_type} {token_data.access_token}",
            "Content-Type": "application/json",
            "User-Agent": f"CLIProxyAPI-Python/1.0.0 ({self.provider_name})",
        }
        
        # Add extra headers from config if available
        provider_config = self.get_provider_config()
        if provider_config and hasattr(provider_config[0], "headers") and provider_config[0].headers:
            headers.update(provider_config[0].headers)
        
        return headers
    
    async def make_authenticated_request(
        self,
        method: str,
        url: str,
        token_data: TokenData,
        **kwargs
    ) -> Any:
        """
        Make an authenticated HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            token_data: Token data for authentication
            **kwargs: Additional arguments for HTTP request
            
        Returns:
            HTTP response
        """
        headers = self.get_default_headers(token_data)
        
        # Merge with any headers provided in kwargs
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        
        # Make the request
        response = await self.http_client.request(
            method=method,
            url=url,
            headers=headers,
            **kwargs
        )
        
        return response
