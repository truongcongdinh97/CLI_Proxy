"""
Authentication manager for CLI Proxy API.
Manages authentication providers and token storage.
"""

from typing import Dict, Optional, Any
from .base import BaseAuthProvider, AuthResult, TokenData, TokenStatus
from .gemini import GeminiAuth
from .openai import OpenAIAuth
from .claude import ClaudeAuth
from .qwen import QwenAuth
from .iflow import iFlowAuth
from ..stores.manager import StoreManager


class AuthManager:
    """Manager for authentication providers."""
    
    def __init__(self, config: Any, store_manager: StoreManager, http_client: Any):
        """
        Initialize authentication manager.
        
        Args:
            config: Application configuration
            store_manager: Store manager for token storage
            http_client: HTTP client for authentication requests
        """
        self.config = config
        self.store_manager = store_manager
        self.http_client = http_client
        self.auth_providers: Dict[str, BaseAuthProvider] = {}
        self._initialize_auth_providers()
    
    def _initialize_auth_providers(self):
        """Initialize authentication providers."""
        # Initialize all supported auth providers
        self.auth_providers["gemini"] = GeminiAuth(self.config, self.http_client)
        self.auth_providers["openai"] = OpenAIAuth(self.config, self.http_client)
        self.auth_providers["claude"] = ClaudeAuth(self.config, self.http_client)
        self.auth_providers["qwen"] = QwenAuth(self.config, self.http_client)
        self.auth_providers["iflow"] = iFlowAuth(self.config, self.http_client)
    
    def get_auth_provider(self, provider_name: str) -> Optional[BaseAuthProvider]:
        """
        Get authentication provider by name.
        
        Args:
            provider_name: Provider name
            
        Returns:
            Authentication provider, or None if not found
        """
        return self.auth_providers.get(provider_name.lower())
    
    async def authenticate(
        self,
        provider_name: str,
        key_id: str,
        **kwargs
    ) -> AuthResult:
        """
        Authenticate with a provider.
        
        Args:
            provider_name: Provider name
            key_id: Unique identifier for the authentication
            **kwargs: Authentication parameters
            
        Returns:
            Authentication result
        """
        auth_provider = self.get_auth_provider(provider_name)
        if not auth_provider:
            return AuthResult.error_result(
                error=f"Authentication provider not found: {provider_name}",
                error_code="provider_not_found",
                provider=provider_name,
            )
        
        # Authenticate with provider
        auth_result = await auth_provider.authenticate(**kwargs)
        
        # Save token if authentication was successful
        if auth_result.success and auth_result.token_data:
            await self.store_manager.save_token(
                provider=provider_name,
                key_id=key_id,
                token_data=auth_result.token_data,
                metadata={
                    "auth_method": kwargs.get("auth_method", "api_key"),
                    "key_id": key_id,
                },
            )
        
        return auth_result
    
    async def get_token(
        self,
        provider_name: str,
        key_id: str,
        validate: bool = True,
    ) -> Optional[TokenData]:
        """
        Get token for a provider.
        
        Args:
            provider_name: Provider name
            key_id: Unique identifier for the token
            validate: Whether to validate the token
            
        Returns:
            Token data, or None if not found or invalid
        """
        # Get token from store
        token_data = await self.store_manager.get_token(provider_name, key_id)
        if not token_data:
            return None
        
        # Validate token if requested
        if validate:
            auth_provider = self.get_auth_provider(provider_name)
            if auth_provider:
                token_status = await auth_provider.validate_token(token_data)
                if token_status != TokenStatus.VALID:
                    # Token is invalid or expired
                    if token_status == TokenStatus.EXPIRED:
                        # Try to refresh if possible
                        if token_data.refresh_token:
                            try:
                                new_token_data = await auth_provider.refresh_token(
                                    token_data.refresh_token
                                )
                                await self.store_manager.save_token(
                                    provider=provider_name,
                                    key_id=key_id,
                                    token_data=new_token_data,
                                )
                                return new_token_data
                            except Exception:
                                # Refresh failed, delete token
                                await self.store_manager.delete_token(provider_name, key_id)
                                return None
                        else:
                            # No refresh token, delete expired token
                            await self.store_manager.delete_token(provider_name, key_id)
                            return None
                    else:
                        # Token is invalid for other reasons
                        await self.store_manager.delete_token(provider_name, key_id)
                        return None
        
        return token_data
    
    async def validate_token(
        self,
        provider_name: str,
        key_id: str,
    ) -> TokenStatus:
        """
        Validate a token.
        
        Args:
            provider_name: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            Token status
        """
        token_data = await self.get_token(provider_name, key_id, validate=False)
        if not token_data:
            return TokenStatus.INVALID
        
        auth_provider = self.get_auth_provider(provider_name)
        if not auth_provider:
            return TokenStatus.INVALID
        
        return await auth_provider.validate_token(token_data)
    
    async def delete_token(
        self,
        provider_name: str,
        key_id: str,
    ) -> bool:
        """
        Delete a token.
        
        Args:
            provider_name: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            True if token was deleted
        """
        return await self.store_manager.delete_token(provider_name, key_id)
    
    async def list_tokens(
        self,
        provider_name: Optional[str] = None,
    ) -> list:
        """
        List all tokens.
        
        Args:
            provider_name: Optional provider filter
            
        Returns:
            List of token metadata
        """
        return await self.store_manager.list_tokens(provider_name)
    
    async def get_auth_url(
        self,
        provider_name: str,
        **kwargs
    ) -> AuthResult:
        """
        Get OAuth authorization URL.
        
        Args:
            provider_name: Provider name
            **kwargs: Additional parameters
            
        Returns:
            Authentication result with auth URL
        """
        auth_provider = self.get_auth_provider(provider_name)
        if not auth_provider:
            return AuthResult.error_result(
                error=f"Authentication provider not found: {provider_name}",
                error_code="provider_not_found",
                provider=provider_name,
            )
        
        return await auth_provider.get_auth_url(**kwargs)
    
    async def exchange_code(
        self,
        provider_name: str,
        code: str,
        state: str,
        code_verifier: str,
        key_id: str,
    ) -> AuthResult:
        """
        Exchange authorization code for tokens.
        
        Args:
            provider_name: Provider name
            code: Authorization code
            state: State parameter
            code_verifier: PKCE code verifier
            key_id: Unique identifier for the token
            
        Returns:
            Authentication result
        """
        auth_provider = self.get_auth_provider(provider_name)
        if not auth_provider:
            return AuthResult.error_result(
                error=f"Authentication provider not found: {provider_name}",
                error_code="provider_not_found",
                provider=provider_name,
            )
        
        # Exchange code for tokens
        auth_result = await auth_provider.exchange_code(code, state, code_verifier)
        
        # Save token if exchange was successful
        if auth_result.success and auth_result.token_data:
            await self.store_manager.save_token(
                provider=provider_name,
                key_id=key_id,
                token_data=auth_result.token_data,
                metadata={
                    "auth_method": "oauth",
                    "key_id": key_id,
                },
            )
        
        return auth_result
    
    async def logout(
        self,
        provider_name: str,
        key_id: str,
    ) -> bool:
        """
        Log out (invalidate tokens).
        
        Args:
            provider_name: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            True if logout successful
        """
        # Get token data
        token_data = await self.get_token(provider_name, key_id, validate=False)
        if not token_data:
            return False
        
        # Call provider logout if supported
        auth_provider = self.get_auth_provider(provider_name)
        if auth_provider:
            try:
                await auth_provider.logout(token_data)
            except Exception:
                # Logout might fail, but we still delete the token locally
                pass
        
        # Delete token from store
        return await self.delete_token(provider_name, key_id)
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens.
        
        Returns:
            Number of tokens cleaned up
        """
        return await self.store_manager.cleanup_expired_tokens()
