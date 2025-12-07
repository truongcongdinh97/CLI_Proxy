"""
Base storage interface for token management.
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from ..auth.base import TokenData


class StoreError(Exception):
    """Base exception for store errors."""
    pass


class TokenNotFoundError(StoreError):
    """Token not found in store."""
    pass


class TokenExpiredError(StoreError):
    """Token has expired."""
    pass


class BaseStore(abc.ABC):
    """Base class for token storage implementations."""
    
    def __init__(self, config: Any):
        """
        Initialize store.
        
        Args:
            config: Application configuration
        """
        self.config = config
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the store."""
        pass
    
    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the store."""
        pass
    
    @abc.abstractmethod
    async def save_token(
        self,
        provider: str,
        key_id: str,
        token_data: TokenData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save token to store.
        
        Args:
            provider: Provider name (e.g., "gemini", "claude")
            key_id: Unique identifier for the token (e.g., API key, user ID)
            token_data: Token data to save
            metadata: Additional metadata to store with the token
        """
        pass
    
    @abc.abstractmethod
    async def get_token(
        self,
        provider: str,
        key_id: str,
    ) -> Optional[TokenData]:
        """
        Get token from store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            TokenData if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    async def delete_token(
        self,
        provider: str,
        key_id: str,
    ) -> bool:
        """
        Delete token from store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            True if token was deleted, False if not found
        """
        pass
    
    @abc.abstractmethod
    async def list_tokens(
        self,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all tokens in store.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            List of token metadata dictionaries
        """
        pass
    
    @abc.abstractmethod
    async def update_token_metadata(
        self,
        provider: str,
        key_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update token metadata.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            metadata: Metadata to update
        """
        pass
    
    async def get_valid_token(
        self,
        provider: str,
        key_id: str,
        min_expiry: int = 60,
    ) -> Optional[TokenData]:
        """
        Get a valid token (not expired or about to expire).
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            min_expiry: Minimum seconds until expiration
            
        Returns:
            Valid TokenData, or None if not found or expired
        """
        token_data = await self.get_token(provider, key_id)
        if not token_data:
            return None
        
        # Check if token is expired
        if token_data.is_expired():
            await self.delete_token(provider, key_id)
            return None
        
        # Check if token will expire soon
        expires_in = token_data.expires_in()
        if expires_in is not None and expires_in < min_expiry:
            # Token will expire soon, mark for refresh
            return None
        
        return token_data
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from store.
        
        Returns:
            Number of tokens cleaned up
        """
        count = 0
        tokens = await self.list_tokens()
        
        for token_info in tokens:
            provider = token_info.get("provider")
            key_id = token_info.get("key_id")
            
            if not provider or not key_id:
                continue
            
            token_data = await self.get_token(provider, key_id)
            if token_data and token_data.is_expired():
                await self.delete_token(provider, key_id)
                count += 1
        
        return count
    
    def _serialize_token(self, token_data: TokenData) -> Dict[str, Any]:
        """Serialize token data for storage."""
        serialized = asdict(token_data)
        
        # Convert datetime to ISO format string
        for key in ["expires_at", "issued_at"]:
            if serialized[key] and isinstance(serialized[key], datetime):
                serialized[key] = serialized[key].isoformat()
        
        return serialized
    
    def _deserialize_token(self, data: Dict[str, Any]) -> TokenData:
        """Deserialize token data from storage."""
        # Convert ISO format strings back to datetime
        for key in ["expires_at", "issued_at"]:
            if key in data and data[key] and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key].replace("Z", "+00:00"))
        
        return TokenData(**data)
