"""
Store manager for CLI Proxy API.
Manages multiple storage backends and provides a unified interface.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base import BaseStore, StoreError
from .file_store import FileStore
from ..auth.base import TokenData


class StoreManager:
    """Manager for token storage backends."""
    
    def __init__(self, config: Any):
        """
        Initialize store manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.stores: Dict[str, BaseStore] = {}
        self.default_store: Optional[BaseStore] = None
    
    async def initialize(self) -> None:
        """Initialize all stores."""
        # Initialize file store as default
        file_store = FileStore(self.config)
        await file_store.initialize()
        
        self.stores["file"] = file_store
        self.default_store = file_store
        
        # Could add other stores here (e.g., Redis, PostgreSQL)
        # redis_store = RedisStore(self.config)
        # await redis_store.initialize()
        # self.stores["redis"] = redis_store
    
    async def shutdown(self) -> None:
        """Shutdown all stores."""
        for store in self.stores.values():
            await store.shutdown()
        
        self.stores.clear()
        self.default_store = None
    
    def get_store(self, store_type: str = "file") -> BaseStore:
        """
        Get store by type.
        
        Args:
            store_type: Store type ("file", "redis", etc.)
            
        Returns:
            Store instance
            
        Raises:
            StoreError: If store type not found
        """
        if store_type not in self.stores:
            raise StoreError(f"Store type not found: {store_type}")
        
        return self.stores[store_type]
    
    async def save_token(
        self,
        provider: str,
        key_id: str,
        token_data: TokenData,
        metadata: Optional[Dict[str, Any]] = None,
        store_type: str = "file",
    ) -> None:
        """
        Save token to store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            token_data: Token data to save
            metadata: Additional metadata
            store_type: Store type to use
        """
        store = self.get_store(store_type)
        await store.save_token(provider, key_id, token_data, metadata)
    
    async def get_token(
        self,
        provider: str,
        key_id: str,
        store_type: str = "file",
    ) -> Optional[TokenData]:
        """
        Get token from store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            store_type: Store type to use
            
        Returns:
            TokenData if found, None otherwise
        """
        store = self.get_store(store_type)
        return await store.get_token(provider, key_id)
    
    async def get_valid_token(
        self,
        provider: str,
        key_id: str,
        min_expiry: int = 60,
        store_type: str = "file",
    ) -> Optional[TokenData]:
        """
        Get a valid token (not expired or about to expire).
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            min_expiry: Minimum seconds until expiration
            store_type: Store type to use
            
        Returns:
            Valid TokenData, or None if not found or expired
        """
        store = self.get_store(store_type)
        return await store.get_valid_token(provider, key_id, min_expiry)
    
    async def delete_token(
        self,
        provider: str,
        key_id: str,
        store_type: str = "file",
    ) -> bool:
        """
        Delete token from store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            store_type: Store type to use
            
        Returns:
            True if token was deleted, False if not found
        """
        store = self.get_store(store_type)
        return await store.delete_token(provider, key_id)
    
    async def list_tokens(
        self,
        provider: Optional[str] = None,
        store_type: str = "file",
    ) -> List[Dict[str, Any]]:
        """
        List all tokens in store.
        
        Args:
            provider: Optional provider filter
            store_type: Store type to use
            
        Returns:
            List of token metadata dictionaries
        """
        store = self.get_store(store_type)
        return await store.list_tokens(provider)
    
    async def update_token_metadata(
        self,
        provider: str,
        key_id: str,
        metadata: Dict[str, Any],
        store_type: str = "file",
    ) -> None:
        """
        Update token metadata.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            metadata: Metadata to update
            store_type: Store type to use
        """
        store = self.get_store(store_type)
        await store.update_token_metadata(provider, key_id, metadata)
    
    async def cleanup_expired_tokens(self, store_type: str = "file") -> int:
        """
        Clean up expired tokens from store.
        
        Args:
            store_type: Store type to use
            
        Returns:
            Number of tokens cleaned up
        """
        store = self.get_store(store_type)
        return await store.cleanup_expired_tokens()
    
    async def sync_tokens(
        self,
        from_store_type: str,
        to_store_type: str,
        provider: Optional[str] = None,
    ) -> int:
        """
        Sync tokens from one store to another.
        
        Args:
            from_store_type: Source store type
            to_store_type: Destination store type
            provider: Optional provider filter
            
        Returns:
            Number of tokens synced
        """
        from_store = self.get_store(from_store_type)
        to_store = self.get_store(to_store_type)
        
        tokens = await from_store.list_tokens(provider)
        synced_count = 0
        
        for token_info in tokens:
            provider_name = token_info.get("provider")
            key_id = token_info.get("key_id")
            
            if not provider_name or not key_id:
                continue
            
            # Get token from source store
            token_data = await from_store.get_token(provider_name, key_id)
            if not token_data:
                continue
            
            # Save to destination store
            await to_store.save_token(provider_name, key_id, token_data, token_info)
            synced_count += 1
        
        return synced_count
    
    async def backup_store(self, store_type: str, backup_dir: str) -> None:
        """
        Backup store to directory.
        
        Args:
            store_type: Store type to backup
            backup_dir: Directory to backup to
        """
        store = self.get_store(store_type)
        
        if hasattr(store, "backup"):
            from pathlib import Path
            backup_path = Path(backup_dir)
            await store.backup(backup_path)
        else:
            raise StoreError(f"Store type {store_type} does not support backup")
    
    async def get_token_stats(self, store_type: str = "file") -> Dict[str, Any]:
        """
        Get token statistics.
        
        Args:
            store_type: Store type to use
            
        Returns:
            Dictionary with token statistics
        """
        store = self.get_store(store_type)
        tokens = await store.list_tokens()
        
        stats = {
            "total_tokens": len(tokens),
            "providers": {},
            "expired_tokens": 0,
            "expiring_soon_tokens": 0,
            "valid_tokens": 0,
        }
        
        now = datetime.utcnow()
        soon_threshold = now + timedelta(minutes=5)
        
        for token_info in tokens:
            provider = token_info.get("provider", "unknown")
            
            # Update provider count
            if provider not in stats["providers"]:
                stats["providers"][provider] = 0
            stats["providers"][provider] += 1
            
            # Check token status
            expires_at = token_info.get("expires_at")
            if expires_at:
                try:
                    if isinstance(expires_at, str):
                        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    
                    if expires_at < now:
                        stats["expired_tokens"] += 1
                    elif expires_at < soon_threshold:
                        stats["expiring_soon_tokens"] += 1
                    else:
                        stats["valid_tokens"] += 1
                except:
                    # If we can't parse the date, count as valid
                    stats["valid_tokens"] += 1
            else:
                # Token without expiration is considered valid
                stats["valid_tokens"] += 1
        
        return stats
