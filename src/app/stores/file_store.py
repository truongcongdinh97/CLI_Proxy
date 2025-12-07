"""
File-based token storage implementation.
Stores tokens as JSON files in a directory structure.
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiofiles
import aiofiles.os

from .base import BaseStore, StoreError, TokenNotFoundError
from ..auth.base import TokenData


class FileStore(BaseStore):
    """File-based token storage."""
    
    def __init__(self, config: Any):
        """
        Initialize file store.
        
        Args:
            config: Application configuration
        """
        super().__init__(config)
        self.data_dir = Path(getattr(config, "auth_dir", "~/.cli-proxy-api")).expanduser()
        self.tokens_dir = self.data_dir / "tokens"
        self.metadata_dir = self.data_dir / "metadata"
    
    async def initialize(self) -> None:
        """Initialize the file store."""
        # Create directories if they don't exist
        await aiofiles.os.makedirs(self.tokens_dir, exist_ok=True)
        await aiofiles.os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Create provider subdirectories
        providers = ["gemini", "claude", "codex", "openai", "qwen", "iflow", "vertex"]
        for provider in providers:
            provider_dir = self.tokens_dir / provider
            await aiofiles.os.makedirs(provider_dir, exist_ok=True)
            
            metadata_dir = self.metadata_dir / provider
            await aiofiles.os.makedirs(metadata_dir, exist_ok=True)
    
    async def shutdown(self) -> None:
        """Shutdown the file store."""
        # Nothing to do for file store
        pass
    
    def _get_token_path(self, provider: str, key_id: str) -> Path:
        """Get path for token file."""
        # Sanitize key_id for filename
        safe_key_id = "".join(c for c in key_id if c.isalnum() or c in "._-")
        return self.tokens_dir / provider / f"{safe_key_id}.json"
    
    def _get_metadata_path(self, provider: str, key_id: str) -> Path:
        """Get path for metadata file."""
        # Sanitize key_id for filename
        safe_key_id = "".join(c for c in key_id if c.isalnum() or c in "._-")
        return self.metadata_dir / provider / f"{safe_key_id}.json"
    
    async def save_token(
        self,
        provider: str,
        key_id: str,
        token_data: TokenData,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save token to file store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            token_data: Token data to save
            metadata: Additional metadata to store with the token
        """
        token_path = self._get_token_path(provider, key_id)
        metadata_path = self._get_metadata_path(provider, key_id)
        
        # Serialize token data
        serialized_token = self._serialize_token(token_data)
        
        # Save token data
        async with aiofiles.open(token_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(serialized_token, indent=2))
        
        # Save metadata if provided
        if metadata is not None:
            metadata_data = {
                "provider": provider,
                "key_id": key_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                **metadata,
            }
            
            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(metadata_data, indent=2))
    
    async def get_token(
        self,
        provider: str,
        key_id: str,
    ) -> Optional[TokenData]:
        """
        Get token from file store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            TokenData if found, None otherwise
        """
        token_path = self._get_token_path(provider, key_id)
        
        if not await aiofiles.os.path.exists(token_path):
            return None
        
        try:
            async with aiofiles.open(token_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
            
            return self._deserialize_token(data)
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Corrupted file, delete it
            await aiofiles.os.remove(token_path)
            raise StoreError(f"Corrupted token file for {provider}/{key_id}: {e}")
    
    async def delete_token(
        self,
        provider: str,
        key_id: str,
    ) -> bool:
        """
        Delete token from file store.
        
        Args:
            provider: Provider name
            key_id: Unique identifier for the token
            
        Returns:
            True if token was deleted, False if not found
        """
        token_path = self._get_token_path(provider, key_id)
        metadata_path = self._get_metadata_path(provider, key_id)
        
        deleted = False
        
        # Delete token file
        if await aiofiles.os.path.exists(token_path):
            await aiofiles.os.remove(token_path)
            deleted = True
        
        # Delete metadata file
        if await aiofiles.os.path.exists(metadata_path):
            await aiofiles.os.remove(metadata_path)
        
        return deleted
    
    async def list_tokens(
        self,
        provider: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all tokens in file store.
        
        Args:
            provider: Optional provider filter
            
        Returns:
            List of token metadata dictionaries
        """
        tokens = []
        
        if provider:
            providers = [provider]
        else:
            # List all provider directories
            providers = []
            if self.tokens_dir.exists():
                for item in self.tokens_dir.iterdir():
                    if item.is_dir():
                        providers.append(item.name)
        
        for prov in providers:
            provider_dir = self.tokens_dir / prov
            if not provider_dir.exists():
                continue
            
            # List all token files in provider directory
            for token_file in provider_dir.glob("*.json"):
                key_id = token_file.stem
                
                # Try to load metadata
                metadata_path = self._get_metadata_path(prov, key_id)
                metadata = {}
                
                if await aiofiles.os.path.exists(metadata_path):
                    try:
                        async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                            content = await f.read()
                            metadata = json.loads(content)
                    except:
                        # If metadata is corrupted, use basic info
                        metadata = {"provider": prov, "key_id": key_id}
                else:
                    metadata = {"provider": prov, "key_id": key_id}
                
                # Add token info if available
                try:
                    token_data = await self.get_token(prov, key_id)
                    if token_data:
                        metadata["expires_at"] = token_data.expires_at.isoformat() if token_data.expires_at else None
                        metadata["expires_in"] = token_data.expires_in()
                        metadata["is_expired"] = token_data.is_expired()
                except:
                    pass
                
                tokens.append(metadata)
        
        return tokens
    
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
        metadata_path = self._get_metadata_path(provider, key_id)
        
        # Load existing metadata
        existing_metadata = {}
        if await aiofiles.os.path.exists(metadata_path):
            try:
                async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    existing_metadata = json.loads(content)
            except:
                # Start with empty metadata if file is corrupted
                existing_metadata = {"provider": provider, "key_id": key_id}
        else:
            existing_metadata = {"provider": provider, "key_id": key_id}
        
        # Update metadata
        existing_metadata.update(metadata)
        existing_metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Ensure created_at exists
        if "created_at" not in existing_metadata:
            existing_metadata["created_at"] = datetime.utcnow().isoformat()
        
        # Save updated metadata
        async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(existing_metadata, indent=2))
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from file store.
        
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
    
    async def backup(self, backup_dir: Path) -> None:
        """
        Backup token store to another directory.
        
        Args:
            backup_dir: Directory to backup to
        """
        await aiofiles.os.makedirs(backup_dir, exist_ok=True)
        
        # Copy tokens directory
        if self.tokens_dir.exists():
            tokens_backup = backup_dir / "tokens"
            await self._copy_directory(self.tokens_dir, tokens_backup)
        
        # Copy metadata directory
        if self.metadata_dir.exists():
            metadata_backup = backup_dir / "metadata"
            await self._copy_directory(self.metadata_dir, metadata_backup)
    
    async def _copy_directory(self, src: Path, dst: Path) -> None:
        """Copy directory recursively."""
        await aiofiles.os.makedirs(dst, exist_ok=True)
        
        for item in src.iterdir():
            if item.is_file():
                # Copy file
                dst_file = dst / item.name
                async with aiofiles.open(item, "rb") as src_file:
                    async with aiofiles.open(dst_file, "wb") as dst_file_obj:
                        content = await src_file.read()
                        await dst_file_obj.write(content)
            elif item.is_dir():
                # Recursively copy directory
                await self._copy_directory(item, dst / item.name)
