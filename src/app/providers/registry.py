"""
Provider registry for managing AI service providers.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import random

from .base import BaseProvider, ProviderConfig, ProviderType, ProviderStatus
from ..auth.manager import AuthManager
from ..utils.http_client import create_http_client_for_provider


class ProviderRegistry:
    """Registry for managing AI service providers."""
    
    def __init__(self, config: Any, auth_manager: AuthManager, http_client: Any):
        """
        Initialize provider registry.
        
        Args:
            config: Application configuration
            auth_manager: Authentication manager
            http_client: HTTP client
        """
        self.config = config
        self.auth_manager = auth_manager
        self.http_client = http_client
        self.providers: Dict[str, BaseProvider] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
    
    async def initialize(self) -> None:
        """Initialize all providers."""
        # Load provider configurations from app config
        await self._load_provider_configs()
        
        # Initialize providers
        for provider_name, provider_config in self.provider_configs.items():
            if provider_config.enabled:
                await self._initialize_provider(provider_name, provider_config)
    
    async def shutdown(self) -> None:
        """Shutdown all providers."""
        for provider in self.providers.values():
            await provider.shutdown()
        
        self.providers.clear()
        self.provider_configs.clear()
    
    async def _load_provider_configs(self) -> None:
        """Load provider configurations from app config."""
        # Load Gemini API keys from config
        gemini_keys = getattr(self.config, 'gemini_api_key', [])
        for i, key_config in enumerate(gemini_keys):
            api_key = getattr(key_config, 'api_key', None)
            base_url = getattr(key_config, 'base_url', 'https://generativelanguage.googleapis.com')
            
            if api_key:
                config = ProviderConfig(
                    name=f"gemini-{i}",
                    provider_type=ProviderType.GEMINI,
                    base_url=base_url,
                    api_key=api_key,
                    priority=1,
                    enabled=True,
                )
                self.provider_configs[config.name] = config
        
        # Load Claude API keys from config
        claude_keys = getattr(self.config, 'claude_api_key', [])
        for i, key_config in enumerate(claude_keys):
            api_key = getattr(key_config, 'api_key', None)
            base_url = getattr(key_config, 'base_url', 'https://api.anthropic.com')
            
            if api_key:
                config = ProviderConfig(
                    name=f"claude-{i}",
                    provider_type=ProviderType.CLAUDE,
                    base_url=base_url,
                    api_key=api_key,
                    priority=2,
                    enabled=True,
                )
                self.provider_configs[config.name] = config
        
        # Load OpenAI/Codex API keys from config
        codex_keys = getattr(self.config, 'codex_api_key', [])
        for i, key_config in enumerate(codex_keys):
            api_key = getattr(key_config, 'api_key', None)
            base_url = getattr(key_config, 'base_url', 'https://api.openai.com')
            
            if api_key:
                config = ProviderConfig(
                    name=f"openai-{i}",
                    provider_type=ProviderType.OPENAI,
                    base_url=base_url,
                    api_key=api_key,
                    priority=3,
                    enabled=True,
                )
                self.provider_configs[config.name] = config
    
    async def _initialize_provider(self, name: str, config: ProviderConfig) -> None:
        """
        Initialize a provider.
        
        Args:
            name: Provider name
            config: Provider configuration
        """
        try:
            # Create provider-specific HTTP client
            provider_http_client = create_http_client_for_provider(
                self.config,
                config,
                base_url=config.base_url,
            )
            
            # Import and create provider based on type
            if config.provider_type == ProviderType.GEMINI:
                from .gemini_provider import GeminiProvider
                provider = GeminiProvider(config, self.auth_manager, provider_http_client)
            elif config.provider_type == ProviderType.OPENAI:
                from .openai_provider import OpenAIProvider
                provider = OpenAIProvider(config, self.auth_manager, provider_http_client)
            elif config.provider_type == ProviderType.CLAUDE:
                from .claude_provider import ClaudeProvider
                provider = ClaudeProvider(config, self.auth_manager, provider_http_client)
            else:
                # Skip unsupported provider types
                return
            
            # Initialize provider
            await provider.initialize()
            self.providers[name] = provider
            
        except Exception as e:
            print(f"Failed to initialize provider {name}: {e}")
    
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """
        Get provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance, or None if not found
        """
        return self.providers.get(name)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List all providers.
        
        Returns:
            List of provider information
        """
        providers_info = []
        for name, provider in self.providers.items():
            stats = provider.get_stats()
            providers_info.append({
                "name": name,
                "type": provider.config.provider_type.value,
                "enabled": provider.is_enabled(),
                "priority": provider.get_priority(),
                "status": stats.status.value,
                "success_rate": stats.success_rate(),
                "total_requests": stats.total_requests,
                "average_response_time": stats.average_response_time,
            })
        
        return providers_info
    
    def get_providers_for_model(self, model: str) -> List[BaseProvider]:
        """
        Get providers that can handle a specific model.
        
        Args:
            model: Model name
            
        Returns:
            List of providers that can handle the model
        """
        providers = []
        for provider in self.providers.values():
            if provider.is_enabled() and provider.can_handle_model(model):
                providers.append(provider)
        
        # Sort by priority (higher priority first)
        providers.sort(key=lambda p: p.get_priority(), reverse=True)
        
        return providers
    
    async def select_provider(
        self,
        model: str,
        strategy: str = "priority"
    ) -> Optional[BaseProvider]:
        """
        Select a provider for a model using the specified strategy.
        
        Args:
            model: Model name
            strategy: Selection strategy ("priority", "round_robin", "random", "health_based")
            
        Returns:
            Selected provider, or None if no provider available
        """
        providers = self.get_providers_for_model(model)
        if not providers:
            return None
        
        if strategy == "priority":
            # Already sorted by priority
            return providers[0]
        
        elif strategy == "round_robin":
            # Simple round-robin based on request count
            providers.sort(key=lambda p: p.stats.total_requests)
            return providers[0]
        
        elif strategy == "random":
            return random.choice(providers)
        
        elif strategy == "health_based":
            # Select based on health status and success rate
            healthy_providers = [
                p for p in providers
                if p.get_status() == ProviderStatus.HEALTHY
            ]
            
            if healthy_providers:
                # Sort by success rate (highest first)
                healthy_providers.sort(
                    key=lambda p: p.stats.success_rate(),
                    reverse=True
                )
                return healthy_providers[0]
            else:
                # Fall back to priority if no healthy providers
                return providers[0]
        
        else:
            # Default to priority
            return providers[0]
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        provider_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using the appropriate provider.
        
        Args:
            model: Model name
            messages: List of messages
            provider_name: Optional specific provider name
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        start_time = datetime.utcnow()
        
        # Select provider
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                raise ValueError(f"Provider not found: {provider_name}")
            if not provider.can_handle_model(model):
                raise ValueError(f"Provider {provider_name} cannot handle model {model}")
        else:
            provider = await self.select_provider(model)
            if not provider:
                raise ValueError(f"No provider available for model {model}")
        
        try:
            # Make request
            response = await provider.chat_completion(
                messages=messages,
                model=model,
                **kwargs
            )
            
            # Update statistics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract token counts if available
            tokens = 0
            if "usage" in response:
                usage = response["usage"]
                tokens = usage.get("total_tokens", 0)
            
            provider.stats.update_request(
                success=True,
                tokens=tokens,
                cost=0.0,  # Would calculate based on provider pricing
                response_time=response_time,
            )
            
            return response
            
        except Exception as e:
            # Update statistics for failed request
            response_time = (datetime.utcnow() - start_time).total_seconds()
            provider.stats.update_request(
                success=False,
                response_time=response_time,
            )
            raise
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all providers.
        
        Returns:
            Dictionary of provider names to health status
        """
        health_status = {}
        
        for name, provider in self.providers.items():
            try:
                healthy = await provider.health_check()
                health_status[name] = healthy
            except Exception:
                health_status[name] = False
        
        return health_status
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics for all providers.
        
        Returns:
            Dictionary with overall statistics
        """
        total_requests = 0
        total_successful = 0
        total_failed = 0
        total_tokens = 0
        total_cost = 0.0
        
        for provider in self.providers.values():
            stats = provider.get_stats()
            total_requests += stats.total_requests
            total_successful += stats.successful_requests
            total_failed += stats.failed_requests
            total_tokens += stats.total_tokens
            total_cost += stats.total_cost
        
        overall_success_rate = 0.0
        if total_requests > 0:
            overall_success_rate = (total_successful / total_requests) * 100
        
        return {
            "total_providers": len(self.providers),
            "enabled_providers": len([p for p in self.providers.values() if p.is_enabled()]),
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_failed,
            "success_rate": overall_success_rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
