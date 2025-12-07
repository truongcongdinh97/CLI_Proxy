"""
Base provider interface for AI service providers.
"""

import abc
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    """Provider type enumeration."""
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"
    IFLOW = "iflow"
    VERTEX = "vertex"


class ProviderStatus(Enum):
    """Provider status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ProviderConfig:
    """Provider configuration."""
    name: str
    provider_type: ProviderType
    base_url: str
    api_key: Optional[str] = None
    priority: int = 1
    enabled: bool = True
    max_requests_per_minute: int = 60
    timeout: float = 30.0
    retry_count: int = 3
    headers: Optional[Dict[str, str]] = None
    proxy_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "priority": self.priority,
            "enabled": self.enabled,
            "max_requests_per_minute": self.max_requests_per_minute,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "headers": self.headers or {},
            "proxy_url": self.proxy_url,
        }


@dataclass
class ProviderStats:
    """Provider statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    status: ProviderStatus = ProviderStatus.HEALTHY
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def update_request(self, success: bool, tokens: int = 0, cost: float = 0.0, response_time: float = 0.0):
        """Update statistics for a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_cost += cost
        
        # Update average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
        
        self.last_request_time = datetime.utcnow()
        
        # Update status based on success rate
        success_rate = self.success_rate()
        if success_rate >= 95:
            self.status = ProviderStatus.HEALTHY
        elif success_rate >= 80:
            self.status = ProviderStatus.DEGRADED
        else:
            self.status = ProviderStatus.UNHEALTHY


class BaseProvider(abc.ABC):
    """Base class for all AI providers."""
    
    def __init__(self, config: ProviderConfig, auth_manager: Any, http_client: Any):
        """
        Initialize provider.
        
        Args:
            config: Provider configuration
            auth_manager: Authentication manager
            http_client: HTTP client
        """
        self.config = config
        self.auth_manager = auth_manager
        self.http_client = http_client
        self.stats = ProviderStats()
    
    def _set_status(self, status: ProviderStatus) -> None:
        """Set provider status."""
        self.stats.status = status
    
    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass
    
    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the provider."""
        pass
    
    @abc.abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            messages: List of messages
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        pass
    
    @abc.abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information
        """
        pass
    
    @abc.abstractmethod
    async def get_model_info(self, model: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Model information, or None if not found
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Perform health check.
        
        Returns:
            True if provider is healthy
        """
        try:
            await self.list_models()
            return True
        except Exception:
            return False
    
    def get_stats(self) -> ProviderStats:
        """
        Get provider statistics.
        
        Returns:
            Provider statistics
        """
        return self.stats
    
    def get_status(self) -> ProviderStatus:
        """
        Get provider status.
        
        Returns:
            Provider status
        """
        return self.stats.status
    
    def can_handle_model(self, model: str) -> bool:
        """
        Check if provider can handle a specific model.
        
        Args:
            model: Model name
            
        Returns:
            True if provider can handle the model
        """
        # Default implementation checks if model starts with provider type
        return model.lower().startswith(self.config.provider_type.value)
    
    def get_priority(self) -> int:
        """
        Get provider priority.
        
        Returns:
            Provider priority
        """
        return self.config.priority
    
    def is_enabled(self) -> bool:
        """
        Check if provider is enabled.
        
        Returns:
            True if provider is enabled
        """
        return self.config.enabled
