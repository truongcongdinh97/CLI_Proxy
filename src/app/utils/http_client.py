"""
HTTP client utilities for CLI Proxy API.
Provides async HTTP client with proxy support, retry logic, and connection pooling.
"""

import asyncio
import ssl
from typing import Dict, Optional, Any
from urllib.parse import urlparse

import httpx
from httpx import AsyncClient, Timeout, Limits
import structlog

logger = structlog.get_logger(__name__)


class HTTPClient:
    """Async HTTP client with proxy support and retry logic."""
    
    def __init__(
        self,
        config: Any,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize HTTP client.
        
        Args:
            config: Application configuration
            base_url: Base URL for all requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        self.config = config
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Configure proxy
        self.proxy_config = self._configure_proxy()
        
        # Create SSL context
        self.ssl_context = self._create_ssl_context()
        
        # Initialize client
        self.client = self._create_client()
        self._default_headers: Dict[str, str] = {}
    
    def set_default_headers(self, headers: Dict[str, str]) -> None:
        """Set default headers for all requests."""
        self._default_headers.update(headers)
    
    def _configure_proxy(self) -> Optional[Dict[str, str]]:
        """Configure proxy from config."""
        proxy_url = getattr(self.config, "proxy_url", None)
        if not proxy_url:
            return None
        
        try:
            parsed = urlparse(proxy_url)
            proxy_config = {
                "http://": proxy_url,
                "https://": proxy_url,
            }
            
            # Add authentication if present
            if parsed.username and parsed.password:
                proxy_config["auth"] = (parsed.username, parsed.password)
            
            logger.info(f"Proxy configured: {parsed.scheme}://{parsed.hostname}:{parsed.port}")
            return proxy_config
            
        except Exception as e:
            logger.warning(f"Failed to parse proxy URL {proxy_url}: {e}")
            return None
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context with appropriate settings."""
        try:
            context = ssl.create_default_context()
            # You can customize SSL settings here if needed
            # context.check_hostname = True
            # context.verify_mode = ssl.CERT_REQUIRED
            return context
        except Exception as e:
            logger.warning(f"Failed to create SSL context: {e}")
            return None
    
    def _create_client(self) -> AsyncClient:
        """Create HTTP client with configured settings."""
        transport_kwargs = {}
        
        # Configure proxy transport if proxy is set
        if self.proxy_config:
            transport = httpx.AsyncHTTPTransport(
                proxy=self.proxy_config,
                verify=self.ssl_context,
                retries=self.max_retries,
            )
            transport_kwargs["transport"] = transport
        
        # Build client kwargs - only add base_url if it's set
        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        # Create client with timeout and limits
        client = AsyncClient(
            **client_kwargs,
            timeout=Timeout(
                connect=5.0,
                read=self.timeout,
                write=self.timeout,
                pool=5.0,
            ),
            limits=Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
            verify=self.ssl_context if not self.proxy_config else False,
            **transport_kwargs,
        )
        
        # Set default headers
        client.headers.update({
            "User-Agent": "CLIProxyAPI-Python/1.0.0",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })
        
        return client
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments for httpx
            
        Returns:
            HTTP response
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Add attempt-specific headers and merge with default headers
                headers = dict(self._default_headers)
                headers.update(kwargs.get("headers", {}))
                headers["X-Retry-Attempt"] = str(attempt)
                kwargs["headers"] = headers
                
                logger.debug(
                    "HTTP request attempt",
                    method=method,
                    url=url,
                    attempt=attempt + 1,
                    max_attempts=self.max_retries + 1,
                )
                
                response = await self.client.request(method, url, **kwargs)
                
                # Check if we should retry
                if self._should_retry(response.status_code, attempt):
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            "Request failed, retrying",
                            method=method,
                            url=url,
                            status_code=response.status_code,
                            attempt=attempt + 1,
                            delay=delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                
                # Log successful request
                logger.debug(
                    "HTTP request completed",
                    method=method,
                    url=url,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )
                
                return response
                
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Network error, retrying",
                        method=method,
                        url=url,
                        error=str(e),
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Network error after all retries",
                        method=method,
                        url=url,
                        error=str(e),
                        max_attempts=self.max_retries + 1,
                    )
                    raise
            
            except Exception as e:
                last_exception = e
                logger.error(
                    "Unexpected error in HTTP request",
                    method=method,
                    url=url,
                    error=str(e),
                    attempt=attempt + 1,
                )
                raise
        
        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("HTTP request failed without exception")
    
    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            status_code: HTTP status code
            attempt: Current attempt number
            
        Returns:
            True if request should be retried
        """
        # Retry on these status codes
        retry_status_codes = {408, 429, 500, 502, 503, 504}
        
        # Also retry on 403 if configured (for quota exceeded)
        if hasattr(self.config, "request_retry") and self.config.request_retry > 0:
            if 403 in retry_status_codes:
                retry_status_codes.add(403)
        
        return status_code in retry_status_codes and attempt < self.max_retries
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def patch(self, url: str, **kwargs) -> httpx.Response:
        """Make PATCH request."""
        return await self.request("PATCH", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)
    
    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()


# Global HTTP client instance
_http_client: Optional[HTTPClient] = None


def get_http_client(config: Optional[Any] = None) -> HTTPClient:
    """
    Get or create global HTTP client instance.
    
    Args:
        config: Application configuration (required on first call)
        
    Returns:
        HTTPClient instance
    """
    global _http_client
    
    if _http_client is None:
        if config is None:
            raise ValueError("Config is required to create HTTP client")
        
        _http_client = HTTPClient(
            config=config,
            timeout=getattr(config, "request_timeout", 30.0),
            max_retries=getattr(config, "request_retry", 3),
        )
    
    return _http_client


async def close_http_client() -> None:
    """Close the global HTTP client."""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


def create_http_client_for_provider(
    config: Any,
    provider_config: Any,
    base_url: Optional[str] = None,
) -> HTTPClient:
    """
    Create HTTP client for a specific provider.
    
    Args:
        config: Application configuration
        provider_config: Provider-specific configuration
        base_url: Base URL for the provider
        
    Returns:
        HTTPClient instance
    """
    # Use provider-specific proxy if available
    proxy_url = None
    if hasattr(provider_config, "proxy_url") and provider_config.proxy_url:
        proxy_url = provider_config.proxy_url
    elif hasattr(config, "proxy_url"):
        proxy_url = config.proxy_url
    
    # Create config copy with provider-specific proxy
    class ProviderConfigWrapper:
        def __init__(self, config, proxy_url):
            self.config = config
            self.proxy_url = proxy_url
            
            # Copy other attributes from config
            for attr in dir(config):
                if not attr.startswith("_"):
                    try:
                        value = getattr(config, attr)
                        if not callable(value):
                            setattr(self, attr, value)
                    except:
                        pass
    
    wrapper_config = ProviderConfigWrapper(config, proxy_url)
    
    return HTTPClient(
        config=wrapper_config,
        base_url=base_url,
        timeout=getattr(config, "request_timeout", 30.0),
        max_retries=getattr(config, "request_retry", 3),
    )
