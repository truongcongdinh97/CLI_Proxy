"""
Main FastAPI application for CLI Proxy API (Python implementation).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from .config import load_config, get_config
from .utils.http_client import get_http_client
from .api.routes import router as api_router
from .auth.manager import AuthManager
from .providers.registry import ProviderRegistry
from .stores.manager import StoreManager
from .translator.registry import TranslatorRegistry


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# Global application state
class AppState:
    """Application state shared across the application."""
    def __init__(self):
        self.config = None
        self.auth_manager = None
        self.provider_registry = None
        self.store_manager = None
        self.translator_registry = None
        self.http_client = None
        self.is_shutting_down = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI application."""
    # Startup
    logger.info("Starting CLI Proxy API (Python)")
    
    try:
        # Load configuration
        app_state.config = load_config()
        logger.info(f"Configuration loaded, port: {app_state.config.port}")
        
        # Initialize HTTP client
        app_state.http_client = get_http_client(app_state.config)
        
        # Initialize store manager
        app_state.store_manager = StoreManager(app_state.config)
        await app_state.store_manager.initialize()
        
        # Initialize auth manager
        app_state.auth_manager = AuthManager(
            config=app_state.config,
            store_manager=app_state.store_manager,
            http_client=app_state.http_client
        )
        
        # Initialize provider registry
        app_state.provider_registry = ProviderRegistry(
            config=app_state.config,
            auth_manager=app_state.auth_manager,
            http_client=app_state.http_client
        )
        await app_state.provider_registry.initialize()
        
        # Initialize translator registry
        app_state.translator_registry = TranslatorRegistry()
        
        # Set state on app for route access
        app.state.config = app_state.config
        app.state.auth_manager = app_state.auth_manager
        app.state.provider_registry = app_state.provider_registry
        app.state.store_manager = app_state.store_manager
        app.state.translator_registry = app_state.translator_registry
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down CLI Proxy API")
        app_state.is_shutting_down = True
        
        if app_state.http_client:
            await app_state.http_client.aclose()
        
        if app_state.store_manager:
            await app_state.store_manager.shutdown()
        
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="CLI Proxy API",
    description="OpenAI/Gemini/Claude compatible API proxy for CLI tools",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url="/redoc" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = asyncio.get_event_loop().time()
    
    # Get request details
    request_id = request.headers.get("X-Request-ID", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    
    logger.info(
        "Request started",
        request_id=request_id,
        client_ip=client_ip,
        method=method,
        path=path,
        query_params=dict(request.query_params),
    )
    
    try:
        response = await call_next(request)
        process_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(
            "Request completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=response.status_code,
            process_time=f"{process_time:.3f}s",
        )
        
        # Add process time header
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        return response
        
    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        logger.error(
            "Request failed",
            request_id=request_id,
            method=method,
            path=path,
            error=str(e),
            process_time=f"{process_time:.3f}s",
            exc_info=True,
        )
        raise


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "internal_error",
            }
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "message": f"Endpoint {request.url.path} not found",
                "type": "not_found",
                "code": "not_found",
            }
        },
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if app_state.is_shutting_down:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "shutting_down"},
        )
    
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "config": "healthy" if app_state.config else "unhealthy",
            "auth": "healthy" if app_state.auth_manager else "unhealthy",
            "providers": "healthy" if app_state.provider_registry else "unhealthy",
            "store": "healthy" if app_state.store_manager else "unhealthy",
        },
    }
    
    return health_status


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CLI Proxy API",
        "version": "1.0.0",
        "description": "OpenAI/Gemini/Claude compatible API proxy for CLI tools",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "openai_compatible": "/v1/chat/completions",
            "models": "/v1/models",
            "gemini": "/v1beta/models",
            "claude": "/v1/messages",
        },
        "supported_providers": [
            "openai",
            "gemini", 
            "claude",
            "codex",
            "qwen",
            "iflow",
            "vertex",
        ],
    }


# Include API routers
app.include_router(api_router, prefix="/v1")


def main():
    """Main entry point for the application."""
    import os
    import sys
    
    # Set up basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="CLI Proxy API Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the server on (overrides config)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    
    args = parser.parse_args()
    
    # Set config file environment variable if provided
    if args.config:
        os.environ["CLIPROXY_CONFIG_FILE"] = args.config
    
    # Determine port
    port = args.port
    if not port:
        # Try to load config to get port
        try:
            config = load_config(args.config)
            port = config.port
        except:
            port = 8317  # Default
    
    # Start the server
    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_config=None,
        access_log=True,
    )


if __name__ == "__main__":
    main()
