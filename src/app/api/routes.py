"""
"""API routes for CLI Proxy API.
"""

import json
from typing import Dict, Any, List, AsyncGenerator
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth.manager import AuthManager
from ..providers.registry import ProviderRegistry
from ..translator.registry import TranslatorRegistry

router = APIRouter()


# Dependency injection functions
async def get_auth_manager(request: Request) -> AuthManager:
    """Get auth manager from app state."""
    return request.app.state.auth_manager


async def get_provider_registry(request: Request) -> ProviderRegistry:
    """Get provider registry from app state."""
    return request.app.state.provider_registry


async def get_translator_registry(request: Request) -> TranslatorRegistry:
    """Get translator registry from app state."""
    return request.app.state.translator_registry


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "cli-proxy-api"}


# OpenAI-compatible endpoints
@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    OpenAI-compatible chat completions endpoint.
    
    This endpoint accepts OpenAI format requests and routes them to the
    appropriate provider.
    """
    try:
        # Parse request body
        request_data = await request.json()
        
        # Extract required fields
        model = request_data.get("model")
        messages = request_data.get("messages", [])
        
        if not model:
            raise HTTPException(status_code=400, detail="Model is required")
        
        if not messages:
            raise HTTPException(status_code=400, detail="Messages are required")
        
        # Extract optional parameters
        temperature = request_data.get("temperature")
        max_tokens = request_data.get("max_tokens")
        top_p = request_data.get("top_p")
        frequency_penalty = request_data.get("frequency_penalty")
        presence_penalty = request_data.get("presence_penalty")
        stop = request_data.get("stop")
        stream = request_data.get("stream", False)
        
        # Prepare kwargs for provider
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if frequency_penalty is not None:
            kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            kwargs["stop"] = stop
        if stream is not None:
            kwargs["stream"] = stream
        
        # Check if streaming is requested
        if stream:
            # Return streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                async for chunk in provider_registry.chat_completion_stream(
                    model=model,
                    messages=messages,
                    **kwargs
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        # Get completion from provider registry (non-streaming)
        response = await provider_registry.chat_completion(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/models")
async def list_models(
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    List available models from all providers.
    
    Returns a combined list of models from all enabled providers.
    """
    try:
        # This would normally aggregate models from all providers
        # For now, return a static list
        models = [
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            },
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai",
            },
            {
                "id": "gemini-pro",
                "object": "model",
                "created": 1692902400,
                "owned_by": "google",
            },
            {
                "id": "claude-3-opus",
                "object": "model",
                "created": 1704067200,
                "owned_by": "anthropic",
            },
        ]
        
        return {
            "object": "list",
            "data": models,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Authentication endpoints
@router.post("/auth/{provider_name}")
async def authenticate(
    provider_name: str,
    request: Request,
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Authenticate with a provider.
    
    Supports API key authentication for most providers.
    """
    try:
        request_data = await request.json()
        
        # Extract authentication parameters
        api_key = request_data.get("api_key")
        key_id = request_data.get("key_id", api_key[:8] if api_key else "default")
        
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Authenticate
        auth_result = await auth_manager.authenticate(
            provider_name=provider_name,
            key_id=key_id,
            api_key=api_key,
        )
        
        if not auth_result.success:
            raise HTTPException(
                status_code=401,
                detail=auth_result.error or "Authentication failed",
            )
        
        return {
            "success": True,
            "provider": provider_name,
            "key_id": key_id,
            "token_data": auth_result.token_data.to_dict() if auth_result.token_data else None,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/auth/{provider_name}/tokens")
async def list_tokens(
    provider_name: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    List tokens for a provider.
    """
    try:
        tokens = await auth_manager.list_tokens(provider_name)
        return {
            "provider": provider_name,
            "tokens": tokens,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/auth/{provider_name}/{key_id}")
async def delete_token(
    provider_name: str,
    key_id: str,
    auth_manager: AuthManager = Depends(get_auth_manager),
):
    """
    Delete a token.
    """
    try:
        success = await auth_manager.delete_token(provider_name, key_id)
        return {
            "success": success,
            "provider": provider_name,
            "key_id": key_id,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Provider management endpoints
@router.get("/providers")
async def list_providers(
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    List all providers and their status.
    """
    try:
        providers = provider_registry.list_providers()
        return {
            "providers": providers,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/providers/stats")
async def get_provider_stats(
    provider_registry: ProviderRegistry = Depends(get_provider_registry),
):
    """
    Get overall provider statistics.
    """
    try:
        stats = provider_registry.get_overall_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Translation endpoints
@router.post("/translate/{source_format}/{target_format}")
async def translate_request(
    source_format: str,
    target_format: str,
    request: Request,
    translator_registry: TranslatorRegistry = Depends(get_translator_registry),
):
    """
    Translate a request from one format to another.
    
    Useful for testing and debugging translation logic.
    """
    try:
        request_data = await request.json()
        
        translation_result = await translator_registry.translate_request(
            source_format=source_format,
            target_format=target_format,
            request_data=request_data,
        )
        
        if not translation_result.success:
            raise HTTPException(
                status_code=400,
                detail=translation_result.error or "Translation failed",
            )
        
        return {
            "success": True,
            "source_format": source_format,
            "target_format": target_format,
            "translated_data": translation_result.translated_data,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# System information endpoint
@router.get("/system/info")
async def system_info():
    """
    Get system information.
    """
    return {
        "name": "CLI Proxy API",
        "version": "1.0.0",
        "description": "OpenAI/Gemini/Claude compatible API proxy for CLI tools",
        "supported_providers": ["openai", "gemini", "claude", "qwen", "iflow"],
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "auth": "/v1/auth/{provider}",
            "providers": "/v1/providers",
            "translate": "/v1/translate/{source}/{target}",
            "health": "/v1/health",
            "system_info": "/v1/system/info",
        },
    }
