"""
Authentication module for CLI Proxy API.
Provides OAuth and API key authentication for various AI providers.
"""

from .base import BaseAuthProvider, AuthResult, TokenData
from .manager import AuthManager
from .claude import ClaudeAuth
from .gemini import GeminiAuth
from .openai import OpenAIAuth
from .qwen import QwenAuth
from .iflow import iFlowAuth

__all__ = [
    "BaseAuthProvider",
    "AuthResult",
    "TokenData",
    "AuthManager",
    "ClaudeAuth",
    "GeminiAuth",
    "OpenAIAuth",
    "QwenAuth",
    "iFlowAuth",
]
