"""
Base translator interface for converting between AI provider API formats.
"""

import abc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    success: bool
    translated_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    source_format: Optional[str] = None
    target_format: Optional[str] = None
    
    @classmethod
    def success_result(cls, translated_data: Dict[str, Any], source_format: str, target_format: str) -> "TranslationResult":
        """Create a successful translation result."""
        return cls(
            success=True,
            translated_data=translated_data,
            source_format=source_format,
            target_format=target_format,
        )
    
    @classmethod
    def error_result(cls, error: str, error_code: str, source_format: str, target_format: str) -> "TranslationResult":
        """Create an error translation result."""
        return cls(
            success=False,
            error=error,
            error_code=error_code,
            source_format=source_format,
            target_format=target_format,
        )


class BaseTranslator(abc.ABC):
    """Base class for all translators."""
    
    def __init__(self, source_format: str, target_format: str):
        """
        Initialize translator.
        
        Args:
            source_format: Source API format (e.g., "openai", "gemini")
            target_format: Target API format (e.g., "gemini", "claude")
        """
        self.source_format = source_format
        self.target_format = target_format
    
    @abc.abstractmethod
    async def translate_request(self, request_data: Dict[str, Any]) -> TranslationResult:
        """
        Translate a request from source format to target format.
        
        Args:
            request_data: Request data in source format
            
        Returns:
            TranslationResult with translated data
        """
        pass
    
    @abc.abstractmethod
    async def translate_response(self, response_data: Dict[str, Any]) -> TranslationResult:
        """
        Translate a response from target format back to source format.
        
        Args:
            response_data: Response data in target format
            
        Returns:
            TranslationResult with translated data
        """
        pass
    
    def _extract_messages(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract messages from request data in a standardized format.
        
        Args:
            request_data: Request data
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # OpenAI format
        if "messages" in request_data:
            messages = request_data["messages"]
        
        # Gemini format
        elif "contents" in request_data:
            contents = request_data["contents"]
            for content in contents:
                if "parts" in content:
                    for part in content["parts"]:
                        if "text" in part:
                            role = content.get("role", "user")
                            messages.append({
                                "role": role,
                                "content": part["text"]
                            })
        
        # Claude format
        elif "messages" in request_data and isinstance(request_data["messages"], list):
            messages = request_data["messages"]
        
        return messages
    
    def _create_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create OpenAI format messages.
        
        Args:
            messages: List of messages in standardized format
            
        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        for msg in messages:
            openai_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            }
            
            # Add name if present
            if "name" in msg:
                openai_msg["name"] = msg["name"]
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    def _create_gemini_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create Gemini format contents.
        
        Args:
            messages: List of messages in standardized format
            
        Returns:
            List of contents in Gemini format
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles to Gemini roles
            gemini_role = "user"
            if role in ["assistant", "system"]:
                gemini_role = "model"
            
            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        return contents
    
    def _create_claude_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create Claude format messages.
        
        Args:
            messages: List of messages in standardized format
            
        Returns:
            List of messages in Claude format
        """
        claude_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Claude uses "assistant" role instead of "model"
            claude_role = role
            if role == "model":
                claude_role = "assistant"
            
            claude_messages.append({
                "role": claude_role,
                "content": content
            })
        
        return claude_messages
