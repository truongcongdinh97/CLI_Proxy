"""
Configuration management for CLI Proxy API (Python implementation).
Uses Pydantic for type-safe configuration with YAML file support.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class TLSConfig(BaseModel):
    """TLS configuration for HTTPS."""
    enable: bool = False
    cert: str = ""
    key: str = ""


class RemoteManagement(BaseModel):
    """Remote management API configuration."""
    allow_remote: bool = Field(default=False, alias="allow-remote")
    secret_key: str = Field(default="", alias="secret-key")
    disable_control_panel: bool = Field(default=False, alias="disable-control-panel")


class QuotaExceeded(BaseModel):
    """Quota exceeded behavior configuration."""
    switch_project: bool = Field(default=True, alias="switch-project")
    switch_preview_model: bool = Field(default=True, alias="switch-preview-model")


class AmpModelMapping(BaseModel):
    """Amp CLI model mapping configuration."""
    from_model: str = Field(alias="from")
    to_model: str = Field(alias="to")


class AmpCode(BaseModel):
    """Amp CLI integration configuration."""
    upstream_url: str = Field(default="", alias="upstream-url")
    upstream_api_key: str = Field(default="", alias="upstream-api-key")
    restrict_management_to_localhost: bool = Field(
        default=True, alias="restrict-management-to-localhost"
    )
    model_mappings: List[AmpModelMapping] = Field(
        default_factory=list, alias="model-mappings"
    )


class ProviderModel(BaseModel):
    """Provider model configuration with alias support."""
    name: str
    alias: str


class GeminiKey(BaseModel):
    """Gemini API key configuration."""
    api_key: str = Field(alias="api-key")
    base_url: Optional[str] = Field(default=None, alias="base-url")
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")
    headers: Optional[Dict[str, str]] = None
    excluded_models: Optional[List[str]] = Field(
        default=None, alias="excluded-models"
    )


class CodexKey(BaseModel):
    """Codex API key configuration."""
    api_key: str = Field(alias="api-key")
    base_url: Optional[str] = Field(default=None, alias="base-url")
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")
    headers: Optional[Dict[str, str]] = None
    excluded_models: Optional[List[str]] = Field(
        default=None, alias="excluded-models"
    )


class ClaudeKey(BaseModel):
    """Claude API key configuration."""
    api_key: str = Field(alias="api-key")
    base_url: Optional[str] = Field(default=None, alias="base-url")
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")
    headers: Optional[Dict[str, str]] = None
    models: Optional[List[ProviderModel]] = None
    excluded_models: Optional[List[str]] = Field(
        default=None, alias="excluded-models"
    )


class OpenAICompatibilityAPIKey(BaseModel):
    """OpenAI compatibility API key entry."""
    api_key: str = Field(alias="api-key")
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")


class OpenAICompatibility(BaseModel):
    """OpenAI compatibility provider configuration."""
    name: str
    base_url: str = Field(alias="base-url")
    api_key_entries: List[OpenAICompatibilityAPIKey] = Field(
        default_factory=list, alias="api-key-entries"
    )
    models: List[ProviderModel]
    headers: Optional[Dict[str, str]] = None


class VertexCompatKey(BaseModel):
    """Vertex-compatible API key configuration."""
    api_key: str = Field(alias="api-key")
    base_url: str = Field(alias="base-url")
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")
    headers: Optional[Dict[str, str]] = None
    models: Optional[List[ProviderModel]] = None


class PayloadModelRule(BaseModel):
    """Payload model rule for parameter injection."""
    name: str
    protocol: str


class PayloadRule(BaseModel):
    """Payload rule for default or override parameters."""
    models: List[PayloadModelRule]
    params: Dict[str, Any]


class PayloadConfig(BaseModel):
    """Payload configuration for parameter injection."""
    default: List[PayloadRule] = Field(default_factory=list)
    override: List[PayloadRule] = Field(default_factory=list)


class AppConfig(BaseSettings):
    """Main application configuration."""
    model_config = SettingsConfigDict(
        env_prefix="CLIPROXY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    port: int = 8317
    tls: TLSConfig = Field(default_factory=TLSConfig)
    remote_management: RemoteManagement = Field(
        default_factory=RemoteManagement, alias="remote-management"
    )
    
    # Authentication
    auth_dir: str = Field(default="~/.cli-proxy-api", alias="auth-dir")
    api_keys: List[str] = Field(default_factory=list, alias="api-keys")
    
    # Logging and debugging
    debug: bool = False
    logging_to_file: bool = Field(default=False, alias="logging-to-file")
    usage_statistics_enabled: bool = Field(
        default=False, alias="usage-statistics-enabled"
    )
    
    # Proxy and retry settings
    proxy_url: Optional[str] = Field(default=None, alias="proxy-url")
    request_retry: int = Field(default=3, alias="request-retry")
    max_retry_interval: int = Field(default=30, alias="max-retry-interval")
    
    # Quota management
    quota_exceeded: QuotaExceeded = Field(default_factory=QuotaExceeded)
    
    # WebSocket
    ws_auth: bool = Field(default=False, alias="ws-auth")
    
    # Provider configurations
    gemini_api_key: List[GeminiKey] = Field(
        default_factory=list, alias="gemini-api-key"
    )
    codex_api_key: List[CodexKey] = Field(
        default_factory=list, alias="codex-api-key"
    )
    claude_api_key: List[ClaudeKey] = Field(
        default_factory=list, alias="claude-api-key"
    )
    openai_compatibility: List[OpenAICompatibility] = Field(
        default_factory=list, alias="openai-compatibility"
    )
    vertex_api_key: List[VertexCompatKey] = Field(
        default_factory=list, alias="vertex-api-key"
    )
    
    # Amp integration
    ampcode: AmpCode = Field(default_factory=AmpCode)
    
    # OAuth excluded models
    oauth_excluded_models: Dict[str, List[str]] = Field(
        default_factory=dict, alias="oauth-excluded-models"
    )
    
    # Payload configuration
    payload: PayloadConfig = Field(default_factory=PayloadConfig)
    
    # Internal fields
    disable_cooling: bool = Field(default=False, alias="disable-cooling")
    
    @validator("auth_dir", pre=True)
    def expand_auth_dir(cls, v: str) -> str:
        """Expand ~ in auth directory path."""
        if v.startswith("~"):
            return os.path.expanduser(v)
        return v
    
    @root_validator(pre=True)
    def load_from_yaml_if_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from YAML file if config_file is provided."""
        config_file = values.get("config_file")
        if config_file and isinstance(config_file, str):
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        # Merge YAML config with existing values
                        # YAML values take precedence over existing values
                        for key, value in yaml_config.items():
                            if key not in values or values[key] is None:
                                values[key] = value
        return values
    
    @classmethod
    def from_file(cls, config_file: str) -> "AppConfig":
        """Load configuration from a YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        
        if not yaml_config:
            yaml_config = {}
        
        # Add config_file to values for the validator
        yaml_config["config_file"] = config_file
        
        return cls(**yaml_config)
    
    def save_to_file(self, config_file: str) -> None:
        """Save configuration to a YAML file."""
        config_path = Path(config_file)
        
        # Convert to dict and remove internal fields
        config_dict = self.model_dump(
            by_alias=True,
            exclude={"config_file"},
            exclude_none=True
        )
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_provider_config(self, provider: str) -> List[Any]:
        """Get configuration for a specific provider."""
        provider_map = {
            "gemini": self.gemini_api_key,
            "codex": self.codex_api_key,
            "claude": self.claude_api_key,
            "openai": self.openai_compatibility,
            "vertex": self.vertex_api_key,
        }
        return provider_map.get(provider.lower(), [])
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate port range
        if not (1 <= self.port <= 65535):
            errors.append(f"Port {self.port} is out of valid range (1-65535)")
        
        # Validate TLS configuration
        if self.tls.enable:
            if not self.tls.cert or not self.tls.key:
                errors.append("TLS enabled but cert or key path is empty")
            elif not Path(self.tls.cert).exists():
                errors.append(f"TLS cert file not found: {self.tls.cert}")
            elif not Path(self.tls.key).exists():
                errors.append(f"TLS key file not found: {self.tls.key}")
        
        # Validate auth directory
        try:
            auth_dir = Path(self.auth_dir).expanduser()
            if not auth_dir.exists():
                auth_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create auth directory: {e}")
        
        return errors


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _config


def load_config(config_file: Optional[str] = None) -> AppConfig:
    """Load configuration from file or environment."""
    global _config
    
    if config_file:
        _config = AppConfig.from_file(config_file)
    else:
        # Try to find config file in common locations
        config_locations = [
            "config.yaml",
            "config/config.yaml",
            "/app/config/config.yaml",
            os.path.expanduser("~/.cli-proxy-api/config.yaml"),
        ]
        
        for location in config_locations:
            if Path(location).exists():
                _config = AppConfig.from_file(location)
                break
        else:
            # No config file found, use defaults
            _config = AppConfig()
    
    # Validate configuration
    errors = _config.validate_config()
    if errors:
        print("Configuration errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    
    return _config


def reload_config(config_file: Optional[str] = None) -> AppConfig:
    """Reload configuration from file."""
    global _config
    _config = None
    return load_config(config_file)
