"""
Configuration management for job application tracking system.

Handles environment variables, API keys, and application settings
using pydantic-settings for validation and type safety.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings with environment variable validation.
    
    All settings can be overridden via environment variables.
    """
    
    # Gmail OAuth2 Configuration
    gmail_credentials_path: str = Field(
        default="./credentials/credentials.json",
        description="Path to Gmail OAuth2 credentials file"
    )
    gmail_token_path: str = Field(
        default="./credentials/token.json", 
        description="Path to Gmail OAuth2 token file"
    )
    gmail_scopes_list: List[str] = Field(
        default=["https://www.googleapis.com/auth/gmail.readonly"],
        description="Gmail API scopes"
    )
    
    # Custom field for handling string input from env
    gmail_scopes_str: Optional[str] = Field(
        default=None,
        description="Gmail scopes as comma-separated string",
        alias="GMAIL_SCOPES"
    )
    
    # Brave Search API
    brave_api_key: Optional[str] = Field(
        default=None,
        description="Brave Search API key"
    )
    brave_base_url: str = Field(
        default="https://api.search.brave.com/res/v1",
        description="Brave Search API base URL"
    )
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM classification"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for classification"
    )
    
    # Application Settings
    email_days_back: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days back to retrieve emails"
    )
    classification_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for job application classification"
    )
    research_max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of research results to return"
    )
    
    # Rate limiting
    gmail_rate_limit_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Delay between Gmail API requests (seconds)"
    )
    brave_rate_limit_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between Brave API requests (seconds)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    # Optional features
    enable_validation: bool = Field(
        default=False,
        description="Enable classification validation step"
    )
    
    @validator('gmail_credentials_path')
    def validate_gmail_credentials_path(cls, v):
        """
        Validate Gmail credentials file path.
        """
        if not v:
            raise ValueError("Gmail credentials path cannot be empty")
        
        path = Path(v)
        if not path.parent.exists():
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(path)
    
    @validator('brave_api_key')
    def validate_brave_api_key(cls, v):
        """
        Validate Brave API key format.
        """
        if v and not v.startswith('BSA'):
            raise ValueError("Brave API key must start with 'BSA'")
        return v
    
    @validator('openai_api_key')
    def validate_openai_api_key(cls, v):
        """
        Validate OpenAI API key format.
        """
        if v and not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def handle_custom_fields(cls, values):
        """
        Handle custom field processing.
        
        Args:
            values (dict): Field values
            
        Returns:
            dict: Updated values
        """
        # Handle gmail_scopes from string input
        if values.get('gmail_scopes_str') or values.get('GMAIL_SCOPES'):
            scopes_str = values.get('gmail_scopes_str') or values.get('GMAIL_SCOPES')
            if isinstance(scopes_str, str) and scopes_str.strip():
                values['gmail_scopes_list'] = [scope.strip() for scope in scopes_str.split(',') if scope.strip()]
        
        return values
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """
        Validate log level.
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def validate_required_keys(self) -> None:
        """
        Validate that required API keys are present.
        
        Raises:
            ValueError: If required keys are missing
        """
        missing_keys = []
        
        if not self.brave_api_key:
            missing_keys.append("BRAVE_API_KEY")
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        
        if missing_keys:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_keys)}"
            )
    
    def get_gmail_credentials_path(self) -> Path:
        """
        Get Gmail credentials path as Path object.
        
        Returns:
            Path: Gmail credentials file path
        """
        return Path(self.gmail_credentials_path)
    
    def get_gmail_token_path(self) -> Path:
        """
        Get Gmail token path as Path object.
        
        Returns:
            Path: Gmail token file path
        """
        return Path(self.gmail_token_path)
    
    @property
    def gmail_scopes(self) -> List[str]:
        """
        Get Gmail scopes list for backward compatibility.
        
        Returns:
            List[str]: Gmail API scopes
        """
        return self.gmail_scopes_list
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow environment variables to override field names
        env_prefix = ""

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).
    
    Returns:
        Settings: Global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def validate_environment() -> None:
    """
    Validate that all required environment variables are present.
    
    Raises:
        ValueError: If validation fails
    """
    settings = get_settings()
    settings.validate_required_keys()
    
    # Check Gmail credentials file exists
    if not settings.get_gmail_credentials_path().exists():
        raise ValueError(
            f"Gmail credentials file not found: {settings.gmail_credentials_path}. "
            "Please download credentials.json from Google Cloud Console."
        )

if __name__ == "__main__":
    # Test configuration loading
    try:
        settings = get_settings()
        print("✅ Configuration loaded successfully")
        print(f"📧 Email days back: {settings.email_days_back}")
        print(f"🔍 Classification threshold: {settings.classification_confidence_threshold}")
        print(f"📊 Research max results: {settings.research_max_results}")
        
        # Test validation
        validate_environment()
        print("✅ Environment validation passed")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        exit(1)