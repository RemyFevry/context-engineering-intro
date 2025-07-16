"""
Tests for configuration and settings module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from config.settings import Settings, get_settings


class TestSettings:
    """Test Settings class."""
    
    def test_settings_default_values(self):
        """Test default settings values."""
        settings = Settings()
        
        assert settings.gmail_credentials_path == "./credentials/credentials.json"
        assert settings.gmail_token_path == "./credentials/token.json"
        assert settings.email_days_back == 30
        assert settings.classification_confidence_threshold == 0.7
        assert settings.research_max_results == 10
        assert settings.rate_limit_delay == 1.0
        assert settings.openai_model == "gpt-4"
        assert settings.log_level == "INFO"
        assert settings.enable_validation is True
    
    def test_settings_with_env_vars(self):
        """Test settings with environment variables."""
        with patch.dict(os.environ, {
            'BRAVE_API_KEY': 'test_brave_key',
            'OPENAI_API_KEY': 'test_openai_key',
            'EMAIL_DAYS_BACK': '60',
            'CLASSIFICATION_CONFIDENCE_THRESHOLD': '0.8',
            'RESEARCH_MAX_RESULTS': '20',
            'RATE_LIMIT_DELAY': '2.0',
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'LOG_LEVEL': 'DEBUG',
            'ENABLE_VALIDATION': 'false'
        }):
            settings = Settings()
            
            assert settings.brave_api_key == 'test_brave_key'
            assert settings.openai_api_key == 'test_openai_key'
            assert settings.email_days_back == 60
            assert settings.classification_confidence_threshold == 0.8
            assert settings.research_max_results == 20
            assert settings.rate_limit_delay == 2.0
            assert settings.openai_model == 'gpt-3.5-turbo'
            assert settings.log_level == 'DEBUG'
            assert settings.enable_validation is False
    
    def test_settings_validation_errors(self):
        """Test settings validation errors."""
        with patch.dict(os.environ, {
            'EMAIL_DAYS_BACK': '0',  # Should be >= 1
        }):
            with pytest.raises(ValidationError):
                Settings()
        
        with patch.dict(os.environ, {
            'EMAIL_DAYS_BACK': '400',  # Should be <= 365
        }):
            with pytest.raises(ValidationError):
                Settings()
        
        with patch.dict(os.environ, {
            'CLASSIFICATION_CONFIDENCE_THRESHOLD': '1.5',  # Should be <= 1.0
        }):
            with pytest.raises(ValidationError):
                Settings()
        
        with patch.dict(os.environ, {
            'RESEARCH_MAX_RESULTS': '0',  # Should be >= 1
        }):
            with pytest.raises(ValidationError):
                Settings()
        
        with patch.dict(os.environ, {
            'RATE_LIMIT_DELAY': '-1',  # Should be >= 0
        }):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_settings_path_resolution(self):
        """Test path resolution in settings."""
        settings = Settings()
        
        # Test default paths
        assert settings.gmail_credentials_path.endswith('credentials.json')
        assert settings.gmail_token_path.endswith('token.json')
        
        # Test custom paths
        with patch.dict(os.environ, {
            'GMAIL_CREDENTIALS_PATH': '/custom/path/credentials.json',
            'GMAIL_TOKEN_PATH': '/custom/path/token.json'
        }):
            custom_settings = Settings()
            assert custom_settings.gmail_credentials_path == '/custom/path/credentials.json'
            assert custom_settings.gmail_token_path == '/custom/path/token.json'
    
    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_settings_repr(self):
        """Test settings string representation."""
        settings = Settings()
        repr_str = repr(settings)
        
        # Should contain key information but not sensitive data
        assert 'Settings' in repr_str
        assert 'email_days_back' in repr_str
        assert 'classification_confidence_threshold' in repr_str
        
        # Should not contain sensitive information
        assert 'api_key' not in repr_str.lower()
    
    def test_settings_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_levels:
            with patch.dict(os.environ, {'LOG_LEVEL': level}):
                settings = Settings()
                assert settings.log_level == level
        
        # Test invalid log level
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID'}):
            with pytest.raises(ValidationError):
                Settings()
    
    def test_settings_openai_model_validation(self):
        """Test OpenAI model validation."""
        valid_models = ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo']
        
        for model in valid_models:
            with patch.dict(os.environ, {'OPENAI_MODEL': model}):
                settings = Settings()
                assert settings.openai_model == model
        
        # Test invalid model
        with patch.dict(os.environ, {'OPENAI_MODEL': 'invalid-model'}):
            with pytest.raises(ValidationError):
                Settings()


class TestSettingsIntegration:
    """Test settings integration with other components."""
    
    def test_settings_with_missing_credentials(self):
        """Test behavior when credentials are missing."""
        with patch.dict(os.environ, {
            'GMAIL_CREDENTIALS_PATH': '/nonexistent/path/credentials.json'
        }):
            settings = Settings()
            
            # Should not raise error during settings creation
            assert settings.gmail_credentials_path == '/nonexistent/path/credentials.json'
    
    def test_settings_with_partial_configuration(self):
        """Test settings with partial configuration."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            # Missing BRAVE_API_KEY
        }):
            settings = Settings()
            
            assert settings.openai_api_key == 'test_key'
            assert settings.brave_api_key is None
    
    @patch('config.settings.load_dotenv')
    def test_settings_loads_dotenv(self, mock_load_dotenv):
        """Test that settings loads .env file."""
        Settings()
        mock_load_dotenv.assert_called_once()
    
    def test_settings_field_descriptions(self):
        """Test that all settings have proper field descriptions."""
        settings = Settings()
        
        # Check that model fields have descriptions
        for field_name, field_info in settings.model_fields.items():
            assert field_info.description, f"Field {field_name} should have a description"
    
    def test_settings_serialization(self):
        """Test settings serialization (excluding sensitive data)."""
        settings = Settings()
        
        # Test model dump
        settings_dict = settings.model_dump()
        
        # Should contain configuration but not sensitive data
        assert 'email_days_back' in settings_dict
        assert 'classification_confidence_threshold' in settings_dict
        
        # Sensitive data should be excluded or masked
        if 'openai_api_key' in settings_dict:
            assert settings_dict['openai_api_key'] is None or settings_dict['openai_api_key'] == 'test_key'
        if 'brave_api_key' in settings_dict:
            assert settings_dict['brave_api_key'] is None or settings_dict['brave_api_key'] == 'test_key'


# Pytest fixtures for testing
@pytest.fixture
def mock_env_vars():
    """Fixture providing mock environment variables."""
    return {
        'BRAVE_API_KEY': 'test_brave_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'EMAIL_DAYS_BACK': '30',
        'CLASSIFICATION_CONFIDENCE_THRESHOLD': '0.8',
        'RESEARCH_MAX_RESULTS': '10',
        'RATE_LIMIT_DELAY': '1.0',
        'OPENAI_MODEL': 'gpt-4',
        'LOG_LEVEL': 'INFO',
        'ENABLE_VALIDATION': 'true'
    }


@pytest.fixture
def settings_with_env(mock_env_vars):
    """Fixture providing settings with environment variables."""
    with patch.dict(os.environ, mock_env_vars):
        yield Settings()


class TestSettingsFixtures:
    """Test using settings fixtures."""
    
    def test_settings_fixture(self, settings_with_env):
        """Test settings fixture works correctly."""
        assert settings_with_env.brave_api_key == 'test_brave_key'
        assert settings_with_env.openai_api_key == 'test_openai_key'
        assert settings_with_env.email_days_back == 30
        assert settings_with_env.classification_confidence_threshold == 0.8
    
    def test_mock_env_vars_fixture(self, mock_env_vars):
        """Test mock environment variables fixture."""
        assert mock_env_vars['BRAVE_API_KEY'] == 'test_brave_key'
        assert mock_env_vars['OPENAI_API_KEY'] == 'test_openai_key'
        assert mock_env_vars['EMAIL_DAYS_BACK'] == '30'


if __name__ == "__main__":
    pytest.main([__file__])