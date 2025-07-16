"""
Tests for utility functions module.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import yaml
import logging

from utils import (
    setup_logging,
    call_llm,
    parse_yaml_response,
    format_datetime,
    get_date_range,
    extract_domain_from_email,
    clean_text,
    truncate_text,
    is_job_related_domain,
    calculate_confidence_score,
    safe_get,
    logger
)


class TestLogging:
    """Test logging utilities."""
    
    @patch('utils.get_settings')
    def test_setup_logging_default(self, mock_get_settings):
        """Test default logging setup."""
        mock_settings = MagicMock()
        mock_settings.log_level = 'INFO'
        mock_get_settings.return_value = mock_settings
        
        test_logger = setup_logging()
        
        assert test_logger.name == 'job_tracker'
        assert test_logger.level == logging.INFO
        assert len(test_logger.handlers) >= 1
    
    @patch('utils.get_settings')
    def test_setup_logging_debug(self, mock_get_settings):
        """Test debug logging setup."""
        mock_settings = MagicMock()
        mock_settings.log_level = 'DEBUG'
        mock_get_settings.return_value = mock_settings
        
        test_logger = setup_logging()
        
        assert test_logger.level == logging.DEBUG
    
    @patch('utils.get_settings')
    def test_setup_logging_prevents_duplicate_handlers(self, mock_get_settings):
        """Test that setup_logging prevents duplicate handlers."""
        mock_settings = MagicMock()
        mock_settings.log_level = 'INFO'
        mock_get_settings.return_value = mock_settings
        
        # Call setup_logging twice
        logger1 = setup_logging()
        original_handler_count = len(logger1.handlers)
        
        logger2 = setup_logging()
        
        # Should be same logger instance with same number of handlers
        assert logger1 is logger2
        assert len(logger2.handlers) == original_handler_count


class TestLLMIntegration:
    """Test LLM integration utilities."""
    
    @patch('utils.OpenAI')
    @patch('utils.get_settings')
    def test_call_llm_success(self, mock_get_settings, mock_openai):
        """Test successful LLM API call."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.openai_api_key = 'test_key'
        mock_settings.openai_model = 'gpt-4'
        mock_get_settings.return_value = mock_settings
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Test response'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = call_llm('Test prompt')
        
        assert result == 'Test response'
        mock_client.chat.completions.create.assert_called_once()
        
        # Verify call arguments
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4'
        assert call_args[1]['messages'][0]['content'] == 'Test prompt'
        assert call_args[1]['temperature'] == 0.1
        assert call_args[1]['max_tokens'] == 1000
    
    @patch('utils.OpenAI')
    @patch('utils.get_settings')
    def test_call_llm_with_custom_model(self, mock_get_settings, mock_openai):
        """Test LLM call with custom model."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.openai_api_key = 'test_key'
        mock_settings.openai_model = 'gpt-4'
        mock_get_settings.return_value = mock_settings
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = 'Test response'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = call_llm('Test prompt', model='gpt-3.5-turbo')
        
        assert result == 'Test response'
        
        # Verify custom model was used
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-3.5-turbo'
    
    @patch('utils.OpenAI')
    @patch('utils.get_settings')
    def test_call_llm_api_error(self, mock_get_settings, mock_openai):
        """Test LLM API error handling."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.openai_api_key = 'test_key'
        mock_settings.openai_model = 'gpt-4'
        mock_get_settings.return_value = mock_settings
        
        # Mock OpenAI client to raise error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception('API Error')
        mock_openai.return_value = mock_client
        
        with pytest.raises(Exception) as exc_info:
            call_llm('Test prompt')
        
        assert 'API Error' in str(exc_info.value)


class TestYAMLParsing:
    """Test YAML parsing utilities."""
    
    def test_parse_yaml_response_with_code_fences(self):
        """Test parsing YAML response with code fences."""
        response = """Here's the analysis:
        
        ```yaml
        result: success
        confidence: 0.85
        items:
          - item1
          - item2
        ```
        
        That's the result."""
        
        result = parse_yaml_response(response)
        
        assert result['result'] == 'success'
        assert result['confidence'] == 0.85
        assert result['items'] == ['item1', 'item2']
    
    def test_parse_yaml_response_without_code_fences(self):
        """Test parsing YAML response without code fences."""
        response = """result: success
confidence: 0.85
items:
  - item1
  - item2"""
        
        result = parse_yaml_response(response)
        
        assert result['result'] == 'success'
        assert result['confidence'] == 0.85
        assert result['items'] == ['item1', 'item2']
    
    def test_parse_yaml_response_invalid_yaml(self):
        """Test parsing invalid YAML."""
        response = """```yaml
        invalid: yaml: content: [unclosed
        ```"""
        
        with pytest.raises(ValueError) as exc_info:
            parse_yaml_response(response)
        
        assert 'Invalid YAML' in str(exc_info.value)
    
    def test_parse_yaml_response_non_dict(self):
        """Test parsing YAML that doesn't return a dict."""
        response = """```yaml
        - item1
        - item2
        ```"""
        
        with pytest.raises(ValueError) as exc_info:
            parse_yaml_response(response)
        
        assert 'must be a dictionary' in str(exc_info.value)
    
    def test_parse_yaml_response_empty(self):
        """Test parsing empty YAML response."""
        response = """```yaml
        ```"""
        
        with pytest.raises(ValueError):
            parse_yaml_response(response)


class TestDateTimeUtils:
    """Test date/time utility functions."""
    
    def test_format_datetime(self):
        """Test datetime formatting."""
        dt = datetime(2023, 12, 25, 15, 30, 45)
        result = format_datetime(dt)
        
        assert result == '2023-12-25 15:30:45'
    
    def test_get_date_range(self):
        """Test date range calculation."""
        with patch('utils.datetime') as mock_datetime:
            mock_now = datetime(2023, 12, 25, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            start, end = get_date_range(7)
            
            assert end == mock_now
            assert start == mock_now - timedelta(days=7)
    
    def test_get_date_range_zero_days(self):
        """Test date range with zero days."""
        with patch('utils.datetime') as mock_datetime:
            mock_now = datetime(2023, 12, 25, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            start, end = get_date_range(0)
            
            assert end == mock_now
            assert start == mock_now


class TestEmailUtils:
    """Test email utility functions."""
    
    def test_extract_domain_from_email(self):
        """Test domain extraction from email."""
        assert extract_domain_from_email('user@example.com') == 'example.com'
        assert extract_domain_from_email('test@GOOGLE.COM') == 'google.com'
        assert extract_domain_from_email('user@sub.domain.com') == 'sub.domain.com'
    
    def test_extract_domain_from_email_invalid(self):
        """Test domain extraction from invalid email."""
        assert extract_domain_from_email('invalid-email') == ''
        assert extract_domain_from_email('') == ''
        assert extract_domain_from_email(None) == ''
    
    def test_is_job_related_domain(self):
        """Test job-related domain detection."""
        # Job-related domains
        assert is_job_related_domain('recruiter@noreply.company.com') is True
        assert is_job_related_domain('hr@careers.google.com') is True
        assert is_job_related_domain('jobs@talent.microsoft.com') is True
        assert is_job_related_domain('user@workday.com') is True
        assert is_job_related_domain('team@greenhouse.io') is True
        assert is_job_related_domain('notifications@linkedin.com') is True
        
        # Non-job-related domains
        assert is_job_related_domain('user@gmail.com') is False
        assert is_job_related_domain('friend@yahoo.com') is False
        assert is_job_related_domain('team@company.com') is False
    
    def test_is_job_related_domain_case_insensitive(self):
        """Test job-related domain detection is case insensitive."""
        assert is_job_related_domain('HR@CAREERS.GOOGLE.COM') is True
        assert is_job_related_domain('Jobs@TALENT.MICROSOFT.COM') is True


class TestTextUtils:
    """Test text processing utilities."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  This   is   <b>HTML</b>   text  with   extra   spaces  "
        clean = clean_text(dirty_text)
        
        assert clean == "This is HTML text with extra spaces"
    
    def test_clean_text_with_html(self):
        """Test text cleaning with HTML tags."""
        html_text = "<div>Hello <span>world</span>!</div>"
        clean = clean_text(html_text)
        
        assert clean == "Hello world!"
    
    def test_clean_text_with_special_chars(self):
        """Test text cleaning with special characters."""
        special_text = "Hello@world.com! How are you? #awesome"
        clean = clean_text(special_text)
        
        # Should keep basic punctuation and @ symbol
        assert "@" in clean
        assert "!" in clean
        assert "?" in clean
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "a" * 5000
        truncated = truncate_text(long_text, 100)
        
        assert len(truncated) == 103  # 100 + "..."
        assert truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncation of short text."""
        short_text = "short"
        truncated = truncate_text(short_text, 100)
        
        assert truncated == "short"
        assert not truncated.endswith("...")
    
    def test_truncate_text_exact_length(self):
        """Test truncation at exact length."""
        text = "a" * 100
        truncated = truncate_text(text, 100)
        
        assert truncated == text
        assert not truncated.endswith("...")


class TestCalculationUtils:
    """Test calculation utility functions."""
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        factors = {
            'factor1': 0.8,
            'factor2': 0.9,
            'factor3': 0.7
        }
        
        score = calculate_confidence_score(factors)
        expected = (0.8 + 0.9 + 0.7) / 3
        
        assert score == expected
    
    def test_calculate_confidence_score_empty(self):
        """Test confidence score with empty factors."""
        assert calculate_confidence_score({}) == 0.0
    
    def test_calculate_confidence_score_high_values(self):
        """Test confidence score with high values."""
        factors = {
            'factor1': 2.0,
            'factor2': 1.5
        }
        
        score = calculate_confidence_score(factors)
        
        # Should be capped at 1.0
        assert score == 1.0
    
    def test_calculate_confidence_score_single_factor(self):
        """Test confidence score with single factor."""
        factors = {'factor1': 0.85}
        score = calculate_confidence_score(factors)
        
        assert score == 0.85


class TestSafetyUtils:
    """Test safety utility functions."""
    
    def test_safe_get_success(self):
        """Test safe get with valid dictionary."""
        data = {'key1': 'value1', 'key2': 'value2'}
        
        assert safe_get(data, 'key1') == 'value1'
        assert safe_get(data, 'key2') == 'value2'
    
    def test_safe_get_missing_key(self):
        """Test safe get with missing key."""
        data = {'key1': 'value1'}
        
        assert safe_get(data, 'missing_key') is None
        assert safe_get(data, 'missing_key', 'default') == 'default'
    
    def test_safe_get_none_dict(self):
        """Test safe get with None dictionary."""
        assert safe_get(None, 'key') is None
        assert safe_get(None, 'key', 'default') == 'default'
    
    def test_safe_get_non_dict(self):
        """Test safe get with non-dictionary."""
        assert safe_get('not_a_dict', 'key') is None
        assert safe_get('not_a_dict', 'key', 'default') == 'default'
        assert safe_get(123, 'key') is None
        assert safe_get([], 'key') is None


class TestGlobalLogger:
    """Test global logger instance."""
    
    def test_global_logger_exists(self):
        """Test that global logger exists."""
        assert logger is not None
        assert logger.name == 'job_tracker'
    
    def test_global_logger_logging(self):
        """Test that global logger can log."""
        with patch.object(logger, 'info') as mock_info:
            logger.info('Test message')
            mock_info.assert_called_once_with('Test message')


# Integration tests
class TestUtilsIntegration:
    """Test integration between utility functions."""
    
    def test_email_and_text_utils_integration(self):
        """Test integration between email and text utils."""
        messy_email = "  recruiter@NOREPLY.company.com  "
        domain = extract_domain_from_email(messy_email.strip())
        is_job_related = is_job_related_domain(messy_email)
        
        assert domain == 'noreply.company.com'
        assert is_job_related is True
    
    def test_datetime_and_formatting_integration(self):
        """Test integration between datetime and formatting utils."""
        start, end = get_date_range(7)
        formatted_start = format_datetime(start)
        formatted_end = format_datetime(end)
        
        assert isinstance(formatted_start, str)
        assert isinstance(formatted_end, str)
        assert len(formatted_start) == 19  # YYYY-MM-DD HH:MM:SS
        assert len(formatted_end) == 19
    
    @patch('utils.call_llm')
    def test_llm_and_yaml_integration(self, mock_call_llm):
        """Test integration between LLM and YAML parsing."""
        mock_call_llm.return_value = """```yaml
        result: success
        confidence: 0.85
        ```"""
        
        llm_response = call_llm('Test prompt')
        parsed_result = parse_yaml_response(llm_response)
        
        assert parsed_result['result'] == 'success'
        assert parsed_result['confidence'] == 0.85


if __name__ == "__main__":
    pytest.main([__file__])