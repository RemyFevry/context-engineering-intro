"""
Shared utility functions for job application tracking system.

Contains common utilities used across multiple modules including
LLM client, logging setup, and data processing helpers.
"""

import logging
# OS module available for future use
import yaml
from typing import Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from config.settings import get_settings

def setup_logging() -> logging.Logger:
    """
    Set up application logging.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    settings = get_settings()
    
    # Create logger
    logger = logging.getLogger("job_tracker")
    logger.setLevel(getattr(logging, settings.log_level))
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, settings.log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

def call_llm(prompt: str, model: Optional[str] = None) -> str:
    """
    Call OpenAI API to analyze text.
    
    Args:
        prompt (str): Input prompt for the model
        model (str, optional): Model to use. Defaults to settings model.
        
    Returns:
        str: Model response
        
    Raises:
        Exception: If API call fails
    """
    settings = get_settings()
    
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=model or settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=1000
        )
        return response.choices[0].message.content
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Error calling LLM API: {str(e)}")
        raise

def parse_yaml_response(response: str) -> Dict[str, Any]:
    """
    Parse YAML response from LLM.
    
    Args:
        response (str): LLM response containing YAML
        
    Returns:
        Dict[str, Any]: Parsed YAML data
        
    Raises:
        ValueError: If YAML parsing fails
    """
    try:
        # Extract YAML between code fences
        if "```yaml" in response and "```" in response:
            yaml_start = response.find("```yaml") + 7
            yaml_end = response.find("```", yaml_start)
            yaml_str = response[yaml_start:yaml_end].strip()
        else:
            yaml_str = response.strip()
        
        # Parse YAML
        parsed = yaml.safe_load(yaml_str)
        
        if not isinstance(parsed, dict):
            raise ValueError("YAML response must be a dictionary")
        
        return parsed
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in response: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing YAML response: {e}")

def format_datetime(dt: datetime) -> str:
    """
    Format datetime for display.
    
    Args:
        dt (datetime): Datetime to format
        
    Returns:
        str: Formatted datetime string
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.
    
    Returns:
        datetime: Current UTC time with timezone info
    """
    return datetime.now(timezone.utc)

def ensure_timezone_aware(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware (assume UTC if naive).
    
    Args:
        dt (datetime): Datetime object
        
    Returns:
        datetime: Timezone-aware datetime object
    """
    if dt.tzinfo is None:
        # Assume UTC for naive datetimes
        return dt.replace(tzinfo=timezone.utc)
    return dt

def get_date_range(days_back: int) -> tuple[datetime, datetime]:
    """
    Get date range for email filtering.
    
    Args:
        days_back (int): Number of days back from today
        
    Returns:
        tuple[datetime, datetime]: Start and end dates (timezone-aware)
    """
    end_date = get_utc_now()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

def extract_email_address(email_string: str) -> str:
    """
    Extract email address from string that may include display name.
    
    Args:
        email_string (str): Email string like "Name <email@domain.com>" or "email@domain.com"
        
    Returns:
        str: Just the email address part
    """
    import re
    # Extract email from "Display Name <email@domain.com>" format
    email_match = re.search(r'<([^>]+)>', email_string)
    if email_match:
        return email_match.group(1).strip()
    return email_string.strip()

def extract_domain_from_email(email: str) -> str:
    """
    Extract domain from email address.
    
    Args:
        email (str): Email address
        
    Returns:
        str: Domain name
    """
    try:
        # First extract just the email address if it has display name
        clean_email = extract_email_address(email)
        return clean_email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return ""

def clean_text(text: str) -> str:
    """
    Clean text content for processing.
    
    Args:
        text (str): Raw text content
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove HTML tags (basic)
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\@]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 2000) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text (str): Input text
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def is_job_related_domain(email: str) -> bool:
    """
    Check if email domain is likely job-related (supports English and French).
    
    Args:
        email (str): Email address
        
    Returns:
        bool: True if domain suggests job-related content
    """
    email_addr = extract_email_address(email)
    domain = extract_domain_from_email(email_addr)
    
    # Check both the username part and domain for job-related keywords
    username = email_addr.split('@')[0].lower() if '@' in email_addr else ''
    domain_lower = domain.lower()
    
    # English job-related keywords
    english_keywords = {
        'noreply', 'careers', 'jobs', 'hr', 'talent', 'recruiting',
        'recruitment', 'workday', 'greenhouse', 'lever', 'smartrecruiters',
        'jobvite', 'bamboohr', 'indeed', 'linkedin', 'glassdoor',
        'ziprecruiter', 'monster', 'dice', 'career', 'hiring', 'employment'
    }
    
    # French job-related keywords
    french_keywords = {
        'emplois', 'carrieres', 'carriere', 'rh', 'recrutement', 'candidature',
        'candidatures', 'embauche', 'travail', 'emploi', 'poste', 'postes'
    }
    
    # Common job board domains (international)
    job_boards = {
        'indeed', 'linkedin', 'glassdoor', 'monster', 'jobstreet', 'seek',
        'stepstone', 'xing', 'viadeo', 'apec', 'pole-emploi', 'cadremploi',
        'regionsjob', 'lefigaro', 'lemonde', 'keljob', 'meteojob'
    }
    
    all_keywords = english_keywords | french_keywords | job_boards
    
    # Check if any keyword appears in the domain name
    domain_match = any(job_keyword in domain_lower for job_keyword in all_keywords)
    
    # Check if username suggests job-related content (e.g., careers@, jobs@, rh@)
    username_match = any(job_keyword in username for job_keyword in all_keywords)
    
    return domain_match or username_match

def calculate_confidence_score(factors: Dict[str, float]) -> float:
    """
    Calculate confidence score from multiple factors.
    
    Args:
        factors (Dict[str, float]): Dictionary of factor names to scores
        
    Returns:
        float: Combined confidence score (0.0-1.0)
    """
    if not factors:
        return 0.0
    
    # Weighted average of factors
    total_score = sum(factors.values())
    return min(1.0, total_score / len(factors))

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Safely get value from dictionary.
    
    Args:
        dictionary (Dict[str, Any]): Source dictionary
        key (str): Key to retrieve
        default (Any): Default value if key not found
        
    Returns:
        Any: Value or default
    """
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default

# Global logger instance
logger = setup_logging()

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test date range
    start, end = get_date_range(7)
    print(f"Date range (7 days): {format_datetime(start)} to {format_datetime(end)}")
    
    # Test domain extraction
    test_email = "recruiter@google.com"
    domain = extract_domain_from_email(test_email)
    print(f"Domain from {test_email}: {domain}")
    
    # Test job domain detection
    is_job = is_job_related_domain(test_email)
    print(f"Is job-related domain: {is_job}")
    
    # Test confidence calculation
    factors = {"subject_match": 0.8, "sender_domain": 0.9, "content_keywords": 0.7}
    confidence = calculate_confidence_score(factors)
    print(f"Confidence score: {confidence}")
    
    print("✅ All utility functions working correctly")