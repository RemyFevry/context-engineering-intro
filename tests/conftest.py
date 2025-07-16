"""
Pytest configuration and fixtures for job application tracking tests.
"""

import pytest
import os
import tempfile
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Import models and components for fixtures
from models.email_models import GmailEmail, EmailBatch
from models.job_models import JobApplication, ApplicationStatus
from models.research_models import JobPosting, ResearchResult
from config.settings import Settings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch.dict(os.environ, {
        'BRAVE_API_KEY': 'test_brave_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'EMAIL_DAYS_BACK': '30',
        'CLASSIFICATION_CONFIDENCE_THRESHOLD': '0.7',
        'RESEARCH_MAX_RESULTS': '10',
        'RATE_LIMIT_DELAY': '1.0',
        'OPENAI_MODEL': 'gpt-4',
        'LOG_LEVEL': 'INFO',
        'ENABLE_VALIDATION': 'true'
    }):
        yield Settings()


@pytest.fixture
def mock_gmail_credentials(temp_dir):
    """Mock Gmail credentials file."""
    credentials_path = os.path.join(temp_dir, 'credentials.json')
    
    credentials_data = {
        "installed": {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"]
        }
    }
    
    with open(credentials_path, 'w') as f:
        json.dump(credentials_data, f)
    
    yield credentials_path


@pytest.fixture
def sample_emails():
    """Sample email data for testing."""
    return [
        GmailEmail(
            id="email_1",
            sender="hr@google.com",
            subject="Application Status Update",
            received_date=datetime.now(timezone.utc) - timedelta(days=1),
            content="Thank you for your application to the Software Engineer position.",
            thread_id="thread_1",
            labels=["INBOX", "UNREAD"]
        ),
        GmailEmail(
            id="email_2",
            sender="recruiter@microsoft.com",
            subject="Interview Invitation",
            received_date=datetime.now(timezone.utc) - timedelta(days=2),
            content="We would like to invite you for an interview for the Senior Developer role.",
            thread_id="thread_2",
            labels=["INBOX"]
        ),
        GmailEmail(
            id="email_3",
            sender="noreply@amazon.com",
            subject="Application Confirmation",
            received_date=datetime.now(timezone.utc) - timedelta(days=3),
            content="Your application for the Cloud Engineer position has been received.",
            thread_id="thread_3",
            labels=["INBOX"]
        ),
        GmailEmail(
            id="email_4",
            sender="friend@gmail.com",
            subject="Weekend Plans",
            received_date=datetime.now(timezone.utc) - timedelta(days=1),
            content="What are your plans for the weekend?",
            thread_id="thread_4",
            labels=["INBOX"]
        ),
        GmailEmail(
            id="email_5",
            sender="no-reply@linkedin.com",
            subject="Your application was viewed",
            received_date=datetime.now(timezone.utc) - timedelta(days=5),
            content="Good news! Your application for the Data Scientist position was viewed.",
            thread_id="thread_5",
            labels=["INBOX"]
        )
    ]


@pytest.fixture
def sample_job_emails(sample_emails):
    """Sample job-related emails for testing."""
    return [email for email in sample_emails if email.id != "email_4"]  # Exclude personal email


@pytest.fixture
def sample_email_batch(sample_emails):
    """Sample email batch for testing."""
    return EmailBatch(
        batch_id="test_batch_123",
        emails=sample_emails,
        created_at=datetime.now(timezone.utc),
        source="gmail"
    )


@pytest.fixture
def sample_job_applications():
    """Sample job applications for testing."""
    return [
        JobApplication(
            email_id="email_1",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=5),
            last_updated=datetime.now(timezone.utc) - timedelta(days=1),
            confidence_score=0.85,
            job_url="https://careers.google.com/jobs/123",
            application_method="online"
        ),
        JobApplication(
            email_id="email_2",
            company="Microsoft",
            position="Senior Developer",
            status=ApplicationStatus.INTERVIEWING,
            applied_date=datetime.now(timezone.utc) - timedelta(days=10),
            last_updated=datetime.now(timezone.utc) - timedelta(days=2),
            confidence_score=0.90,
            job_url="https://careers.microsoft.com/jobs/456",
            application_method="online"
        ),
        JobApplication(
            email_id="email_3",
            company="Amazon",
            position="Cloud Engineer",
            status=ApplicationStatus.ACKNOWLEDGED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=15),
            last_updated=datetime.now(timezone.utc) - timedelta(days=3),
            confidence_score=0.75,
            job_url="https://amazon.jobs/jobs/789",
            application_method="online"
        ),
        JobApplication(
            email_id="email_5",
            company="LinkedIn",
            position="Data Scientist",
            status=ApplicationStatus.REJECTED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=20),
            last_updated=datetime.now(timezone.utc) - timedelta(days=5),
            confidence_score=0.80,
            job_url="https://linkedin.com/jobs/101112",
            application_method="online"
        )
    ]


@pytest.fixture
def sample_job_postings():
    """Sample job postings for testing."""
    return [
        JobPosting(
            title="Software Engineer",
            company="Google",
            url="https://careers.google.com/jobs/123",
            location="Mountain View, CA",
            relevance_score=0.85,
            posted_date=datetime.now(timezone.utc) - timedelta(days=2),
            description="Join our team as a Software Engineer..."
        ),
        JobPosting(
            title="Senior Software Engineer",
            company="Google",
            url="https://careers.google.com/jobs/456",
            location="San Francisco, CA",
            relevance_score=0.92,
            posted_date=datetime.now(timezone.utc) - timedelta(days=1),
            description="Lead our engineering team..."
        ),
        JobPosting(
            title="Cloud Engineer",
            company="Amazon",
            url="https://amazon.jobs/jobs/789",
            location="Seattle, WA",
            relevance_score=0.78,
            posted_date=datetime.now(timezone.utc) - timedelta(days=3),
            description="Build scalable cloud solutions..."
        )
    ]


@pytest.fixture
def sample_research_results(sample_job_postings):
    """Sample research results for testing."""
    return [
        ResearchResult(
            target_company="Google",
            target_position="Software Engineer",
            job_postings=sample_job_postings[:2],  # First 2 Google postings
            total_results=2,
            average_relevance=0.885,
            search_timestamp=datetime.now(timezone.utc)
        ),
        ResearchResult(
            target_company="Amazon",
            target_position="Cloud Engineer",
            job_postings=sample_job_postings[2:3],  # Amazon posting
            total_results=1,
            average_relevance=0.78,
            search_timestamp=datetime.now(timezone.utc)
        )
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """```yaml
classification: job_application
confidence: 0.85
company: Google
position: Software Engineer
status: applied
```"""
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('utils.OpenAI', return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_gmail_service():
    """Mock Gmail service for testing."""
    mock_service = MagicMock()
    
    # Mock messages list
    mock_service.users().messages().list().execute.return_value = {
        'messages': [
            {'id': 'email_1', 'threadId': 'thread_1'},
            {'id': 'email_2', 'threadId': 'thread_2'}
        ]
    }
    
    # Mock message get
    def mock_get_message(userId, id):
        if id == 'email_1':
            return MagicMock(execute=lambda: {
                'id': 'email_1',
                'threadId': 'thread_1',
                'labelIds': ['INBOX', 'UNREAD'],
                'snippet': 'Thank you for your application...',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'hr@google.com'},
                        {'name': 'Subject', 'value': 'Application Status Update'},
                        {'name': 'Date', 'value': 'Mon, 1 Jan 2024 12:00:00 +0000'}
                    ],
                    'body': {
                        'data': 'VGhhbmsgZm9yIHlvdXIgYXBwbGljYXRpb24='  # Base64 encoded
                    }
                }
            })
        elif id == 'email_2':
            return MagicMock(execute=lambda: {
                'id': 'email_2',
                'threadId': 'thread_2',
                'labelIds': ['INBOX'],
                'snippet': 'Interview invitation...',
                'payload': {
                    'headers': [
                        {'name': 'From', 'value': 'recruiter@microsoft.com'},
                        {'name': 'Subject', 'value': 'Interview Invitation'},
                        {'name': 'Date', 'value': 'Sun, 31 Dec 2023 12:00:00 +0000'}
                    ],
                    'body': {
                        'data': 'SW50ZXJ2aWV3IGludml0YXRpb24='  # Base64 encoded
                    }
                }
            })
    
    mock_service.users().messages().get = mock_get_message
    
    yield mock_service


@pytest.fixture
def mock_brave_api():
    """Mock Brave Search API for testing."""
    mock_response = {
        'web': {
            'results': [
                {
                    'title': 'Software Engineer - Google Careers',
                    'url': 'https://careers.google.com/jobs/123',
                    'description': 'Join our team as a Software Engineer...',
                    'displayUrl': 'careers.google.com'
                },
                {
                    'title': 'Senior Software Engineer - Google',
                    'url': 'https://careers.google.com/jobs/456',
                    'description': 'Lead our engineering team...',
                    'displayUrl': 'careers.google.com'
                }
            ]
        }
    }
    
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response_obj = MagicMock()
        mock_response_obj.status = 200
        mock_response_obj.json.return_value = mock_response
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response_obj
        
        yield mock_response


@pytest.fixture
def mock_shared_state():
    """Mock shared state for workflow testing."""
    return {
        'workflow_id': 'test_workflow_123',
        'workflow_started': datetime.now(timezone.utc),
        'days_back': 30,
        'email_batch': None,
        'total_emails_retrieved': 0,
        'job_related_emails': [],
        'total_job_emails': 0,
        'job_applications_info': [],
        'total_classified': 0,
        'job_applications': [],
        'total_job_applications': 0,
        'research_results': [],
        'total_researched': 0,
        'company_insights': {},
        'aggregated_research_insights': {},
        'status_report': {},
        'ai_insights': {},
        'final_report': {},
        'email_retrieval_completed': False,
        'email_filtering_completed': False,
        'email_preprocessing_completed': False,
        'classification_completed': False,
        'job_application_creation_completed': False,
        'research_completed': False,
        'company_insights_completed': False,
        'research_aggregation_completed': False,
        'status_reporting_completed': False,
        'insights_generation_completed': False,
        'final_report_completed': False,
        'workflow_completed': False
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        'BRAVE_API_KEY': 'test_brave_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'GMAIL_CREDENTIALS_PATH': './test_credentials.json',
        'GMAIL_TOKEN_PATH': './test_token.json',
        'EMAIL_DAYS_BACK': '30',
        'CLASSIFICATION_CONFIDENCE_THRESHOLD': '0.7',
        'RESEARCH_MAX_RESULTS': '10',
        'RATE_LIMIT_DELAY': '1.0',
        'OPENAI_MODEL': 'gpt-4',
        'LOG_LEVEL': 'INFO',
        'ENABLE_VALIDATION': 'true'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(autouse=True)
def cleanup_settings_singleton():
    """Clean up settings singleton after each test."""
    yield
    # Clear the settings singleton if it exists
    if hasattr(Settings, '_instance'):
        delattr(Settings, '_instance')


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch('utils.logger') as mock_logger:
        yield mock_logger


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "api: marks tests that require API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        if "test_" in item.nodeid and "integration" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to slow tests
        if "slow" in item.nodeid or "full_workflow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add api marker to tests that use external APIs
        if any(keyword in item.nodeid for keyword in ["api", "gmail", "brave", "openai"]):
            item.add_marker(pytest.mark.api)


# Custom assertions for testing
def assert_valid_email(email):
    """Assert that an email object is valid."""
    assert hasattr(email, 'id')
    assert hasattr(email, 'sender')
    assert hasattr(email, 'subject')
    assert hasattr(email, 'received_date')
    assert hasattr(email, 'content')
    assert email.id is not None
    assert email.sender is not None
    assert email.subject is not None
    assert email.received_date is not None


def assert_valid_job_application(application):
    """Assert that a job application object is valid."""
    assert hasattr(application, 'email_id')
    assert hasattr(application, 'company')
    assert hasattr(application, 'position')
    assert hasattr(application, 'status')
    assert hasattr(application, 'applied_date')
    assert hasattr(application, 'confidence_score')
    assert application.email_id is not None
    assert application.company is not None
    assert application.position is not None
    assert application.status is not None
    assert application.applied_date is not None
    assert 0.0 <= application.confidence_score <= 1.0


def assert_valid_research_result(result):
    """Assert that a research result object is valid."""
    assert hasattr(result, 'target_company')
    assert hasattr(result, 'target_position')
    assert hasattr(result, 'job_postings')
    assert hasattr(result, 'total_results')
    assert hasattr(result, 'average_relevance')
    assert result.target_company is not None
    assert result.target_position is not None
    assert isinstance(result.job_postings, list)
    assert result.total_results >= 0
    assert 0.0 <= result.average_relevance <= 1.0


# Make custom assertions available
__all__ = [
    'assert_valid_email',
    'assert_valid_job_application', 
    'assert_valid_research_result'
]