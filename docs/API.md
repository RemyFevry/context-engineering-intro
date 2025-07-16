# API Documentation

This document provides detailed API documentation for the Job Application Tracker system.

## Table of Contents

- [Core Components](#core-components)
- [Data Models](#data-models)
- [Agents](#agents)
- [Tools](#tools)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Core Components

### Flow Orchestration

The main workflow orchestrator that coordinates all agents and nodes.

#### `JobApplicationTrackingFlow`

Main flow class that manages the complete job application tracking workflow.

```python
class JobApplicationTrackingFlow:
    def __init__(self) -> None
    async def run_full_workflow(self, days_back: int = None) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, Any]
    def reset_statistics(self) -> None
```

**Methods:**

- `run_full_workflow(days_back: int = None)`: Execute the complete workflow
  - **Parameters**: `days_back` - Number of days back to retrieve emails (default: 30)
  - **Returns**: Dictionary containing workflow results and metadata
  - **Raises**: Exception if workflow fails

- `get_statistics()`: Get workflow execution statistics
  - **Returns**: Dictionary with run counts, success rates, and error information

**Example Usage:**

```python
from flow import JobApplicationTrackingFlow

# Create flow instance
flow = JobApplicationTrackingFlow()

# Run workflow
result = await flow.run_full_workflow(days_back=30)

if result['success']:
    print(f"Found {len(result['results']['job_applications'])} applications")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
else:
    print(f"Workflow failed: {result['error']}")

# Get statistics
stats = flow.get_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
```

#### Factory Functions

```python
def create_job_application_flow() -> JobApplicationTrackingFlow
async def run_job_application_tracking(days_back: int = None) -> Dict[str, Any]
```

## Data Models

### Email Models

#### `GmailEmail`

Represents a Gmail email with metadata and content.

```python
class GmailEmail(BaseModel):
    id: str
    sender: str
    subject: str
    received_date: datetime
    content: str
    thread_id: Optional[str] = None
    labels: List[str] = []
    snippet: Optional[str] = None
    
    def get_domain(self) -> str
    def is_recent(self, days: int) -> bool
    def word_count(self) -> int
```

**Methods:**

- `get_domain()`: Extract domain from sender email
- `is_recent(days: int)`: Check if email is within specified days
- `word_count()`: Count words in email content

**Example:**

```python
from models.email_models import GmailEmail
from datetime import datetime, timezone

email = GmailEmail(
    id="email_123",
    sender="hr@google.com",
    subject="Application Status Update",
    received_date=datetime.now(timezone.utc),
    content="Thank you for your application..."
)

print(f"Domain: {email.get_domain()}")  # "google.com"
print(f"Recent: {email.is_recent(7)}")  # True/False
print(f"Words: {email.word_count()}")   # 5
```

#### `EmailBatch`

Container for multiple emails with batch metadata.

```python
class EmailBatch(BaseModel):
    batch_id: str
    emails: List[GmailEmail]
    created_at: datetime
    source: str = "gmail"
    
    def total_emails(self) -> int
    def recent_emails(self, days: int) -> int
    def get_date_range(self) -> tuple[Optional[datetime], Optional[datetime]]
```

### Job Application Models

#### `JobApplication`

Represents a job application extracted from emails.

```python
class JobApplication(BaseModel):
    email_id: str
    company: str
    position: str
    status: ApplicationStatus
    applied_date: datetime
    last_updated: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)
    job_url: Optional[str] = None
    application_method: Optional[str] = None
    notes: Optional[str] = None
    
    def get_days_since_applied(self) -> int
    def get_days_since_updated(self) -> int
    def is_stale(self, days: int) -> bool
    def to_dict(self) -> Dict[str, Any]
```

**Enums:**

```python
class ApplicationStatus(Enum):
    APPLIED = "applied"
    ACKNOWLEDGED = "acknowledged"
    INTERVIEWING = "interviewing"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"
```

#### `ApplicationSummary`

Statistical summary of job applications.

```python
class ApplicationSummary(BaseModel):
    total_applications: int = Field(ge=0)
    response_rate: float = Field(ge=0.0, le=1.0)
    interview_rate: float = Field(ge=0.0, le=1.0)
    offer_rate: float = Field(ge=0.0, le=1.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)
    status_counts: Dict[ApplicationStatus, int] = {}
    average_response_time: float = 0.0
    stale_applications: int = 0
```

### Research Models

#### `JobPosting`

Represents a job posting found through research.

```python
class JobPosting(BaseModel):
    title: str
    company: str
    url: str = Field(..., regex=r'^https?://')
    location: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    posted_date: Optional[datetime] = None
    description: Optional[str] = None
    salary_range: Optional[str] = None
    
    def is_recent(self, days: int) -> bool
```

#### `ResearchResult`

Contains research results for a specific job application.

```python
class ResearchResult(BaseModel):
    target_company: str
    target_position: str
    job_postings: List[JobPosting]
    total_results: int = Field(ge=0)
    average_relevance: float = Field(ge=0.0, le=1.0)
    search_timestamp: datetime
    
    @property
    def best_match(self) -> Optional[JobPosting]
```

## Agents

### Email Agent

Handles Gmail API integration and email processing.

#### `EmailAgent`

```python
class EmailAgent:
    def __init__(self) -> None
    async def initialize(self) -> None
    async def retrieve_recent_emails(self, days_back: int) -> EmailBatch
    async def filter_job_related_emails(self, email_batch: EmailBatch) -> List[GmailEmail]
    async def preprocess_emails(self, emails: List[GmailEmail]) -> List[GmailEmail]
    def get_statistics(self) -> Dict[str, Any]
```

**Methods:**

- `initialize()`: Set up Gmail API authentication
- `retrieve_recent_emails(days_back: int)`: Retrieve emails from specified time period
- `filter_job_related_emails(email_batch: EmailBatch)`: Filter emails for job-related content
- `preprocess_emails(emails: List[GmailEmail])`: Clean and preprocess email content

**Example:**

```python
from agents.email_agent import create_email_agent

agent = create_email_agent()
await agent.initialize()

# Retrieve emails
batch = await agent.retrieve_recent_emails(days_back=30)
print(f"Retrieved {len(batch.emails)} emails")

# Filter job-related emails
job_emails = await agent.filter_job_related_emails(batch)
print(f"Found {len(job_emails)} job-related emails")
```

### Classification Agent

Uses LLM to classify emails and extract job application information.

#### `ClassificationAgent`

```python
class ClassificationAgent:
    def __init__(self) -> None
    async def classify_email(self, email: GmailEmail) -> Optional[Dict[str, Any]]
    async def classify_email_batch(self, emails: List[GmailEmail]) -> List[Dict[str, Any]]
    async def create_job_applications(self, job_infos: List[Dict[str, Any]]) -> List[JobApplication]
    async def validate_classification_accuracy(self, job_infos: List[Dict[str, Any]], sample_size: int) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, Any]
```

**Methods:**

- `classify_email(email: GmailEmail)`: Classify single email using LLM
- `classify_email_batch(emails: List[GmailEmail])`: Batch classify multiple emails
- `create_job_applications(job_infos: List[Dict[str, Any]])`: Create JobApplication objects from classifications
- `validate_classification_accuracy(job_infos: List[Dict[str, Any]], sample_size: int)`: Validate classification accuracy

### Research Agent

Searches for job postings and company information.

#### `ResearchAgent`

```python
class ResearchAgent:
    def __init__(self) -> None
    async def research_job_application(self, application: JobApplication) -> ResearchResult
    async def research_batch(self, applications: List[JobApplication]) -> List[ResearchResult]
    async def get_company_insights(self, company: str) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, Any]
```

### Status Agent

Generates reports and analytics on job applications.

#### `StatusAgent`

```python
class StatusAgent:
    def __init__(self) -> None
    def generate_status_summary(self, applications: List[JobApplication]) -> ApplicationSummary
    def generate_detailed_report(self, applications: List[JobApplication], research_results: List[ResearchResult] = None) -> Dict[str, Any]
    async def generate_insights(self, applications: List[JobApplication]) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, Any]
```

## Tools

### Gmail Tool

Low-level Gmail API integration.

#### `GmailTool`

```python
class GmailTool:
    def __init__(self, credentials_path: str, token_path: str) -> None
    async def authenticate(self) -> None
    async def get_recent_emails(self, days_back: int = None) -> List[GmailEmail]
    async def search_emails(self, query: str, max_results: int = None) -> List[GmailEmail]
    def get_statistics(self) -> Dict[str, Any]
```

### Brave Search Tool

Brave Search API integration for job research.

#### `BraveSearchTool`

```python
class BraveSearchTool:
    def __init__(self, api_key: str) -> None
    async def search_job_postings(self, query: str, max_results: int = None) -> List[JobPosting]
    async def search_company_info(self, company: str) -> Dict[str, Any]
    def get_statistics(self) -> Dict[str, Any]
```

## Configuration

### Settings

Configuration management using Pydantic settings.

#### `Settings`

```python
class Settings(BaseSettings):
    # Gmail configuration
    gmail_credentials_path: str = Field(default="./credentials/credentials.json")
    gmail_token_path: str = Field(default="./credentials/token.json")
    
    # API keys
    brave_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    
    # Processing configuration
    email_days_back: int = Field(default=30, ge=1, le=365)
    classification_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    research_max_results: int = Field(default=10, ge=1, le=50)
    rate_limit_delay: float = Field(default=1.0, ge=0.0)
    
    # LLM configuration
    openai_model: str = Field(default="gpt-4")
    
    # System configuration
    log_level: str = Field(default="INFO")
    enable_validation: bool = Field(default=True)
```

**Factory Function:**

```python
def get_settings() -> Settings
```

## Utilities

### Core Utilities

#### Logging

```python
def setup_logging() -> logging.Logger
```

#### LLM Integration

```python
def call_llm(prompt: str, model: Optional[str] = None) -> str
def parse_yaml_response(response: str) -> Dict[str, Any]
```

#### Date/Time Utilities

```python
def format_datetime(dt: datetime) -> str
def get_date_range(days_back: int) -> tuple[datetime, datetime]
```

#### Text Processing

```python
def clean_text(text: str) -> str
def truncate_text(text: str, max_length: int = 2000) -> str
def extract_domain_from_email(email: str) -> str
```

#### Data Processing

```python
def calculate_confidence_score(factors: Dict[str, float]) -> float
def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any
def is_job_related_domain(email: str) -> bool
```

### Error Handling

All API methods use consistent error handling patterns:

```python
try:
    result = await some_operation()
    return {'success': True, 'data': result}
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return {'success': False, 'error': str(e)}
```

### Rate Limiting

Rate limiting is implemented across all external API calls:

```python
# Example with rate limiting
async def rate_limited_operation():
    await asyncio.sleep(settings.rate_limit_delay)
    return await api_call()
```

## Integration Examples

### Custom Agent Integration

```python
from agents.base_agent import BaseAgent
from models.job_models import JobApplication

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.custom_config = {}
    
    async def process_application(self, application: JobApplication) -> Dict[str, Any]:
        # Custom processing logic
        return {'processed': True, 'data': application}
    
    def get_statistics(self) -> Dict[str, Any]:
        return {'processed_count': self.stats.get('processed', 0)}
```

### Custom Node Integration

```python
from pocketflow import AsyncNode
from typing import Dict, Any

class CustomProcessingNode(AsyncNode):
    def __init__(self, node_id: str = "custom_processing"):
        super().__init__()
        self.node_id = node_id
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        # Preparation logic
        return {'ready': True, 'data': shared.get('input_data', [])}
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Processing logic
        if not inputs.get('ready'):
            return {'success': False, 'error': 'Not ready'}
        
        # Custom processing
        result = self.custom_process(inputs['data'])
        return {'success': True, 'result': result}
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        # Post-processing logic
        if exec_res.get('success'):
            shared['custom_result'] = exec_res['result']
            return "success"
        return "error"
```

## Best Practices

### Error Handling

Always handle errors gracefully and provide meaningful error messages:

```python
try:
    result = await risky_operation()
    if not result['success']:
        logger.warning(f"Operation failed: {result['error']}")
        return default_value
    return result['data']
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Resource Management

Use context managers for external resources:

```python
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.json()
        return data
```

### Configuration

Always use the settings system for configuration:

```python
from config.settings import get_settings

settings = get_settings()
max_retries = settings.research_max_results
timeout = settings.rate_limit_delay
```

### Testing

Write comprehensive tests for custom components:

```python
import pytest
from unittest.mock import Mock, patch
from your_custom_agent import CustomAgent

class TestCustomAgent:
    @pytest.fixture
    def agent(self):
        return CustomAgent()
    
    @pytest.mark.asyncio
    async def test_process_application(self, agent):
        # Test implementation
        pass
```