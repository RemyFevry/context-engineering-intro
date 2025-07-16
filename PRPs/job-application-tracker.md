name: "Job Application Tracker: PocketFlow Multi-Agent Email Analysis System"
description: |

## Purpose
Build a comprehensive PocketFlow-based multi-agent system that retrieves emails from Gmail, filters for job applications, tracks application status, and researches job postings on LinkedIn/Glassdoor. Features CLI interface for user interaction and automated status tracking.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready multi-agent job application tracking system that automatically processes Gmail emails, identifies job applications, tracks their status, and researches related job postings. The system should provide a CLI interface for user interaction and generate comprehensive status reports.

## Why
- **Business value**: Automates job application tracking and reduces manual work
- **Integration**: Demonstrates PocketFlow multi-agent patterns with real-world APIs
- **Problems solved**: Eliminates manual email scanning and job application status tracking
- **User impact**: Provides comprehensive view of job search progress with enriched data

## What
A CLI-based PocketFlow application where:
- Email Agent retrieves and filters Gmail emails from the past month
- Classification Agent identifies job application emails by title/content
- Research Agent searches for related job postings on LinkedIn/Glassdoor
- Status Agent tracks application states and generates summary reports
- CLI provides interactive commands and real-time status updates

### Success Criteria
- [ ] Email Agent successfully retrieves Gmail emails with OAuth2 authentication
- [ ] Classification Agent accurately identifies job application emails
- [ ] Research Agent finds related job postings using Brave Search API
- [ ] Status Agent generates comprehensive summary tables
- [ ] CLI provides intuitive commands for all system operations
- [ ] All tests pass and code meets quality standards
- [ ] System handles rate limits and API errors gracefully

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://the-pocket.github.io/PocketFlow/
  why: Core PocketFlow concepts, Node/Flow patterns, multi-agent architectures
  
- url: https://developers.google.com/gmail/api/guides/filtering
  why: Gmail API email filtering, date ranges, label management
  
- url: https://developers.google.com/gmail/api/auth/scopes
  why: Gmail API scopes, OAuth2 authentication patterns
  
- url: https://brave.com/search/api/
  why: Brave Search API authentication, endpoints, response formats
  
- file: examples/pocketflow-multi-agent/main.py
  why: AsyncNode patterns, multi-agent communication via queues
  
- file: examples/pocketflow-multi-agent/utils.py
  why: LLM wrapper patterns, OpenAI integration
  
- file: examples/pocketflow-tool-search/
  why: Tool integration patterns, search API usage, CLI structure
  
- file: examples/pocketflow-map-reduce/
  why: Map-Reduce patterns for batch processing, data aggregation
  
- url: https://github.com/googleworkspace/python-samples/tree/main/gmail/quickstart
  why: Gmail API Python authentication setup and examples
```

### Current Codebase tree
```bash
.
├── examples/
│   ├── pocketflow-multi-agent/
│   │   ├── main.py              # AsyncNode multi-agent patterns
│   │   ├── utils.py             # LLM wrapper patterns
│   │   └── requirements.txt     # Dependencies
│   ├── pocketflow-tool-search/
│   │   ├── main.py              # CLI interface patterns
│   │   ├── nodes.py             # Node implementation patterns
│   │   ├── flow.py              # Flow configuration patterns
│   │   ├── tools/
│   │   │   ├── search.py        # External API integration
│   │   │   └── parser.py        # LLM analysis patterns
│   │   └── utils/
│   │       └── call_llm.py      # LLM client patterns
│   └── pocketflow-map-reduce/
│       ├── main.py              # Map-Reduce workflow
│       ├── nodes.py             # Batch processing patterns
│       └── utils.py             # Utility functions
├── PRPs/
│   ├── templates/prp_base.md    # PRP template structure
│   └── EXAMPLE_multi_agent_prp.md # Multi-agent example
├── CLAUDE.md                    # Global project rules
├── INITIAL.md                   # Feature requirements
└── README.md                    # Project documentation
```

### Desired Codebase tree with files to be added
```bash
.
├── agents/
│   ├── __init__.py              # Package initialization
│   ├── email_agent.py           # Gmail email retrieval and filtering
│   ├── classification_agent.py  # Job application email classification
│   ├── research_agent.py        # Job posting research via Brave Search
│   └── status_agent.py          # Status tracking and report generation
├── nodes/
│   ├── __init__.py              # Package initialization
│   ├── email_nodes.py           # PocketFlow nodes for email processing
│   ├── classification_nodes.py  # PocketFlow nodes for email classification
│   ├── research_nodes.py        # PocketFlow nodes for research operations
│   └── status_nodes.py          # PocketFlow nodes for status tracking
├── tools/
│   ├── __init__.py              # Package initialization
│   ├── gmail_tool.py            # Gmail API integration
│   ├── brave_search_tool.py     # Brave Search API integration
│   └── data_processor.py        # Data processing utilities
├── models/
│   ├── __init__.py              # Package initialization
│   ├── email_models.py          # Email data models
│   ├── job_models.py            # Job application data models
│   └── research_models.py       # Research result data models
├── config/
│   ├── __init__.py              # Package initialization
│   └── settings.py              # Environment configuration
├── tests/
│   ├── __init__.py              # Package initialization
│   ├── test_email_agent.py      # Email agent tests
│   ├── test_classification_agent.py # Classification agent tests
│   ├── test_research_agent.py    # Research agent tests
│   ├── test_status_agent.py      # Status agent tests
│   ├── test_gmail_tool.py        # Gmail tool tests
│   ├── test_brave_search_tool.py # Brave search tool tests
│   └── fixtures/                # Test fixtures and mock data
├── flow.py                      # Main PocketFlow configuration
├── main.py                      # CLI interface and entry point
├── utils.py                     # Shared utility functions
├── .env.example                 # Environment variables template
├── requirements.txt             # Updated dependencies
├── README.md                    # Comprehensive documentation
└── credentials/.gitkeep         # Directory for OAuth credentials
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: PocketFlow requires careful shared state management between nodes
# CRITICAL: Gmail API requires OAuth2 flow - credentials.json needed for first run
# CRITICAL: Gmail API has quotas - 1 billion quota units per day, 250 quota units per user per second
# CRITICAL: Brave API free tier has 2000 requests/month limit
# CRITICAL: Email filtering requires specific Gmail query syntax (from:, subject:, after:)
# CRITICAL: PocketFlow AsyncNode requires proper asyncio queue management
# CRITICAL: Gmail API returns RFC 2822 formatted emails - need proper parsing
# CRITICAL: Job classification requires robust NLP - consider false positives/negatives
# CRITICAL: Rate limiting crucial for both Gmail and Brave APIs
# CRITICAL: Store sensitive credentials in .env, never commit them
# CRITICAL: Email content can be HTML, plain text, or multipart - handle all formats
```

## Implementation Blueprint

### Data models and structure

```python
# models/email_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EmailSource(str, Enum):
    GMAIL = "gmail"
    OUTLOOK = "outlook"

class EmailFormat(str, Enum):
    HTML = "html"
    PLAIN = "plain"
    MULTIPART = "multipart"

class GmailEmail(BaseModel):
    id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field(..., description="Gmail thread ID")
    sender: str = Field(..., description="Email sender address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    received_date: datetime = Field(..., description="Date email was received")
    labels: List[str] = Field(default_factory=list, description="Gmail labels")
    format: EmailFormat = Field(..., description="Email content format")
    raw_headers: Dict[str, str] = Field(default_factory=dict)

# models/job_models.py
class ApplicationStatus(str, Enum):
    APPLIED = "applied"
    ACKNOWLEDGED = "acknowledged"
    INTERVIEWING = "interviewing"
    REJECTED = "rejected"
    OFFER = "offer"
    UNKNOWN = "unknown"

class JobApplication(BaseModel):
    email_id: str = Field(..., description="Reference to original email")
    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job position title")
    status: ApplicationStatus = Field(..., description="Application status")
    applied_date: datetime = Field(..., description="Date application was sent")
    last_updated: datetime = Field(..., description="Last status update")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Classification confidence")
    
    @validator('company')
    def validate_company(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Company name cannot be empty")
        return v.strip().title()

# models/research_models.py
class JobPosting(BaseModel):
    title: str = Field(..., description="Job posting title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    url: str = Field(..., description="Job posting URL")
    description: str = Field(..., description="Job description excerpt")
    source: str = Field(..., description="Source platform (LinkedIn, Glassdoor)")
    posted_date: Optional[datetime] = Field(None)
    salary_range: Optional[str] = Field(None)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
```

### List of tasks to be completed

```yaml
Task 1: Setup Configuration and Environment
CREATE config/settings.py:
  - PATTERN: Use python-dotenv like examples use os.getenv
  - Load environment variables with validation
  - Include Gmail OAuth and Brave API configurations
  - Handle missing credentials gracefully

CREATE .env.example:
  - Include all required environment variables
  - Add clear setup instructions
  - Follow pattern from examples documentation

Task 2: Implement Gmail Tool
CREATE tools/gmail_tool.py:
  - PATTERN: Follow Gmail API Python quickstart
  - Implement OAuth2 authentication flow
  - Handle token refresh automatically
  - Filter emails by date range (past month)
  - Parse email content (HTML, plain text, multipart)
  - Handle API rate limits and quotas

Task 3: Implement Brave Search Tool
CREATE tools/brave_search_tool.py:
  - PATTERN: Follow examples/pocketflow-tool-search/tools/search.py
  - Implement REST API client with httpx
  - Handle authentication with X-Subscription-Token
  - Search for job postings with company and position
  - Parse and structure results
  - Handle rate limits and errors

Task 4: Create Data Models
CREATE models/ package:
  - PATTERN: Use pydantic models throughout
  - Implement email, job application, and research models
  - Add validation and type safety
  - Include serialization methods

Task 5: Implement Email Agent
CREATE agents/email_agent.py:
  - PATTERN: Follow examples/pocketflow-multi-agent/main.py AsyncNode
  - Retrieve emails from Gmail API
  - Filter by date range (past month)
  - Parse email content properly
  - Store in shared state for next agents

Task 6: Implement Classification Agent
CREATE agents/classification_agent.py:
  - PATTERN: Use LLM analysis like examples/pocketflow-tool-search/tools/parser.py
  - Analyze email content for job application indicators
  - Extract company name, position, application status
  - Generate confidence scores
  - Handle false positives/negatives

Task 7: Implement Research Agent
CREATE agents/research_agent.py:
  - PATTERN: Combine search and analysis patterns
  - Search for job postings using Brave API
  - Match company and position from applications
  - Enrich job application data
  - Handle search failures gracefully

Task 8: Implement Status Agent
CREATE agents/status_agent.py:
  - PATTERN: Follow examples/pocketflow-map-reduce/nodes.py reduce pattern
  - Aggregate job application data
  - Generate status summary tables
  - Track application timeline
  - Provide insights and statistics

Task 9: Create PocketFlow Nodes
CREATE nodes/ package:
  - PATTERN: Follow examples node implementation patterns
  - Create nodes for each agent operation
  - Implement proper prep/exec/post methods
  - Handle shared state management
  - Include error handling

Task 10: Create Main Flow
CREATE flow.py:
  - PATTERN: Follow examples/pocketflow-tool-search/flow.py
  - Connect nodes in proper sequence
  - Handle asynchronous operations
  - Manage shared state between nodes
  - Include error recovery

Task 11: Implement CLI Interface
CREATE main.py:
  - PATTERN: Follow examples/pocketflow-tool-search/main.py
  - Provide interactive commands
  - Stream real-time updates
  - Handle user input validation
  - Display results in formatted tables

Task 12: Add Comprehensive Tests
CREATE tests/ package:
  - PATTERN: Mirror examples test structure
  - Mock Gmail and Brave API calls
  - Test all agent operations
  - Test error conditions
  - Ensure 80%+ coverage

Task 13: Create Documentation
CREATE README.md:
  - PATTERN: Follow examples README structure
  - Include setup and installation steps
  - Document API key configuration
  - Add usage examples
  - Include troubleshooting guide
```

### Per task pseudocode

```python
# Task 2: Gmail Tool Implementation
async def get_recent_emails(credentials_path: str, days_back: int = 30) -> List[GmailEmail]:
    # PATTERN: OAuth2 flow from Gmail quickstart
    service = build_gmail_service(credentials_path)
    
    # CRITICAL: Use proper Gmail query syntax for date filtering
    query = f"after:{(datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')}"
    
    # GOTCHA: Gmail API returns max 100 results per request
    all_emails = []
    next_page_token = None
    
    while True:
        # PATTERN: Handle pagination properly
        results = service.users().messages().list(
            userId="me",
            q=query,
            pageToken=next_page_token,
            maxResults=100
        ).execute()
        
        # CRITICAL: Rate limiting - Gmail allows 250 quota units per user per second
        await asyncio.sleep(0.1)
        
        # Parse each email message
        for msg in results.get('messages', []):
            email_data = await parse_email_message(service, msg['id'])
            all_emails.append(GmailEmail(**email_data))
        
        if 'nextPageToken' not in results:
            break
        next_page_token = results['nextPageToken']
    
    return all_emails

# Task 6: Classification Agent
async def classify_job_application(email: GmailEmail) -> Optional[JobApplication]:
    # PATTERN: LLM analysis with structured output
    prompt = f"""
    Analyze this email to determine if it's a job application:
    
    From: {email.sender}
    Subject: {email.subject}
    Content: {email.body[:2000]}...
    
    Output in YAML format:
    ```yaml
    is_job_application: true/false
    company: "Company Name"
    position: "Job Title"
    status: "applied/acknowledged/interviewing/rejected/offer"
    confidence: 0.95
    ```
    """
    
    # CRITICAL: Use structured output parsing
    response = await call_llm(prompt)
    classification = parse_yaml_response(response)
    
    # GOTCHA: Handle false positives with confidence threshold
    if classification['is_job_application'] and classification['confidence'] > 0.7:
        return JobApplication(
            email_id=email.id,
            company=classification['company'],
            position=classification['position'],
            status=ApplicationStatus(classification['status']),
            applied_date=email.received_date,
            last_updated=email.received_date,
            confidence_score=classification['confidence']
        )
    
    return None

# Task 7: Research Agent
async def research_job_posting(job_app: JobApplication) -> List[JobPosting]:
    # PATTERN: External API integration with error handling
    search_query = f"{job_app.company} {job_app.position} job"
    
    try:
        # CRITICAL: Brave API rate limiting
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": brave_api_key},
                params={"q": search_query, "count": 10},
                timeout=30.0
            )
            
            if response.status_code != 200:
                logger.warning(f"Brave API error: {response.status_code}")
                return []
            
            # PATTERN: Structure API response data
            results = response.json()
            job_postings = []
            
            for result in results.get('web', {}).get('results', []):
                if is_job_posting_relevant(result, job_app):
                    posting = JobPosting(
                        title=result['title'],
                        company=extract_company_name(result),
                        location=extract_location(result),
                        url=result['url'],
                        description=result['description'],
                        source=identify_source(result['url']),
                        relevance_score=calculate_relevance(result, job_app)
                    )
                    job_postings.append(posting)
            
            return job_postings
    
    except httpx.TimeoutException:
        logger.error(f"Timeout searching for {job_app.company} {job_app.position}")
        return []
```

### Integration Points
```yaml
ENVIRONMENT:
  - add to: .env
  - vars: |
      # Gmail OAuth2 Configuration
      GMAIL_CREDENTIALS_PATH=./credentials/credentials.json
      GMAIL_TOKEN_PATH=./credentials/token.json
      
      # Brave Search API
      BRAVE_API_KEY=BSA_your_api_key_here
      
      # LLM Configuration
      OPENAI_API_KEY=sk-your_openai_key_here
      
      # Application Settings
      EMAIL_DAYS_BACK=30
      CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
      RESEARCH_MAX_RESULTS=10

OAUTH_SETUP:
  - Gmail credentials: Download from Google Cloud Console
  - First run: Browser opens for user authorization
  - Token storage: ./credentials/token.json (auto-created)
  - Refresh: Automatic token refresh handled

DEPENDENCIES:
  - Update requirements.txt with:
    - pocketflow>=0.1.0
    - google-api-python-client>=2.0.0
    - google-auth-httplib2>=0.1.0
    - google-auth-oauthlib>=1.0.0
    - httpx>=0.25.0
    - pydantic>=2.0.0
    - python-dotenv>=1.0.0
    - openai>=1.0.0
    - pyyaml>=6.0.0
    - rich>=13.0.0  # For CLI formatting
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check . --fix              # Auto-fix style issues
mypy .                          # Type checking

# Expected: No errors. If errors, READ and fix.
```

### Level 2: Unit Tests
```python
# test_email_agent.py
@pytest.mark.asyncio
async def test_email_retrieval():
    """Test email agent retrieves emails correctly"""
    agent = EmailAgent()
    with patch('tools.gmail_tool.build_gmail_service') as mock_service:
        mock_service.return_value = create_mock_gmail_service()
        
        shared = {"days_back": 30}
        result = await agent.exec_async(shared)
        
        assert len(result) > 0
        assert all(isinstance(email, GmailEmail) for email in result)

# test_classification_agent.py
@pytest.mark.asyncio
async def test_job_application_classification():
    """Test classification agent identifies job applications"""
    agent = ClassificationAgent()
    
    # Test job application email
    job_email = create_test_email(
        subject="Application for Software Engineer Position",
        body="Thank you for applying to our Software Engineer position..."
    )
    
    result = await agent.classify_email(job_email)
    assert result is not None
    assert result.company == "Test Company"
    assert result.position == "Software Engineer"
    assert result.confidence_score > 0.7

    # Test non-job email
    normal_email = create_test_email(
        subject="Meeting tomorrow",
        body="Don't forget our meeting tomorrow at 2pm"
    )
    
    result = await agent.classify_email(normal_email)
    assert result is None

# test_research_agent.py
@pytest.mark.asyncio
async def test_job_posting_research():
    """Test research agent finds relevant job postings"""
    agent = ResearchAgent()
    
    job_app = JobApplication(
        email_id="test_123",
        company="Google",
        position="Software Engineer",
        status=ApplicationStatus.APPLIED,
        applied_date=datetime.now(),
        last_updated=datetime.now(),
        confidence_score=0.9
    )
    
    with patch('httpx.AsyncClient.get') as mock_get:
        mock_get.return_value = create_mock_brave_response()
        
        results = await agent.research_job_posting(job_app)
        
        assert len(results) > 0
        assert all(posting.company == "Google" for posting in results)
        assert all(posting.relevance_score > 0.5 for posting in results)

# test_status_agent.py
def test_status_summary_generation():
    """Test status agent generates comprehensive summaries"""
    agent = StatusAgent()
    
    job_apps = [
        create_test_job_application("Google", "SWE", ApplicationStatus.APPLIED),
        create_test_job_application("Meta", "SWE", ApplicationStatus.INTERVIEWING),
        create_test_job_application("Netflix", "SWE", ApplicationStatus.REJECTED)
    ]
    
    summary = agent.generate_status_summary(job_apps)
    
    assert summary.total_applications == 3
    assert summary.status_counts[ApplicationStatus.APPLIED] == 1
    assert summary.status_counts[ApplicationStatus.INTERVIEWING] == 1
    assert summary.status_counts[ApplicationStatus.REJECTED] == 1
```

```bash
# Run tests iteratively until passing:
pytest tests/ -v --cov=agents --cov=tools --cov=models --cov-report=term-missing

# If failing: Debug specific test, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test Gmail OAuth setup
python -c "from tools.gmail_tool import build_gmail_service; print('Gmail auth successful')"

# Test Brave API connection
python -c "from tools.brave_search_tool import test_brave_connection; test_brave_connection()"

# Test full CLI workflow
python main.py --help
python main.py run-analysis --days-back 7 --dry-run

# Expected output:
# 📧 Retrieving emails from last 7 days...
# 🔍 Found 45 emails to analyze
# 🤖 Classifying job applications...
# 📊 Identified 3 job applications
# 🔬 Researching job postings...
# 📈 Generated status report
# 
# Job Application Summary:
# =====================
# Total Applications: 3
# Applied: 2
# Interviewing: 1
# Rejected: 0
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check .`
- [ ] No type errors: `mypy .`
- [ ] Gmail OAuth flow works (credentials.json → token.json)
- [ ] Brave Search API returns relevant results
- [ ] Email classification achieves >80% accuracy on test data
- [ ] Research agent finds relevant job postings
- [ ] Status agent generates comprehensive reports
- [ ] CLI provides clear, formatted output
- [ ] Error cases handled gracefully (network, API limits, auth)
- [ ] Documentation includes complete setup guide
- [ ] .env.example has all required variables

---

## Anti-Patterns to Avoid
- ❌ Don't hardcode API keys or credentials
- ❌ Don't ignore Gmail API quota limits
- ❌ Don't skip email content format handling (HTML/plain/multipart)
- ❌ Don't use synchronous operations in async context
- ❌ Don't ignore rate limits for external APIs
- ❌ Don't trust LLM classification without confidence thresholds
- ❌ Don't commit OAuth tokens or credentials
- ❌ Don't skip error handling for network operations
- ❌ Don't ignore timezone handling in date filtering
- ❌ Don't assume email content is always in English

## Confidence Score: 8/10

High confidence due to:
- Clear PocketFlow patterns from examples
- Well-documented Gmail and Brave APIs
- Proven multi-agent architecture patterns
- Comprehensive validation gates

Minor uncertainty on:
- Email content parsing complexity (HTML/multipart)
- Job classification accuracy without training data
- Brave API response format variations for job-specific searches

The implementation follows established patterns and includes robust error handling and testing approaches.