# Job Application Tracker

A comprehensive AI-powered job application tracking system that automatically retrieves, analyzes, and provides insights on your job applications using email data and web research.

## Features

- **📧 Email Integration**: Automatically retrieve job-related emails from Gmail
- **🤖 AI Classification**: Use OpenAI to intelligently classify and extract job application information
- **🔍 Job Research**: Search for related job postings using Brave Search API
- **📊 Analytics & Insights**: Generate detailed reports and AI-powered insights
- **💻 CLI Interface**: Rich command-line interface with progress indicators and formatted output
- **🔄 Multi-Agent Architecture**: Built on PocketFlow for scalable, modular processing

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features Overview](#features-overview)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd job-application-tracker
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure Gmail API**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable Gmail API
   - Create OAuth 2.0 credentials
   - Download credentials.json to `./credentials/`

5. **Run the tracker**:
   ```bash
   python cli.py track
   ```

## Installation

### Prerequisites

- Python 3.8 or higher
- Gmail account with API access
- (Optional) OpenAI API key for AI insights
- (Optional) Brave Search API key for job research

### Install from Source

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd job-application-tracker
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PocketFlow**:
   ```bash
   pip install pocketflow
   ```

### Docker Installation (Optional)

```bash
# Build the Docker image
docker build -t job-tracker .

# Run the container
docker run -it --rm -v $(pwd)/credentials:/app/credentials job-tracker
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Required for Gmail integration
GMAIL_CREDENTIALS_PATH=./credentials/credentials.json
GMAIL_TOKEN_PATH=./credentials/token.json

# Optional: OpenAI API for AI insights
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Optional: Brave Search API for job research
BRAVE_API_KEY=your_brave_api_key_here

# Configuration options
EMAIL_DAYS_BACK=30
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
RESEARCH_MAX_RESULTS=10
RATE_LIMIT_DELAY=1.0
LOG_LEVEL=INFO
ENABLE_VALIDATION=true
```

### Gmail API Setup

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable the Gmail API

2. **Create OAuth 2.0 Credentials**:
   - Go to "Credentials" in the Google Cloud Console
   - Click "Create Credentials" → "OAuth 2.0 Client ID"
   - Choose "Desktop application"
   - Download the credentials file

3. **Place Credentials**:
   - Rename the downloaded file to `credentials.json`
   - Place it in the `./credentials/` directory

4. **First Run Authentication**:
   - Run the application for the first time
   - Follow the OAuth flow in your browser
   - Grant access to Gmail
   - The system will save a token for future use

### API Keys (Optional)

- **OpenAI API**: Required for AI-powered insights and recommendations
- **Brave Search API**: Required for job posting research and market analysis

## Usage

### Command Line Interface

The system provides a rich CLI with multiple commands:

#### Basic Usage

```bash
# Track job applications from last 30 days
python cli.py track

# Track specific number of days
python cli.py track --days 60

# Save results to file
python cli.py track --output results.json

# Verbose output
python cli.py track --verbose
```

#### Other Commands

```bash
# Show system status
python cli.py status

# Interactive setup
python cli.py setup

# Show configuration
python cli.py config

# Show help
python cli.py help
```

#### Output Formats

```bash
# Rich formatted output (default)
python cli.py track --format rich

# JSON output
python cli.py track --format json

# Plain text output
python cli.py track --format plain
```

### Python API

You can also use the system programmatically:

```python
from flow import run_job_application_tracking

# Run the workflow
result = await run_job_application_tracking(days_back=30)

if result['success']:
    print(f"Found {len(result['results']['job_applications'])} job applications")
    for app in result['results']['job_applications']:
        print(f"- {app['company']}: {app['position']}")
else:
    print(f"Error: {result['error']}")
```

## Features Overview

### 📧 Email Processing

The system automatically:
- Retrieves emails from Gmail using OAuth2
- Filters job-related emails using domain analysis and keywords
- Processes HTML and plain text email content
- Extracts structured information from email threads

### 🤖 AI Classification

Using OpenAI's GPT models to:
- Identify job application emails with high accuracy
- Extract company names, positions, and application status
- Assign confidence scores to classifications
- Handle various email formats and languages

### 🔍 Job Research

Leverages Brave Search API to:
- Find related job postings for each application
- Analyze market activity and competition
- Gather company insights and hiring patterns
- Calculate relevance scores for job matches

### 📊 Analytics & Reporting

Generates comprehensive reports including:
- Application success rates and conversion funnels
- Time-based analysis of application patterns
- Company and position performance metrics
- Stale application alerts and follow-up recommendations
- AI-powered insights and improvement suggestions

### 💻 Rich CLI Interface

Features include:
- Progress indicators for long-running operations
- Colored output with tables and panels
- Interactive prompts for setup and configuration
- Multiple output formats (rich, JSON, plain text)
- Error handling with helpful messages

## API Documentation

### Core Components

#### Flow Orchestration

```python
from flow import JobApplicationTrackingFlow

# Create flow instance
flow = JobApplicationTrackingFlow()

# Run full workflow
result = await flow.run_full_workflow(days_back=30)

# Get flow statistics
stats = flow.get_statistics()
```

#### Data Models

```python
from models.job_models import JobApplication, ApplicationStatus
from models.email_models import GmailEmail, EmailBatch
from models.research_models import JobPosting, ResearchResult

# Create job application
app = JobApplication(
    email_id="email_123",
    company="Google",
    position="Software Engineer",
    status=ApplicationStatus.APPLIED,
    applied_date=datetime.now(),
    confidence_score=0.85
)
```

#### Configuration

```python
from config.settings import get_settings

# Get current settings
settings = get_settings()

# Access configuration
print(f"Email days back: {settings.email_days_back}")
print(f"OpenAI model: {settings.openai_model}")
```

### Agents

The system includes four specialized agents:

#### Email Agent
- Handles Gmail API authentication and email retrieval
- Filters and preprocesses email content
- Manages rate limiting and pagination

#### Classification Agent
- Uses LLM to classify emails as job applications
- Extracts structured information from email content
- Validates classification accuracy

#### Research Agent
- Searches for job postings using Brave Search API
- Analyzes company hiring patterns
- Calculates relevance scores and market insights

#### Status Agent
- Aggregates application data and generates reports
- Provides analytics and success metrics
- Generates AI-powered insights and recommendations

## Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --unit

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file test_models.py

# Run specific test function
python run_tests.py --function test_email_classification
```

### Test Structure

- `tests/test_config.py` - Configuration and settings tests
- `tests/test_utils.py` - Utility function tests
- `tests/test_models.py` - Data model tests
- `tests/test_flow.py` - Workflow orchestration tests
- `tests/conftest.py` - Test fixtures and configuration

### Test Coverage

The test suite includes:
- Unit tests for all core components
- Integration tests for workflow execution
- Mock tests for external API interactions
- Edge case and error handling tests

## Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Flow Manager   │    │  PocketFlow     │
│                 │───▶│                 │───▶│  Nodes          │
│  - Commands     │    │  - Orchestration│    │  - Email        │
│  - Formatting   │    │  - Error Handling│    │  - Classification│
│  - Progress     │    │  - Statistics   │    │  - Research     │
└─────────────────┘    └─────────────────┘    │  - Status       │
                                              └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Data Models   │    │     Agents      │
                       │                 │    │                 │
                       │  - Email        │◀───│  - Email Agent  │
                       │  - Job Apps     │    │  - Classification│
                       │  - Research     │    │  - Research     │
                       └─────────────────┘    │  - Status       │
                                              └─────────────────┘
                                                       │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   External APIs │    │     Tools       │
                       │                 │    │                 │
                       │  - Gmail API    │◀───│  - Gmail Tool   │
                       │  - OpenAI API   │    │  - Brave Tool   │
                       │  - Brave API    │    │  - Data Processor│
                       └─────────────────┘    └─────────────────┘
```

### Workflow Pipeline

1. **Email Processing**: Retrieve → Filter → Preprocess
2. **Classification**: Analyze → Extract → Validate
3. **Research**: Search → Analyze → Aggregate
4. **Reporting**: Generate → Insights → Export

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run tests**:
   ```bash
   python run_tests.py
   ```

5. **Format code**:
   ```bash
   black .
   ruff check .
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Use meaningful variable and function names

### Submitting Changes

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add entries to CHANGELOG.md**
4. **Submit a pull request** with a clear description

## Troubleshooting

### Common Issues

#### Gmail Authentication Errors

```bash
# Error: credentials not found
Solution: Ensure credentials.json is in ./credentials/ directory

# Error: access denied
Solution: Re-run OAuth flow and grant Gmail access

# Error: quota exceeded
Solution: Check Google Cloud Console for API limits
```

#### OpenAI API Errors

```bash
# Error: API key not found
Solution: Set OPENAI_API_KEY environment variable

# Error: rate limit exceeded
Solution: Add delays between requests or upgrade API plan
```

#### Brave Search API Errors

```bash
# Error: API key invalid
Solution: Verify BRAVE_API_KEY environment variable

# Error: no results found
Solution: Check search query and API endpoint
```

### Performance Optimization

- **Large Email Volumes**: Use pagination and batch processing
- **Rate Limiting**: Adjust `RATE_LIMIT_DELAY` setting
- **Memory Usage**: Process emails in smaller batches
- **API Costs**: Configure confidence thresholds to reduce API calls

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please:
1. Check the [troubleshooting section](#troubleshooting)
2. Search [existing issues](https://github.com/your-repo/issues)
3. Create a new issue with detailed information

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Acknowledgments

- **PocketFlow** for the multi-agent framework
- **OpenAI** for GPT models and API
- **Google** for Gmail API
- **Brave** for Search API
- **Rich** for beautiful CLI formatting