"""
Data models for job application tracking system.

This package contains Pydantic models for emails, job applications,
and research results with validation and serialization.
"""

from .email_models import GmailEmail, EmailSource, EmailFormat
from .job_models import JobApplication, ApplicationStatus
from .research_models import JobPosting

__all__ = [
    "GmailEmail",
    "EmailSource", 
    "EmailFormat",
    "JobApplication",
    "ApplicationStatus",
    "JobPosting"
]