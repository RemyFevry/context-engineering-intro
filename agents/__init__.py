"""
Multi-agent system for job application tracking.

This package contains the core agents that handle different aspects
of job application analysis and tracking.
"""

from .email_agent import EmailAgent
from .classification_agent import ClassificationAgent
from .research_agent import ResearchAgent
from .status_agent import StatusAgent

__all__ = [
    "EmailAgent",
    "ClassificationAgent", 
    "ResearchAgent",
    "StatusAgent"
]