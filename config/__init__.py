"""
Configuration management for job application tracking system.

This package handles environment variables, API keys, and
application settings.
"""

from .settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings"
]