"""
Job application data models for job application tracking system.

Contains Pydantic models for job applications, application status,
and related job tracking data structures.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator, model_validator
from urllib.parse import urlparse

class ApplicationStatus(str, Enum):
    """Job application status values."""
    APPLIED = "applied"
    ACKNOWLEDGED = "acknowledged"
    INTERVIEWING = "interviewing"
    REJECTED = "rejected"
    OFFER = "offer"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"

class ApplicationSource(str, Enum):
    """Source of job application."""
    DIRECT = "direct"
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GLASSDOOR = "glassdoor"
    COMPANY_WEBSITE = "company_website"
    REFERRAL = "referral"
    RECRUITER = "recruiter"
    OTHER = "other"
    UNKNOWN = "unknown"

class InterviewType(str, Enum):
    """Types of interviews."""
    PHONE = "phone"
    VIDEO = "video"
    ONSITE = "onsite"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PANEL = "panel"
    FINAL = "final"

class JobApplication(BaseModel):
    """
    Job application model with comprehensive tracking.
    
    Represents a single job application with all relevant metadata,
    status tracking, and analysis results.
    """
    
    # Core identification
    email_id: str = Field(..., description="Reference to original email")
    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job position title")
    
    # Status tracking
    status: ApplicationStatus = Field(..., description="Current application status")
    applied_date: datetime = Field(..., description="Date application was sent")
    last_updated: datetime = Field(..., description="Last status update")
    
    # Analysis metadata
    confidence_score: float = Field(
        0.0, 
        ge=0.0, 
        le=1.0, 
        description="Classification confidence score"
    )
    
    # Additional details
    job_description: Optional[str] = Field(None, description="Job description if available")
    location: Optional[str] = Field(None, description="Job location")
    salary_range: Optional[str] = Field(None, description="Salary range if mentioned")
    application_source: ApplicationSource = Field(
        ApplicationSource.UNKNOWN, 
        description="Source of application"
    )
    
    # Application tracking
    application_url: Optional[str] = Field(None, description="Link to job posting")
    reference_number: Optional[str] = Field(None, description="Application reference number")
    contact_person: Optional[str] = Field(None, description="Contact person name")
    contact_email: Optional[str] = Field(None, description="Contact email address")
    
    # Interview tracking
    interviews: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of interview details"
    )
    
    # Notes and comments
    notes: Optional[str] = Field(None, description="Additional notes")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    
    # System metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Record creation time")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update time")
    
    @validator('company')
    def validate_company(cls, v):
        """
        Validate and normalize company name.
        
        Args:
            v (str): Company name
            
        Returns:
            str: Normalized company name
            
        Raises:
            ValueError: If company name is empty or invalid
        """
        if not v or v.strip() == "":
            raise ValueError("Company name cannot be empty")
        
        # Clean and normalize company name
        company = v.strip()
        
        # Remove common suffixes for normalization
        suffixes = [', Inc.', ', LLC', ', Corp.', ', Ltd.', ' Inc.', ' LLC', ' Corp.', ' Ltd.']
        for suffix in suffixes:
            if company.endswith(suffix):
                company = company[:-len(suffix)]
                break
        
        # Title case
        company = company.title()
        
        # Handle special cases
        special_cases = {
            'Google': 'Google',
            'Microsoft': 'Microsoft',
            'Apple': 'Apple',
            'Amazon': 'Amazon',
            'Meta': 'Meta',
            'Netflix': 'Netflix',
            'Tesla': 'Tesla',
            'Uber': 'Uber',
            'Airbnb': 'Airbnb'
        }
        
        for key, value in special_cases.items():
            if key.lower() in company.lower():
                company = value
                break
        
        return company
    
    @validator('position')
    def validate_position(cls, v):
        """
        Validate and normalize position title.
        
        Args:
            v (str): Position title
            
        Returns:
            str: Normalized position title
            
        Raises:
            ValueError: If position is empty
        """
        if not v or v.strip() == "":
            raise ValueError("Position title cannot be empty")
        
        # Clean and normalize position
        position = v.strip()
        
        # Common position normalizations
        normalizations = {
            'swe': 'Software Engineer',
            'se': 'Software Engineer',
            'dev': 'Developer',
            'eng': 'Engineer',
            'mgr': 'Manager',
            'sr': 'Senior',
            'jr': 'Junior'
        }
        
        # Apply normalizations (case-insensitive)
        for abbrev, full in normalizations.items():
            if abbrev.lower() in position.lower():
                position = position.replace(abbrev, full)
                position = position.replace(abbrev.upper(), full)
                position = position.replace(abbrev.capitalize(), full)
        
        return position.title()
    
    @validator('applied_date')
    def validate_applied_date(cls, v):
        """
        Validate applied date is reasonable.
        
        Args:
            v (datetime): Applied date
            
        Returns:
            datetime: Validated date
            
        Raises:
            ValueError: If date is unreasonable
        """
        # Ensure both dates are timezone-aware for comparison
        now = datetime.now(timezone.utc)
        v_aware = v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v
        
        # Date should not be in the future
        if v_aware > now:
            raise ValueError("Applied date cannot be in the future")
        
        # Date should not be too old (2 years max)
        two_years_ago = now - timedelta(days=730)
        if v_aware < two_years_ago:
            raise ValueError("Applied date cannot be more than 2 years ago")
        
        return v
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """
        Validate confidence score range.
        
        Args:
            v (float): Confidence score
            
        Returns:
            float: Validated score
        """
        return max(0.0, min(1.0, v))
    
    @validator('application_url')
    def validate_application_url(cls, v):
        """
        Validate application URL format.
        
        Args:
            v (str): Application URL
            
        Returns:
            str: Validated URL
        """
        if not v:
            return v
        
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")
    
    @model_validator(mode='before')
    @classmethod
    def set_derived_fields(cls, values):
        """
        Set derived fields and update timestamps.
        
        Args:
            values (dict): Field values
            
        Returns:
            dict: Updated values
        """
        # Update last_updated timestamp
        values['last_updated'] = datetime.now(timezone.utc)
        
        # Infer application source from URL if available
        url = values.get('application_url', '')
        if url and values.get('application_source') == ApplicationSource.UNKNOWN:
            if 'linkedin.com' in url:
                values['application_source'] = ApplicationSource.LINKEDIN
            elif 'indeed.com' in url:
                values['application_source'] = ApplicationSource.INDEED
            elif 'glassdoor.com' in url:
                values['application_source'] = ApplicationSource.GLASSDOOR
            else:
                values['application_source'] = ApplicationSource.COMPANY_WEBSITE
        
        return values
    
    def add_interview(self, interview_type: InterviewType, scheduled_date: datetime, 
                     notes: str = None, completed: bool = False) -> None:
        """
        Add interview to application.
        
        Args:
            interview_type (InterviewType): Type of interview
            scheduled_date (datetime): Interview date/time
            notes (str, optional): Interview notes
            completed (bool): Whether interview is completed
        """
        interview = {
            'type': interview_type.value,
            'scheduled_date': scheduled_date.isoformat(),
            'notes': notes or '',
            'completed': completed,
            'added_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.interviews.append(interview)
        self.last_updated = datetime.now(timezone.utc)
        
        # Update status if interview is scheduled
        if not completed and self.status == ApplicationStatus.APPLIED:
            self.status = ApplicationStatus.INTERVIEWING
    
    def update_status(self, new_status: ApplicationStatus, notes: str = None) -> None:
        """
        Update application status.
        
        Args:
            new_status (ApplicationStatus): New status
            notes (str, optional): Status update notes
        """
        old_status = self.status
        self.status = new_status
        self.last_updated = datetime.now(timezone.utc)
        
        if notes:
            if self.notes:
                self.notes += f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] Status changed from {old_status.value} to {new_status.value}: {notes}"
            else:
                self.notes = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d')}] Status changed from {old_status.value} to {new_status.value}: {notes}"
    
    def add_tag(self, tag: str) -> None:
        """
        Add tag to application.
        
        Args:
            tag (str): Tag to add
        """
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.last_updated = datetime.now(timezone.utc)
    
    def remove_tag(self, tag: str) -> None:
        """
        Remove tag from application.
        
        Args:
            tag (str): Tag to remove
        """
        tag = tag.strip().lower()
        if tag in self.tags:
            self.tags.remove(tag)
            self.last_updated = datetime.now(timezone.utc)
    
    def get_days_since_applied(self) -> int:
        """
        Get number of days since application was sent.
        
        Returns:
            int: Days since applied
        """
        now = datetime.now(timezone.utc)
        applied_date_aware = self.applied_date.replace(tzinfo=timezone.utc) if self.applied_date.tzinfo is None else self.applied_date
        return (now - applied_date_aware).days
    
    def get_days_since_updated(self) -> int:
        """
        Get number of days since last update.
        
        Returns:
            int: Days since last update
        """
        now = datetime.now(timezone.utc)
        last_updated_aware = self.last_updated.replace(tzinfo=timezone.utc) if self.last_updated.tzinfo is None else self.last_updated
        return (now - last_updated_aware).days
    
    def is_stale(self, days_threshold: int = 30) -> bool:
        """
        Check if application is stale (no updates for a while).
        
        Args:
            days_threshold (int): Days threshold for staleness
            
        Returns:
            bool: True if application is stale
        """
        return self.get_days_since_updated() > days_threshold
    
    def get_interview_count(self) -> int:
        """
        Get number of interviews scheduled/completed.
        
        Returns:
            int: Interview count
        """
        return len(self.interviews)
    
    def get_completed_interviews(self) -> List[Dict[str, Any]]:
        """
        Get list of completed interviews.
        
        Returns:
            List[Dict[str, Any]]: Completed interviews
        """
        return [interview for interview in self.interviews if interview.get('completed')]
    
    def get_upcoming_interviews(self) -> List[Dict[str, Any]]:
        """
        Get list of upcoming interviews.
        
        Returns:
            List[Dict[str, Any]]: Upcoming interviews
        """
        now = datetime.now(timezone.utc)
        upcoming = []
        
        for interview in self.interviews:
            if not interview.get('completed'):
                scheduled_str = interview.get('scheduled_date', '')
                try:
                    scheduled = datetime.fromisoformat(scheduled_str)
                    if scheduled > now:
                        upcoming.append(interview)
                except ValueError:
                    continue
        
        return upcoming
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert application to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Application data as dictionary
        """
        return {
            'email_id': self.email_id,
            'company': self.company,
            'position': self.position,
            'status': self.status.value,
            'applied_date': self.applied_date.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'confidence_score': self.confidence_score,
            'location': self.location,
            'salary_range': self.salary_range,
            'application_source': self.application_source.value,
            'application_url': self.application_url,
            'reference_number': self.reference_number,
            'contact_person': self.contact_person,
            'contact_email': self.contact_email,
            'interview_count': self.get_interview_count(),
            'completed_interviews': len(self.get_completed_interviews()),
            'upcoming_interviews': len(self.get_upcoming_interviews()),
            'days_since_applied': self.get_days_since_applied(),
            'days_since_updated': self.get_days_since_updated(),
            'is_stale': self.is_stale(),
            'tags': self.tags,
            'notes': self.notes
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

class ApplicationSummary(BaseModel):
    """
    Summary statistics for job applications.
    
    Provides aggregated statistics and insights across multiple applications.
    """
    
    total_applications: int = Field(0, description="Total number of applications")
    status_counts: Dict[ApplicationStatus, int] = Field(
        default_factory=dict, 
        description="Count by status"
    )
    companies: List[str] = Field(default_factory=list, description="List of companies")
    positions: List[str] = Field(default_factory=list, description="List of positions")
    
    # Date ranges
    earliest_application: Optional[datetime] = Field(None, description="Earliest application date")
    latest_application: Optional[datetime] = Field(None, description="Latest application date")
    
    # Statistics
    average_confidence: float = Field(0.0, description="Average confidence score")
    total_interviews: int = Field(0, description="Total interviews scheduled")
    stale_applications: int = Field(0, description="Number of stale applications")
    
    # Success metrics
    response_rate: float = Field(0.0, description="Response rate (acknowledged + interviewing + offer)")
    interview_rate: float = Field(0.0, description="Interview rate")
    offer_rate: float = Field(0.0, description="Offer rate")
    
    def calculate_metrics(self, applications: List[JobApplication]) -> None:
        """
        Calculate summary metrics from applications.
        
        Args:
            applications (List[JobApplication]): List of applications
        """
        if not applications:
            return
        
        self.total_applications = len(applications)
        
        # Status counts
        for status in ApplicationStatus:
            self.status_counts[status] = len([app for app in applications if app.status == status])
        
        # Companies and positions
        self.companies = list(set(app.company for app in applications))
        self.positions = list(set(app.position for app in applications))
        
        # Date ranges
        dates = [app.applied_date for app in applications]
        self.earliest_application = min(dates)
        self.latest_application = max(dates)
        
        # Statistics
        self.average_confidence = sum(app.confidence_score for app in applications) / len(applications)
        self.total_interviews = sum(app.get_interview_count() for app in applications)
        self.stale_applications = len([app for app in applications if app.is_stale()])
        
        # Success metrics
        responded = self.status_counts.get(ApplicationStatus.ACKNOWLEDGED, 0) + \
                   self.status_counts.get(ApplicationStatus.INTERVIEWING, 0) + \
                   self.status_counts.get(ApplicationStatus.OFFER, 0)
        
        self.response_rate = responded / self.total_applications if self.total_applications > 0 else 0.0
        
        interviewing = self.status_counts.get(ApplicationStatus.INTERVIEWING, 0) + \
                      self.status_counts.get(ApplicationStatus.OFFER, 0)
        
        self.interview_rate = interviewing / self.total_applications if self.total_applications > 0 else 0.0
        
        offers = self.status_counts.get(ApplicationStatus.OFFER, 0)
        self.offer_rate = offers / self.total_applications if self.total_applications > 0 else 0.0

# Helper functions
def create_application_summary(applications: List[JobApplication]) -> ApplicationSummary:
    """
    Create application summary from list of applications.
    
    Args:
        applications (List[JobApplication]): List of applications
        
    Returns:
        ApplicationSummary: Summary statistics
    """
    summary = ApplicationSummary()
    summary.calculate_metrics(applications)
    return summary

def filter_applications_by_status(applications: List[JobApplication], 
                                 status: ApplicationStatus) -> List[JobApplication]:
    """
    Filter applications by status.
    
    Args:
        applications (List[JobApplication]): List of applications
        status (ApplicationStatus): Status to filter by
        
    Returns:
        List[JobApplication]: Filtered applications
    """
    return [app for app in applications if app.status == status]

def filter_applications_by_company(applications: List[JobApplication], 
                                  company: str) -> List[JobApplication]:
    """
    Filter applications by company.
    
    Args:
        applications (List[JobApplication]): List of applications
        company (str): Company name to filter by
        
    Returns:
        List[JobApplication]: Filtered applications
    """
    return [app for app in applications if company.lower() in app.company.lower()]

if __name__ == "__main__":
    # Test job application models
    from datetime import datetime, timezone
    
    # Test application creation
    test_app = JobApplication(
        email_id="test_email_123",
        company="Google Inc.",
        position="Software Engineer",
        status=ApplicationStatus.APPLIED,
        applied_date=datetime.now(timezone.utc),
        last_updated=datetime.now(timezone.utc),
        confidence_score=0.85,
        location="Mountain View, CA",
        application_source=ApplicationSource.LINKEDIN,
        application_url="https://linkedin.com/jobs/12345"
    )
    
    print("✅ Job application model created successfully")
    print(f"🏢 Company: {test_app.company}")
    print(f"💼 Position: {test_app.position}")
    print(f"📊 Status: {test_app.status.value}")
    print(f"📅 Applied: {test_app.applied_date}")
    print(f"🎯 Confidence: {test_app.confidence_score}")
    print(f"📍 Source: {test_app.application_source.value}")
    print(f"⏱️ Days since applied: {test_app.get_days_since_applied()}")
    
    # Test interview addition
    test_app.add_interview(InterviewType.PHONE, datetime.now(timezone.utc), "Initial screening")
    print(f"📞 Interviews: {test_app.get_interview_count()}")
    
    # Test summary creation
    summary = create_application_summary([test_app])
    print(f"📈 Summary - Total: {summary.total_applications}, Response rate: {summary.response_rate:.2%}")
    
    print("✅ Job application models working correctly")