"""
Email data models for job application tracking system.

Contains Pydantic models for email data structures with validation
and serialization capabilities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any
import re
from pydantic import BaseModel, Field, validator, root_validator, model_validator
from email_validator import validate_email, EmailNotValidError

class EmailSource(str, Enum):
    """Email source platforms."""
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    IMAP = "imap"
    POP3 = "pop3"

class EmailFormat(str, Enum):
    """Email content formats."""
    HTML = "html"
    PLAIN = "plain"
    MULTIPART = "multipart"
    RICH_TEXT = "rich_text"

class EmailPriority(str, Enum):
    """Email priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class GmailEmail(BaseModel):
    """
    Gmail email model with validation and processing.
    
    Represents a single email from Gmail with all necessary
    metadata and content for job application analysis.
    """
    
    id: str = Field(..., description="Gmail message ID")
    thread_id: str = Field(..., description="Gmail thread ID")
    sender: str = Field(..., description="Email sender address")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email body content")
    received_date: datetime = Field(..., description="Date email was received")
    labels: List[str] = Field(default_factory=list, description="Gmail labels")
    format: EmailFormat = Field(..., description="Email content format")
    raw_headers: Dict[str, str] = Field(default_factory=dict, description="Raw email headers")
    
    # Additional metadata
    sender_domain: Optional[str] = Field(None, description="Sender domain")
    priority: EmailPriority = Field(EmailPriority.NORMAL, description="Email priority")
    is_read: bool = Field(False, description="Whether email has been read")
    is_starred: bool = Field(False, description="Whether email is starred")
    has_attachments: bool = Field(False, description="Whether email has attachments")
    
    # Processing metadata
    processed_at: Optional[datetime] = Field(None, description="When email was processed")
    processing_notes: Optional[str] = Field(None, description="Notes from processing")
    
    @validator('sender')
    def validate_sender_email(cls, v):
        """
        Validate sender email address format.
        
        Args:
            v (str): Sender email address (may include display name)
            
        Returns:
            str: Validated email address
            
        Raises:
            ValueError: If email format is invalid
        """
        if not v:
            raise ValueError("Sender email cannot be empty")
        
        # Extract email from "Display Name <email@domain.com>" format
        from utils import extract_email_address
        email_addr = extract_email_address(v)
        
        try:
            # Validate email format
            validated = validate_email(email_addr)
            return validated.email
        except EmailNotValidError as e:
            # For automated emails and marketing domains, be more lenient
            lenient_keywords = [
                'noreply', 'no-reply', 'mail', 'marketing', 'newsletter', 
                'deals', 'info', 'support', 'hello', 'team', 'my@'
            ]
            if any(keyword in email_addr.lower() for keyword in lenient_keywords):
                return email_addr
            raise ValueError(f"Invalid sender email format: {e}")
    
    @validator('subject')
    def validate_subject(cls, v):
        """
        Validate and clean email subject.
        
        Args:
            v (str): Email subject
            
        Returns:
            str: Cleaned subject
        """
        if not v:
            return "[No Subject]"
        
        # Clean subject line
        subject = v.strip()
        
        # Remove common prefixes
        prefixes = ['Re:', 'RE:', 'Fwd:', 'FWD:', 'Fw:']
        for prefix in prefixes:
            if subject.startswith(prefix):
                subject = subject[len(prefix):].strip()
        
        return subject
    
    @validator('body')
    def validate_body(cls, v):
        """
        Validate and clean email body.
        
        Args:
            v (str): Email body content
            
        Returns:
            str: Cleaned body content
        """
        if not v:
            return ""
        
        # Remove excessive whitespace
        body = ' '.join(v.split())
        
        # Truncate if too long
        if len(body) > 10000:
            body = body[:10000] + "..."
        
        return body
    
    @validator('received_date')
    def validate_received_date(cls, v):
        """
        Validate received date is not in the future.
        
        Args:
            v (datetime): Received date
            
        Returns:
            datetime: Validated date
            
        Raises:
            ValueError: If date is in the future
        """
        # Ensure both dates are timezone-aware for comparison
        now = datetime.now(timezone.utc)
        v_aware = v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v
        
        if v_aware > now:
            raise ValueError("Received date cannot be in the future")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def set_derived_fields(cls, values):
        """
        Set derived fields based on other values.
        
        Args:
            values (dict): Field values
            
        Returns:
            dict: Updated values with derived fields
        """
        sender = values.get('sender', '')
        labels = values.get('labels', [])
        
        # Extract sender domain
        if sender and '@' in sender:
            try:
                domain = sender.split('@')[1].lower()
                values['sender_domain'] = domain
            except IndexError:
                values['sender_domain'] = None
        
        # Determine priority from labels
        if 'IMPORTANT' in labels:
            values['priority'] = EmailPriority.HIGH
        elif 'CATEGORY_PROMOTIONS' in labels:
            values['priority'] = EmailPriority.LOW
        
        # Check if read
        values['is_read'] = 'UNREAD' not in labels
        
        # Check if starred
        values['is_starred'] = 'STARRED' in labels
        
        # Check for attachments (basic check)
        values['has_attachments'] = 'CATEGORY_ATTACHMENTS' in labels
        
        return values
    
    def is_automated(self) -> bool:
        """
        Check if email is automated/system-generated.
        
        Returns:
            bool: True if email appears to be automated
        """
        automated_keywords = [
            'noreply', 'no-reply', 'donotreply', 'do-not-reply',
            'system', 'automated', 'notification', 'alerts',
            'daemon', 'mailer', 'postmaster'
        ]
        
        sender_lower = self.sender.lower()
        return any(keyword in sender_lower for keyword in automated_keywords)
    
    def is_job_related_domain(self) -> bool:
        """
        Check if sender domain suggests job-related content.
        
        Returns:
            bool: True if domain suggests job-related content
        """
        if not self.sender_domain:
            return False
        
        job_related_domains = [
            'linkedin', 'glassdoor', 'indeed', 'monster', 'dice',
            'ziprecruiter', 'careerbuilder', 'simplyhired',
            'greenhouse', 'lever', 'workday', 'smartrecruiters',
            'jobvite', 'bamboohr', 'workable', 'greenhouse.io'
        ]
        
        job_keywords = [
            'careers', 'jobs', 'hr', 'talent', 'recruiting',
            'recruitment', 'hire', 'employment'
        ]
        
        domain = self.sender_domain.lower()
        
        # Check direct domain matches
        if any(job_domain in domain for job_domain in job_related_domains):
            return True
        
        # Check keywords in domain
        if any(keyword in domain for keyword in job_keywords):
            return True
        
        return False
    
    def get_subject_keywords(self) -> List[str]:
        """
        Extract keywords from email subject.
        
        Returns:
            List[str]: List of keywords
        """
        if not self.subject:
            return []
        
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', self.subject.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
            're', 'fwd', 'fw', 'your', 'you', 'this', 'that', 'from'
        }
        
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def get_body_preview(self, max_length: int = 200) -> str:
        """
        Get preview of email body.
        
        Args:
            max_length (int): Maximum preview length
            
        Returns:
            str: Body preview
        """
        if not self.body:
            return ""
        
        preview = self.body[:max_length]
        if len(self.body) > max_length:
            preview += "..."
        
        return preview
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert email to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Email data as dictionary
        """
        return {
            'id': self.id,
            'thread_id': self.thread_id,
            'sender': self.sender,
            'sender_domain': self.sender_domain,
            'subject': self.subject,
            'body_preview': self.get_body_preview(),
            'received_date': self.received_date.isoformat(),
            'format': self.format.value,
            'priority': self.priority.value,
            'is_read': self.is_read,
            'is_starred': self.is_starred,
            'has_attachments': self.has_attachments,
            'labels': self.labels,
            'is_automated': self.is_automated(),
            'is_job_related_domain': self.is_job_related_domain(),
            'keywords': self.get_subject_keywords()
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

class EmailBatch(BaseModel):
    """
    Collection of emails for batch processing.
    
    Provides utilities for processing multiple emails together.
    """
    
    emails: List[GmailEmail] = Field(..., description="List of emails")
    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Batch creation time")
    processed_at: Optional[datetime] = Field(None, description="Batch processing time")
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """
        Get date range of emails in batch.
        
        Returns:
            tuple[datetime, datetime]: Start and end dates
        """
        if not self.emails:
            now = datetime.now(timezone.utc)
            return now, now
        
        dates = [email.received_date for email in self.emails]
        return min(dates), max(dates)
    
    def get_senders(self) -> List[str]:
        """
        Get unique senders in batch.
        
        Returns:
            List[str]: List of unique sender emails
        """
        return list(set(email.sender for email in self.emails))
    
    def get_domains(self) -> List[str]:
        """
        Get unique sender domains in batch.
        
        Returns:
            List[str]: List of unique domains
        """
        domains = []
        for email in self.emails:
            if email.sender_domain:
                domains.append(email.sender_domain)
        return list(set(domains))
    
    def filter_by_domain(self, domain: str) -> List[GmailEmail]:
        """
        Filter emails by sender domain.
        
        Args:
            domain (str): Domain to filter by
            
        Returns:
            List[GmailEmail]: Filtered emails
        """
        return [
            email for email in self.emails 
            if email.sender_domain and domain.lower() in email.sender_domain.lower()
        ]
    
    def filter_job_related(self) -> List[GmailEmail]:
        """
        Filter emails that appear to be job-related.
        
        Returns:
            List[GmailEmail]: Job-related emails
        """
        return [email for email in self.emails if email.is_job_related_domain()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get batch statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        if not self.emails:
            return {}
        
        start_date, end_date = self.get_date_range()
        
        return {
            'total_emails': len(self.emails),
            'unique_senders': len(self.get_senders()),
            'unique_domains': len(self.get_domains()),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'job_related_count': len(self.filter_job_related()),
            'automated_count': len([e for e in self.emails if e.is_automated()]),
            'unread_count': len([e for e in self.emails if not e.is_read]),
            'starred_count': len([e for e in self.emails if e.is_starred]),
            'with_attachments_count': len([e for e in self.emails if e.has_attachments])
        }

# Helper functions
def create_email_batch(emails: List[GmailEmail], batch_id: str = None) -> EmailBatch:
    """
    Create email batch from list of emails.
    
    Args:
        emails (List[GmailEmail]): List of emails
        batch_id (str, optional): Batch ID. Auto-generated if not provided.
        
    Returns:
        EmailBatch: Email batch object
    """
    if not batch_id:
        batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    return EmailBatch(emails=emails, batch_id=batch_id)

if __name__ == "__main__":
    # Test email models
    from datetime import datetime, timezone
    
    # Test email creation
    test_email = GmailEmail(
        id="test_123",
        thread_id="thread_456",
        sender="recruiter@google.com",
        subject="Re: Software Engineer Position",
        body="Thank you for your application. We will review it shortly.",
        received_date=datetime.now(timezone.utc),
        labels=["INBOX", "IMPORTANT"],
        format=EmailFormat.PLAIN,
        raw_headers={"from": "recruiter@google.com", "to": "applicant@example.com"}
    )
    
    print("✅ Email model created successfully")
    print(f"📧 Subject: {test_email.subject}")
    print(f"📧 Sender domain: {test_email.sender_domain}")
    print(f"📧 Is job-related: {test_email.is_job_related_domain()}")
    print(f"📧 Keywords: {test_email.get_subject_keywords()}")
    print(f"📧 Preview: {test_email.get_body_preview()}")
    
    # Test batch creation
    batch = create_email_batch([test_email])
    stats = batch.get_statistics()
    print(f"📊 Batch stats: {stats}")
    
    print("✅ Email models working correctly")