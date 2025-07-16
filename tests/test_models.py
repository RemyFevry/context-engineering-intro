"""
Tests for data models (email, job, research).
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from models.email_models import (
    GmailEmail, 
    EmailBatch, 
    create_email_batch,
    clean_email_content,
    extract_job_keywords,
    is_job_related_email
)
from models.job_models import (
    JobApplication,
    ApplicationStatus,
    ApplicationSummary,
    create_job_application,
    create_application_summary
)
from models.research_models import (
    JobPosting,
    CompanyInsights,
    ResearchResult,
    create_job_posting,
    create_research_result
)


class TestGmailEmail:
    """Test GmailEmail model."""
    
    def test_gmail_email_creation(self):
        """Test creating a GmailEmail instance."""
        email = GmailEmail(
            id="test_id",
            sender="sender@example.com",
            subject="Test Subject",
            received_date=datetime.now(timezone.utc),
            content="Test content"
        )
        
        assert email.id == "test_id"
        assert email.sender == "sender@example.com"
        assert email.subject == "Test Subject"
        assert email.content == "Test content"
        assert isinstance(email.received_date, datetime)
    
    def test_gmail_email_validation(self):
        """Test GmailEmail validation."""
        # Test with missing required fields
        with pytest.raises(ValidationError):
            GmailEmail()
        
        # Test with invalid email format
        with pytest.raises(ValidationError):
            GmailEmail(
                id="test",
                sender="invalid_email",
                subject="Test",
                received_date=datetime.now(timezone.utc),
                content="Test"
            )
    
    def test_gmail_email_optional_fields(self):
        """Test GmailEmail with optional fields."""
        email = GmailEmail(
            id="test_id",
            sender="sender@example.com",
            subject="Test Subject",
            received_date=datetime.now(timezone.utc),
            content="Test content",
            thread_id="thread_123",
            labels=['INBOX', 'UNREAD'],
            snippet="Test snippet"
        )
        
        assert email.thread_id == "thread_123"
        assert email.labels == ['INBOX', 'UNREAD']
        assert email.snippet == "Test snippet"
    
    def test_gmail_email_get_domain(self):
        """Test get_domain method."""
        email = GmailEmail(
            id="test_id",
            sender="user@example.com",
            subject="Test",
            received_date=datetime.now(timezone.utc),
            content="Test"
        )
        
        assert email.get_domain() == "example.com"
    
    def test_gmail_email_get_domain_invalid(self):
        """Test get_domain with invalid email."""
        email = GmailEmail(
            id="test_id",
            sender="invalid_email",
            subject="Test",
            received_date=datetime.now(timezone.utc),
            content="Test"
        )
        
        with pytest.raises(ValidationError):
            # This should fail during model creation
            pass
    
    def test_gmail_email_is_recent(self):
        """Test is_recent method."""
        # Recent email (yesterday)
        recent_email = GmailEmail(
            id="recent_id",
            sender="sender@example.com",
            subject="Recent",
            received_date=datetime.now(timezone.utc) - timedelta(days=1),
            content="Recent content"
        )
        
        assert recent_email.is_recent(7) is True
        assert recent_email.is_recent(0) is False
        
        # Old email (10 days ago)
        old_email = GmailEmail(
            id="old_id",
            sender="sender@example.com",
            subject="Old",
            received_date=datetime.now(timezone.utc) - timedelta(days=10),
            content="Old content"
        )
        
        assert old_email.is_recent(7) is False
        assert old_email.is_recent(15) is True
    
    def test_gmail_email_word_count(self):
        """Test word_count method."""
        email = GmailEmail(
            id="test_id",
            sender="sender@example.com",
            subject="Test Subject",
            received_date=datetime.now(timezone.utc),
            content="This is a test email with ten words exactly here."
        )
        
        assert email.word_count() == 10
    
    def test_gmail_email_word_count_empty(self):
        """Test word_count with empty content."""
        email = GmailEmail(
            id="test_id",
            sender="sender@example.com",
            subject="Test Subject",
            received_date=datetime.now(timezone.utc),
            content=""
        )
        
        assert email.word_count() == 0


class TestEmailBatch:
    """Test EmailBatch model."""
    
    def test_email_batch_creation(self):
        """Test creating EmailBatch."""
        emails = [
            GmailEmail(
                id="email1",
                sender="sender1@example.com",
                subject="Subject 1",
                received_date=datetime.now(timezone.utc),
                content="Content 1"
            ),
            GmailEmail(
                id="email2",
                sender="sender2@example.com",
                subject="Subject 2",
                received_date=datetime.now(timezone.utc),
                content="Content 2"
            )
        ]
        
        batch = EmailBatch(
            batch_id="batch_123",
            emails=emails,
            created_at=datetime.now(timezone.utc),
            source="gmail"
        )
        
        assert batch.batch_id == "batch_123"
        assert len(batch.emails) == 2
        assert batch.source == "gmail"
    
    def test_email_batch_statistics(self):
        """Test EmailBatch statistics methods."""
        emails = [
            GmailEmail(
                id="email1",
                sender="sender1@example.com",
                subject="Job Application",
                received_date=datetime.now(timezone.utc) - timedelta(days=1),
                content="Thank you for your application"
            ),
            GmailEmail(
                id="email2",
                sender="sender2@example.com",
                subject="Newsletter",
                received_date=datetime.now(timezone.utc) - timedelta(days=10),
                content="Monthly newsletter"
            )
        ]
        
        batch = EmailBatch(
            batch_id="batch_123",
            emails=emails,
            created_at=datetime.now(timezone.utc),
            source="gmail"
        )
        
        assert batch.total_emails() == 2
        assert batch.recent_emails(7) == 1
        assert batch.get_date_range()[0] < batch.get_date_range()[1]
    
    def test_email_batch_empty(self):
        """Test EmailBatch with empty emails."""
        batch = EmailBatch(
            batch_id="empty_batch",
            emails=[],
            created_at=datetime.now(timezone.utc),
            source="gmail"
        )
        
        assert batch.total_emails() == 0
        assert batch.recent_emails(7) == 0
        
        # get_date_range should return None for empty batch
        date_range = batch.get_date_range()
        assert date_range == (None, None)


class TestJobApplication:
    """Test JobApplication model."""
    
    def test_job_application_creation(self):
        """Test creating JobApplication."""
        app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc),
            confidence_score=0.85
        )
        
        assert app.email_id == "email_123"
        assert app.company == "Google"
        assert app.position == "Software Engineer"
        assert app.status == ApplicationStatus.APPLIED
        assert app.confidence_score == 0.85
    
    def test_job_application_validation(self):
        """Test JobApplication validation."""
        # Test confidence score validation
        with pytest.raises(ValidationError):
            JobApplication(
                email_id="email_123",
                company="Google",
                position="Software Engineer",
                status=ApplicationStatus.APPLIED,
                applied_date=datetime.now(timezone.utc),
                confidence_score=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValidationError):
            JobApplication(
                email_id="email_123",
                company="Google",
                position="Software Engineer",
                status=ApplicationStatus.APPLIED,
                applied_date=datetime.now(timezone.utc),
                confidence_score=-0.1  # Invalid: < 0.0
            )
    
    def test_job_application_optional_fields(self):
        """Test JobApplication with optional fields."""
        app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc),
            confidence_score=0.85,
            job_url="https://jobs.google.com/123",
            application_method="online",
            notes="Applied through website"
        )
        
        assert app.job_url == "https://jobs.google.com/123"
        assert app.application_method == "online"
        assert app.notes == "Applied through website"
    
    def test_job_application_get_days_since_applied(self):
        """Test get_days_since_applied method."""
        # Application from 5 days ago
        app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=5),
            confidence_score=0.85
        )
        
        days_since = app.get_days_since_applied()
        assert days_since == 5
    
    def test_job_application_get_days_since_updated(self):
        """Test get_days_since_updated method."""
        # Application updated 3 days ago
        app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=5),
            last_updated=datetime.now(timezone.utc) - timedelta(days=3),
            confidence_score=0.85
        )
        
        days_since = app.get_days_since_updated()
        assert days_since == 3
    
    def test_job_application_is_stale(self):
        """Test is_stale method."""
        # Fresh application
        fresh_app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=1),
            confidence_score=0.85
        )
        
        assert fresh_app.is_stale(7) is False
        
        # Stale application
        stale_app = JobApplication(
            email_id="email_456",
            company="Microsoft",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc) - timedelta(days=10),
            confidence_score=0.85
        )
        
        assert stale_app.is_stale(7) is True
    
    def test_job_application_to_dict(self):
        """Test to_dict method."""
        app = JobApplication(
            email_id="email_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc),
            confidence_score=0.85
        )
        
        app_dict = app.to_dict()
        
        assert app_dict['email_id'] == "email_123"
        assert app_dict['company'] == "Google"
        assert app_dict['position'] == "Software Engineer"
        assert app_dict['status'] == ApplicationStatus.APPLIED.value
        assert app_dict['confidence_score'] == 0.85


class TestApplicationSummary:
    """Test ApplicationSummary model."""
    
    def test_application_summary_creation(self):
        """Test creating ApplicationSummary."""
        summary = ApplicationSummary(
            total_applications=10,
            response_rate=0.3,
            interview_rate=0.2,
            offer_rate=0.1,
            rejection_rate=0.4,
            status_counts={ApplicationStatus.APPLIED: 5, ApplicationStatus.REJECTED: 4}
        )
        
        assert summary.total_applications == 10
        assert summary.response_rate == 0.3
        assert summary.interview_rate == 0.2
        assert summary.offer_rate == 0.1
        assert summary.rejection_rate == 0.4
        assert summary.status_counts[ApplicationStatus.APPLIED] == 5
    
    def test_application_summary_validation(self):
        """Test ApplicationSummary validation."""
        # Test rate validation (should be 0-1)
        with pytest.raises(ValidationError):
            ApplicationSummary(
                total_applications=10,
                response_rate=1.5,  # Invalid: > 1.0
                interview_rate=0.2,
                offer_rate=0.1,
                rejection_rate=0.4
            )
        
        # Test negative total applications
        with pytest.raises(ValidationError):
            ApplicationSummary(
                total_applications=-1,  # Invalid: < 0
                response_rate=0.3,
                interview_rate=0.2,
                offer_rate=0.1,
                rejection_rate=0.4
            )


class TestJobPosting:
    """Test JobPosting model."""
    
    def test_job_posting_creation(self):
        """Test creating JobPosting."""
        posting = JobPosting(
            title="Software Engineer",
            company="Google",
            url="https://jobs.google.com/123",
            location="Mountain View, CA",
            relevance_score=0.85
        )
        
        assert posting.title == "Software Engineer"
        assert posting.company == "Google"
        assert posting.url == "https://jobs.google.com/123"
        assert posting.location == "Mountain View, CA"
        assert posting.relevance_score == 0.85
    
    def test_job_posting_validation(self):
        """Test JobPosting validation."""
        # Test relevance score validation
        with pytest.raises(ValidationError):
            JobPosting(
                title="Software Engineer",
                company="Google",
                url="https://jobs.google.com/123",
                location="Mountain View, CA",
                relevance_score=1.5  # Invalid: > 1.0
            )
        
        # Test invalid URL
        with pytest.raises(ValidationError):
            JobPosting(
                title="Software Engineer",
                company="Google",
                url="not_a_url",
                location="Mountain View, CA",
                relevance_score=0.85
            )
    
    def test_job_posting_is_recent(self):
        """Test is_recent method."""
        # Recent posting
        recent_posting = JobPosting(
            title="Software Engineer",
            company="Google",
            url="https://jobs.google.com/123",
            location="Mountain View, CA",
            relevance_score=0.85,
            posted_date=datetime.now(timezone.utc) - timedelta(days=5)
        )
        
        assert recent_posting.is_recent(7) is True
        assert recent_posting.is_recent(3) is False
        
        # Old posting
        old_posting = JobPosting(
            title="Software Engineer",
            company="Google",
            url="https://jobs.google.com/456",
            location="Mountain View, CA",
            relevance_score=0.85,
            posted_date=datetime.now(timezone.utc) - timedelta(days=10)
        )
        
        assert old_posting.is_recent(7) is False
        assert old_posting.is_recent(15) is True
    
    def test_job_posting_is_recent_no_date(self):
        """Test is_recent with no posted date."""
        posting = JobPosting(
            title="Software Engineer",
            company="Google",
            url="https://jobs.google.com/123",
            location="Mountain View, CA",
            relevance_score=0.85
        )
        
        # Should return False if no posted date
        assert posting.is_recent(7) is False


class TestResearchResult:
    """Test ResearchResult model."""
    
    def test_research_result_creation(self):
        """Test creating ResearchResult."""
        job_postings = [
            JobPosting(
                title="Software Engineer",
                company="Google",
                url="https://jobs.google.com/123",
                location="Mountain View, CA",
                relevance_score=0.85
            )
        ]
        
        result = ResearchResult(
            target_company="Google",
            target_position="Software Engineer",
            job_postings=job_postings,
            total_results=1,
            average_relevance=0.85
        )
        
        assert result.target_company == "Google"
        assert result.target_position == "Software Engineer"
        assert len(result.job_postings) == 1
        assert result.total_results == 1
        assert result.average_relevance == 0.85
    
    def test_research_result_best_match(self):
        """Test best_match property."""
        job_postings = [
            JobPosting(
                title="Software Engineer",
                company="Google",
                url="https://jobs.google.com/123",
                location="Mountain View, CA",
                relevance_score=0.85
            ),
            JobPosting(
                title="Senior Software Engineer",
                company="Google",
                url="https://jobs.google.com/456",
                location="Mountain View, CA",
                relevance_score=0.95
            )
        ]
        
        result = ResearchResult(
            target_company="Google",
            target_position="Software Engineer",
            job_postings=job_postings,
            total_results=2,
            average_relevance=0.9
        )
        
        best_match = result.best_match
        assert best_match is not None
        assert best_match.relevance_score == 0.95
        assert best_match.title == "Senior Software Engineer"
    
    def test_research_result_best_match_empty(self):
        """Test best_match with empty job postings."""
        result = ResearchResult(
            target_company="Google",
            target_position="Software Engineer",
            job_postings=[],
            total_results=0,
            average_relevance=0.0
        )
        
        assert result.best_match is None
    
    def test_research_result_validation(self):
        """Test ResearchResult validation."""
        # Test average relevance validation
        with pytest.raises(ValidationError):
            ResearchResult(
                target_company="Google",
                target_position="Software Engineer",
                job_postings=[],
                total_results=0,
                average_relevance=1.5  # Invalid: > 1.0
            )
        
        # Test negative total results
        with pytest.raises(ValidationError):
            ResearchResult(
                target_company="Google",
                target_position="Software Engineer",
                job_postings=[],
                total_results=-1,  # Invalid: < 0
                average_relevance=0.0
            )


class TestModelFactories:
    """Test model factory functions."""
    
    def test_create_email_batch(self):
        """Test create_email_batch factory."""
        emails = [
            GmailEmail(
                id="email1",
                sender="sender1@example.com",
                subject="Subject 1",
                received_date=datetime.now(timezone.utc),
                content="Content 1"
            )
        ]
        
        batch = create_email_batch(emails)
        
        assert len(batch.emails) == 1
        assert batch.source == "gmail"
        assert batch.batch_id.startswith("batch_")
    
    def test_create_job_application(self):
        """Test create_job_application factory."""
        email = GmailEmail(
            id="email1",
            sender="recruiter@google.com",
            subject="Application Confirmation",
            received_date=datetime.now(timezone.utc),
            content="Thank you for your application"
        )
        
        app = create_job_application(
            email=email,
            company="Google",
            position="Software Engineer",
            confidence_score=0.85
        )
        
        assert app.email_id == "email1"
        assert app.company == "Google"
        assert app.position == "Software Engineer"
        assert app.confidence_score == 0.85
        assert app.status == ApplicationStatus.APPLIED
    
    def test_create_application_summary(self):
        """Test create_application_summary factory."""
        applications = [
            JobApplication(
                email_id="email1",
                company="Google",
                position="Software Engineer",
                status=ApplicationStatus.APPLIED,
                applied_date=datetime.now(timezone.utc),
                confidence_score=0.85
            ),
            JobApplication(
                email_id="email2",
                company="Microsoft",
                position="Software Engineer",
                status=ApplicationStatus.REJECTED,
                applied_date=datetime.now(timezone.utc),
                confidence_score=0.75
            )
        ]
        
        summary = create_application_summary(applications)
        
        assert summary.total_applications == 2
        assert summary.rejection_rate == 0.5
        assert ApplicationStatus.APPLIED in summary.status_counts
        assert ApplicationStatus.REJECTED in summary.status_counts
    
    def test_create_job_posting(self):
        """Test create_job_posting factory."""
        posting = create_job_posting(
            title="Software Engineer",
            company="Google",
            url="https://jobs.google.com/123",
            location="Mountain View, CA",
            relevance_score=0.85
        )
        
        assert posting.title == "Software Engineer"
        assert posting.company == "Google"
        assert posting.relevance_score == 0.85
    
    def test_create_research_result(self):
        """Test create_research_result factory."""
        job_postings = [
            JobPosting(
                title="Software Engineer",
                company="Google",
                url="https://jobs.google.com/123",
                location="Mountain View, CA",
                relevance_score=0.85
            )
        ]
        
        result = create_research_result(
            target_company="Google",
            target_position="Software Engineer",
            job_postings=job_postings
        )
        
        assert result.target_company == "Google"
        assert result.target_position == "Software Engineer"
        assert result.total_results == 1
        assert result.average_relevance == 0.85


class TestModelHelpers:
    """Test model helper functions."""
    
    def test_clean_email_content(self):
        """Test clean_email_content function."""
        dirty_content = "<html><body>Hello <b>world</b>!</body></html>"
        clean_content = clean_email_content(dirty_content)
        
        assert "<html>" not in clean_content
        assert "<body>" not in clean_content
        assert "<b>" not in clean_content
        assert "Hello world!" in clean_content
    
    def test_extract_job_keywords(self):
        """Test extract_job_keywords function."""
        email_content = "Thank you for your application to the Software Engineer position at Google. We appreciate your interest in joining our team."
        
        keywords = extract_job_keywords(email_content)
        
        assert "application" in keywords
        assert "position" in keywords
        assert "team" in keywords
    
    def test_is_job_related_email(self):
        """Test is_job_related_email function."""
        job_email = GmailEmail(
            id="job_email",
            sender="hr@google.com",
            subject="Application Status Update",
            received_date=datetime.now(timezone.utc),
            content="Thank you for your application"
        )
        
        non_job_email = GmailEmail(
            id="non_job_email",
            sender="friend@gmail.com",
            subject="Weekend Plans",
            received_date=datetime.now(timezone.utc),
            content="What are your plans for the weekend?"
        )
        
        assert is_job_related_email(job_email) is True
        assert is_job_related_email(non_job_email) is False


if __name__ == "__main__":
    pytest.main([__file__])