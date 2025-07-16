"""
Data processing utilities for job application tracking system.

Contains utilities for processing and analyzing email and job data,
including text processing, similarity calculations, and data transformations.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter
import difflib

from models.email_models import GmailEmail, EmailBatch
from models.job_models import JobApplication, ApplicationStatus, ApplicationSummary
from models.research_models import JobPosting, ResearchResult
from utils import logger, clean_text, calculate_confidence_score

class DataProcessor:
    """
    Data processing utilities for job application tracking.
    
    Provides methods for processing emails, job applications, and research results
    with text analysis, similarity calculations, and data transformations.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so',
            'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time',
            'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first',
            'water', 'been', 'call', 'who', 'its', 'now', 'find', 'long',
            'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        }
        
        # Job application keywords for classification
        self.job_application_keywords = {
            'application': 3.0,
            'position': 2.5,
            'role': 2.0,
            'opportunity': 2.0,
            'hiring': 2.5,
            'interview': 3.0,
            'resume': 3.0,
            'cv': 2.5,
            'candidate': 2.0,
            'recruitment': 2.5,
            'talent': 2.0,
            'career': 2.0,
            'job': 2.5,
            'apply': 3.0,
            'application': 3.0,
            'thank you for applying': 4.0,
            'we received your application': 4.0,
            'application status': 3.5,
            'phone screen': 3.0,
            'next steps': 2.5,
            'offer': 3.5,
            'rejection': 3.0,
            'unfortunately': 2.5,
            'move forward': 2.5,
            'schedule': 2.0,
            'benefits': 2.0,
            'salary': 2.0,
            'compensation': 2.0
        }
        
        # Status keywords for determining application status
        self.status_keywords = {
            ApplicationStatus.ACKNOWLEDGED: [
                'received', 'thank you for applying', 'application received',
                'acknowledge', 'review', 'reviewing', 'under review'
            ],
            ApplicationStatus.INTERVIEWING: [
                'interview', 'phone screen', 'video call', 'meet', 'schedule',
                'next round', 'technical interview', 'onsite', 'panel'
            ],
            ApplicationStatus.REJECTED: [
                'unfortunately', 'regret', 'not moving forward', 'declined',
                'rejected', 'not selected', 'different direction', 'other candidates'
            ],
            ApplicationStatus.OFFER: [
                'offer', 'congratulations', 'pleased to offer', 'job offer',
                'position is yours', 'welcome to', 'start date', 'salary offer'
            ]
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of extracted keywords
        """
        if not text:
            return []
        
        # Clean and normalize text
        text = clean_text(text.lower())
        
        # Extract words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Get most common keywords
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(20)]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_clean = clean_text(text1.lower())
        text2_clean = clean_text(text2.lower())
        
        # Use difflib for similarity calculation
        similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        return similarity
    
    def classify_email_job_relevance(self, email: GmailEmail) -> Tuple[float, Dict[str, float]]:
        """
        Classify email for job application relevance.
        
        Args:
            email (GmailEmail): Email to classify
            
        Returns:
            Tuple[float, Dict[str, float]]: Confidence score and factor breakdown
        """
        factors = {}
        
        # Subject line analysis
        subject_score = self._analyze_subject_for_job_keywords(email.subject)
        factors['subject_keywords'] = subject_score
        
        # Sender domain analysis
        domain_score = 1.0 if email.is_job_related_domain() else 0.0
        factors['sender_domain'] = domain_score
        
        # Body content analysis
        body_score = self._analyze_body_for_job_keywords(email.body)
        factors['body_keywords'] = body_score
        
        # Automated email check (reduce score for automated emails)
        automated_penalty = 0.8 if email.is_automated() else 1.0
        factors['automated_penalty'] = automated_penalty
        
        # Calculate weighted confidence score
        weights = {
            'subject_keywords': 0.3,
            'sender_domain': 0.2,
            'body_keywords': 0.4,
            'automated_penalty': 0.1
        }
        
        confidence = sum(factors[key] * weights[key] for key in factors)
        confidence = min(1.0, confidence)  # Cap at 1.0
        
        return confidence, factors
    
    def _analyze_subject_for_job_keywords(self, subject: str) -> float:
        """
        Analyze subject line for job-related keywords.
        
        Args:
            subject (str): Email subject
            
        Returns:
            float: Keyword score (0.0-1.0)
        """
        if not subject:
            return 0.0
        
        subject_lower = subject.lower()
        score = 0.0
        max_score = 0.0
        
        for keyword, weight in self.job_application_keywords.items():
            max_score += weight
            if keyword in subject_lower:
                score += weight
        
        # Normalize score
        return min(1.0, score / max_score) if max_score > 0 else 0.0
    
    def _analyze_body_for_job_keywords(self, body: str) -> float:
        """
        Analyze email body for job-related keywords.
        
        Args:
            body (str): Email body
            
        Returns:
            float: Keyword score (0.0-1.0)
        """
        if not body:
            return 0.0
        
        body_lower = body.lower()
        score = 0.0
        max_score = 0.0
        
        for keyword, weight in self.job_application_keywords.items():
            max_score += weight
            if keyword in body_lower:
                score += weight
        
        # Normalize score
        return min(1.0, score / max_score) if max_score > 0 else 0.0
    
    def infer_application_status(self, email: GmailEmail) -> ApplicationStatus:
        """
        Infer application status from email content.
        
        Args:
            email (GmailEmail): Email to analyze
            
        Returns:
            ApplicationStatus: Inferred status
        """
        content = f"{email.subject} {email.body}".lower()
        
        # Check for status keywords
        for status, keywords in self.status_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    return status
        
        # Default to applied if no specific status found
        return ApplicationStatus.APPLIED
    
    def extract_company_from_email(self, email: GmailEmail) -> str:
        """
        Extract company name from email.
        
        Args:
            email (GmailEmail): Email to analyze
            
        Returns:
            str: Extracted company name
        """
        # Try sender domain first
        if email.sender_domain:
            domain = email.sender_domain
            
            # Skip common email providers
            skip_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
            if domain not in skip_domains:
                # Convert domain to company name
                company = domain.replace('.com', '').replace('.org', '').replace('.net', '')
                company = company.replace('-', ' ').replace('_', ' ').title()
                return company
        
        # Try to extract from subject line
        subject_patterns = [
            r'from\s+([A-Z][a-zA-Z\s&\.,]+)',
            r'at\s+([A-Z][a-zA-Z\s&\.,]+)',
            r'@\s+([A-Z][a-zA-Z\s&\.,]+)',
            r'([A-Z][a-zA-Z\s&\.,]+)\s+Team'
        ]
        
        for pattern in subject_patterns:
            match = re.search(pattern, email.subject)
            if match:
                company = match.group(1).strip()
                if len(company) > 2:
                    return company
        
        # Try to extract from body
        body_patterns = [
            r'on behalf of\s+([A-Z][a-zA-Z\s&\.,]+)',
            r'team at\s+([A-Z][a-zA-Z\s&\.,]+)',
            r'([A-Z][a-zA-Z\s&\.,]+)\s+is\s+looking',
            r'([A-Z][a-zA-Z\s&\.,]+)\s+hiring\s+team'
        ]
        
        for pattern in body_patterns:
            match = re.search(pattern, email.body)
            if match:
                company = match.group(1).strip()
                if len(company) > 2:
                    return company
        
        return "Unknown Company"
    
    def extract_position_from_email(self, email: GmailEmail) -> str:
        """
        Extract position title from email.
        
        Args:
            email (GmailEmail): Email to analyze
            
        Returns:
            str: Extracted position title
        """
        # Common position patterns
        position_patterns = [
            r'(?:position|role|opportunity)\s+(?:for|of|as)\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+)\s+(?:position|role|opportunity)',
            r'applying\s+for\s+(?:the\s+)?([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+)\s+(?:at|@)\s+[A-Z]',
            r'(Software\s+Engineer)',
            r'(Data\s+Scientist)',
            r'(Product\s+Manager)',
            r'(DevOps\s+Engineer)'
        ]
        
        content = f"{email.subject} {email.body}"
        
        for pattern in position_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                position = match.group(1).strip()
                if len(position) > 2:
                    return position.title()
        
        return "Unknown Position"
    
    def process_email_batch(self, email_batch: EmailBatch) -> List[Dict[str, Any]]:
        """
        Process a batch of emails for job application analysis.
        
        Args:
            email_batch (EmailBatch): Batch of emails to process
            
        Returns:
            List[Dict[str, Any]]: Processing results
        """
        results = []
        
        for email in email_batch.emails:
            # Calculate job relevance
            confidence, factors = self.classify_email_job_relevance(email)
            
            # Skip if confidence too low
            if confidence < 0.3:
                continue
            
            # Extract details
            company = self.extract_company_from_email(email)
            position = self.extract_position_from_email(email)
            status = self.infer_application_status(email)
            
            result = {
                'email_id': email.id,
                'company': company,
                'position': position,
                'status': status,
                'confidence': confidence,
                'factors': factors,
                'received_date': email.received_date,
                'subject': email.subject,
                'sender': email.sender,
                'body_preview': email.get_body_preview()
            }
            
            results.append(result)
        
        logger.info(f"Processed {len(email_batch.emails)} emails, found {len(results)} potential job applications")
        
        return results
    
    def merge_duplicate_applications(self, applications: List[JobApplication]) -> List[JobApplication]:
        """
        Merge duplicate job applications.
        
        Args:
            applications (List[JobApplication]): List of applications
            
        Returns:
            List[JobApplication]: Deduplicated applications
        """
        if not applications:
            return []
        
        # Group by company and position
        groups = {}
        for app in applications:
            key = (app.company.lower(), app.position.lower())
            if key not in groups:
                groups[key] = []
            groups[key].append(app)
        
        merged_applications = []
        
        for key, group in groups.items():
            if len(group) == 1:
                merged_applications.append(group[0])
            else:
                # Merge duplicates
                merged_app = self._merge_application_group(group)
                merged_applications.append(merged_app)
        
        return merged_applications
    
    def _merge_application_group(self, applications: List[JobApplication]) -> JobApplication:
        """
        Merge a group of duplicate applications.
        
        Args:
            applications (List[JobApplication]): Applications to merge
            
        Returns:
            JobApplication: Merged application
        """
        if not applications:
            return None
        
        # Sort by confidence score (highest first)
        applications.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Use the highest confidence application as base
        base_app = applications[0]
        
        # Merge information from other applications
        for app in applications[1:]:
            # Update status to most recent
            if app.last_updated > base_app.last_updated:
                base_app.status = app.status
                base_app.last_updated = app.last_updated
            
            # Merge notes
            if app.notes and app.notes not in (base_app.notes or ''):
                if base_app.notes:
                    base_app.notes += f"\n\n[Merged] {app.notes}"
                else:
                    base_app.notes = f"[Merged] {app.notes}"
            
            # Merge tags
            for tag in app.tags:
                if tag not in base_app.tags:
                    base_app.tags.append(tag)
        
        return base_app
    
    def calculate_application_metrics(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Calculate metrics for job applications.
        
        Args:
            applications (List[JobApplication]): List of applications
            
        Returns:
            Dict[str, Any]: Calculated metrics
        """
        if not applications:
            return {}
        
        # Basic counts
        total_apps = len(applications)
        status_counts = Counter(app.status for app in applications)
        
        # Date range
        dates = [app.applied_date for app in applications]
        date_range = {
            'earliest': min(dates),
            'latest': max(dates),
            'span_days': (max(dates) - min(dates)).days
        }
        
        # Success rates
        responded = status_counts.get(ApplicationStatus.ACKNOWLEDGED, 0) + \
                   status_counts.get(ApplicationStatus.INTERVIEWING, 0) + \
                   status_counts.get(ApplicationStatus.OFFER, 0)
        
        response_rate = responded / total_apps if total_apps > 0 else 0.0
        
        interviewing = status_counts.get(ApplicationStatus.INTERVIEWING, 0) + \
                      status_counts.get(ApplicationStatus.OFFER, 0)
        
        interview_rate = interviewing / total_apps if total_apps > 0 else 0.0
        
        offer_rate = status_counts.get(ApplicationStatus.OFFER, 0) / total_apps if total_apps > 0 else 0.0
        
        # Company and position analysis
        companies = Counter(app.company for app in applications)
        positions = Counter(app.position for app in applications)
        
        # Average confidence
        avg_confidence = sum(app.confidence_score for app in applications) / total_apps
        
        # Stale applications
        stale_count = len([app for app in applications if app.is_stale()])
        
        return {
            'total_applications': total_apps,
            'status_distribution': dict(status_counts),
            'date_range': date_range,
            'response_rate': response_rate,
            'interview_rate': interview_rate,
            'offer_rate': offer_rate,
            'top_companies': companies.most_common(10),
            'top_positions': positions.most_common(10),
            'average_confidence': avg_confidence,
            'stale_applications': stale_count
        }
    
    def generate_insights(self, applications: List[JobApplication]) -> List[str]:
        """
        Generate insights from job applications.
        
        Args:
            applications (List[JobApplication]): List of applications
            
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        if not applications:
            return ["No applications to analyze"]
        
        metrics = self.calculate_application_metrics(applications)
        
        # Response rate insights
        response_rate = metrics['response_rate']
        if response_rate > 0.3:
            insights.append(f"Great response rate of {response_rate:.1%}! Your applications are getting noticed.")
        elif response_rate > 0.1:
            insights.append(f"Response rate of {response_rate:.1%} is average. Consider tailoring your applications more.")
        else:
            insights.append(f"Low response rate of {response_rate:.1%}. Consider improving your resume and cover letters.")
        
        # Interview rate insights
        interview_rate = metrics['interview_rate']
        if interview_rate > 0.2:
            insights.append(f"Strong interview rate of {interview_rate:.1%}! Your profile is compelling.")
        elif interview_rate > 0.05:
            insights.append(f"Interview rate of {interview_rate:.1%} shows promise. Keep applying!")
        else:
            insights.append(f"Interview rate of {interview_rate:.1%} needs improvement. Focus on relevant skills.")
        
        # Application frequency insights
        span_days = metrics['date_range']['span_days']
        if span_days > 0:
            apps_per_day = len(applications) / span_days
            if apps_per_day > 2:
                insights.append("High application frequency. Make sure you're maintaining quality over quantity.")
            elif apps_per_day < 0.5:
                insights.append("Low application frequency. Consider applying to more positions.")
        
        # Company diversity insights
        top_companies = metrics['top_companies']
        if len(top_companies) > 0:
            top_company, count = top_companies[0]
            if count > len(applications) * 0.3:
                insights.append(f"Heavy focus on {top_company}. Consider diversifying your applications.")
        
        # Stale applications insights
        stale_count = metrics['stale_applications']
        if stale_count > 0:
            insights.append(f"{stale_count} applications haven't been updated recently. Consider following up.")
        
        return insights

# Factory function
def create_data_processor() -> DataProcessor:
    """
    Create and return data processor instance.
    
    Returns:
        DataProcessor: Configured data processor
    """
    return DataProcessor()

if __name__ == "__main__":
    # Test data processor
    processor = create_data_processor()
    
    # Test keyword extraction
    text = "Software Engineer position at Google Inc. Full-time opportunity in Mountain View, CA"
    keywords = processor.extract_keywords(text)
    print(f"✅ Keywords extracted: {keywords}")
    
    # Test similarity calculation
    text1 = "Software Engineer at Google"
    text2 = "Senior Software Engineer at Google Inc"
    similarity = processor.calculate_text_similarity(text1, text2)
    print(f"✅ Text similarity: {similarity:.2f}")
    
    print("✅ Data processor working correctly")