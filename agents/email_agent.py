"""
Email Agent for job application tracking system.

Handles email retrieval, filtering, and preprocessing for job application
analysis using Gmail API integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from tools.gmail_tool import GmailTool, create_gmail_tool
from tools.data_processor import DataProcessor, create_data_processor
from models.email_models import GmailEmail, EmailBatch, create_email_batch
from config.settings import get_settings
from utils import logger, format_datetime

class EmailAgent:
    """
    Email Agent for retrieving and processing emails.
    
    Handles Gmail API integration, email filtering, and preprocessing
    for job application analysis using PocketFlow patterns.
    """
    
    def __init__(self):
        """Initialize Email Agent with tools and settings."""
        self.settings = get_settings()
        self.gmail_tool = create_gmail_tool()
        self.data_processor = create_data_processor()
        
        # Processing statistics
        self.stats = {
            'total_emails_retrieved': 0,
            'job_related_emails': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the email agent.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Test Gmail connection
            success = await self.gmail_tool.test_connection()
            if not success:
                logger.error("Failed to initialize Gmail connection")
                return False
            
            logger.info("Email Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Email Agent: {e}")
            self.stats['errors'].append(f"Initialization failed: {e}")
            return False
    
    async def retrieve_recent_emails(self, days_back: int = None) -> EmailBatch:
        """
        Retrieve recent emails from Gmail.
        
        Args:
            days_back (int, optional): Days back to retrieve emails
            
        Returns:
            EmailBatch: Batch of retrieved emails
        """
        start_time = datetime.now()
        days_back = days_back or self.settings.email_days_back
        
        try:
            logger.info(f"Retrieving emails from the last {days_back} days")
            
            # Get emails from Gmail
            emails = await self.gmail_tool.get_recent_emails(days_back)
            
            # Create email batch
            batch = create_email_batch(emails)
            
            # Update statistics
            self.stats['total_emails_retrieved'] = len(emails)
            self.stats['last_run'] = datetime.now()
            self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully retrieved {len(emails)} emails")
            
            return batch
            
        except Exception as e:
            logger.error(f"Failed to retrieve emails: {e}")
            self.stats['errors'].append(f"Email retrieval failed: {e}")
            raise
    
    async def filter_job_related_emails(self, email_batch: EmailBatch) -> List[GmailEmail]:
        """
        Filter emails for job-related content.
        
        Args:
            email_batch (EmailBatch): Batch of emails to filter
            
        Returns:
            List[GmailEmail]: Job-related emails
        """
        try:
            logger.info(f"Filtering {len(email_batch.emails)} emails for job-related content")
            
            job_related_emails = []
            
            for email in email_batch.emails:
                # Calculate job relevance
                confidence, factors = self.data_processor.classify_email_job_relevance(email)
                
                # Filter by confidence threshold
                if confidence >= self.settings.classification_confidence_threshold:
                    job_related_emails.append(email)
                    logger.debug(f"Job-related email found: {email.subject[:50]}... (confidence: {confidence:.2f})")
            
            # Update statistics
            self.stats['job_related_emails'] = len(job_related_emails)
            
            logger.info(f"Found {len(job_related_emails)} job-related emails")
            
            return job_related_emails
            
        except Exception as e:
            logger.error(f"Failed to filter job-related emails: {e}")
            self.stats['errors'].append(f"Email filtering failed: {e}")
            raise
    
    async def preprocess_emails(self, emails: List[GmailEmail]) -> List[Dict[str, Any]]:
        """
        Preprocess emails for job application analysis.
        
        Args:
            emails (List[GmailEmail]): Emails to preprocess
            
        Returns:
            List[Dict[str, Any]]: Preprocessed email data
        """
        try:
            logger.info(f"Preprocessing {len(emails)} emails")
            
            preprocessed_emails = []
            
            for email in emails:
                # Extract job application details
                company = self.data_processor.extract_company_from_email(email)
                position = self.data_processor.extract_position_from_email(email)
                status = self.data_processor.infer_application_status(email)
                
                # Calculate confidence and factors
                confidence, factors = self.data_processor.classify_email_job_relevance(email)
                
                # Extract keywords
                keywords = self.data_processor.extract_keywords(f"{email.subject} {email.body}")
                
                # Create preprocessed data
                preprocessed = {
                    'email_id': email.id,
                    'thread_id': email.thread_id,
                    'sender': email.sender,
                    'sender_domain': email.sender_domain,
                    'subject': email.subject,
                    'body': email.body,
                    'body_preview': email.get_body_preview(),
                    'received_date': email.received_date,
                    'labels': email.labels,
                    'format': email.format,
                    'is_automated': email.is_automated(),
                    'is_job_related_domain': email.is_job_related_domain(),
                    
                    # Extracted information
                    'company': company,
                    'position': position,
                    'status': status,
                    'confidence': confidence,
                    'factors': factors,
                    'keywords': keywords,
                    
                    # Metadata
                    'processed_at': datetime.now(),
                    'processing_notes': f"Processed by EmailAgent with confidence {confidence:.2f}"
                }
                
                preprocessed_emails.append(preprocessed)
                
                logger.debug(f"Preprocessed email: {company} - {position} (confidence: {confidence:.2f})")
            
            logger.info(f"Successfully preprocessed {len(preprocessed_emails)} emails")
            
            return preprocessed_emails
            
        except Exception as e:
            logger.error(f"Failed to preprocess emails: {e}")
            self.stats['errors'].append(f"Email preprocessing failed: {e}")
            raise
    
    async def search_job_emails(self, query: str, max_results: int = 50) -> List[GmailEmail]:
        """
        Search for job-related emails using custom query.
        
        Args:
            query (str): Search query
            max_results (int): Maximum results to return
            
        Returns:
            List[GmailEmail]: Matching emails
        """
        try:
            logger.info(f"Searching for job emails with query: {query}")
            
            # Add job-related keywords to query
            job_enhanced_query = f"{query} (job OR position OR role OR hiring OR career OR application OR interview)"
            
            # Search emails
            emails = await self.gmail_tool.search_emails(job_enhanced_query, max_results)
            
            # Filter for job relevance
            job_emails = []
            for email in emails:
                confidence, _ = self.data_processor.classify_email_job_relevance(email)
                if confidence >= 0.5:  # Lower threshold for search results
                    job_emails.append(email)
            
            logger.info(f"Found {len(job_emails)} job-related emails from search")
            
            return job_emails
            
        except Exception as e:
            logger.error(f"Failed to search job emails: {e}")
            self.stats['errors'].append(f"Email search failed: {e}")
            raise
    
    async def get_company_emails(self, company_name: str, days_back: int = 90) -> List[GmailEmail]:
        """
        Get emails from a specific company.
        
        Args:
            company_name (str): Company name to search for
            days_back (int): Days back to search
            
        Returns:
            List[GmailEmail]: Company emails
        """
        try:
            logger.info(f"Getting emails from {company_name}")
            
            # Create search query
            query = f"from:{company_name} OR from:@{company_name.lower().replace(' ', '')}.com"
            
            # Add date filter
            start_date = datetime.now() - timedelta(days=days_back)
            query += f" after:{start_date.strftime('%Y/%m/%d')}"
            
            # Search emails
            emails = await self.gmail_tool.search_emails(query)
            
            logger.info(f"Found {len(emails)} emails from {company_name}")
            
            return emails
            
        except Exception as e:
            logger.error(f"Failed to get emails from {company_name}: {e}")
            self.stats['errors'].append(f"Company email search failed: {e}")
            raise
    
    async def analyze_email_patterns(self, emails: List[GmailEmail]) -> Dict[str, Any]:
        """
        Analyze patterns in email data.
        
        Args:
            emails (List[GmailEmail]): Emails to analyze
            
        Returns:
            Dict[str, Any]: Pattern analysis results
        """
        try:
            logger.info(f"Analyzing patterns in {len(emails)} emails")
            
            # Domain analysis
            domains = {}
            for email in emails:
                domain = email.sender_domain or 'unknown'
                domains[domain] = domains.get(domain, 0) + 1
            
            # Time analysis
            hourly_distribution = {}
            daily_distribution = {}
            
            for email in emails:
                hour = email.received_date.hour
                day = email.received_date.strftime('%A')
                
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
                daily_distribution[day] = daily_distribution.get(day, 0) + 1
            
            # Subject analysis
            subject_keywords = []
            for email in emails:
                keywords = self.data_processor.extract_keywords(email.subject)
                subject_keywords.extend(keywords)
            
            from collections import Counter
            top_subject_keywords = Counter(subject_keywords).most_common(10)
            
            # Job-related domain analysis
            job_domains = sum(1 for email in emails if email.is_job_related_domain())
            automated_emails = sum(1 for email in emails if email.is_automated())
            
            analysis = {
                'total_emails': len(emails),
                'unique_domains': len(domains),
                'top_domains': sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10],
                'hourly_distribution': hourly_distribution,
                'daily_distribution': daily_distribution,
                'top_subject_keywords': top_subject_keywords,
                'job_related_domains': job_domains,
                'automated_emails': automated_emails,
                'job_domain_percentage': (job_domains / len(emails)) * 100 if emails else 0,
                'automated_percentage': (automated_emails / len(emails)) * 100 if emails else 0
            }
            
            logger.info(f"Pattern analysis complete: {analysis['unique_domains']} unique domains, {analysis['job_domain_percentage']:.1f}% job-related")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze email patterns: {e}")
            self.stats['errors'].append(f"Pattern analysis failed: {e}")
            raise
    
    async def process_email_batch_for_jobs(self, email_batch: EmailBatch) -> Dict[str, Any]:
        """
        Process email batch for job application analysis.
        
        Args:
            email_batch (EmailBatch): Batch of emails to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            logger.info(f"Processing email batch for job applications")
            
            # Filter job-related emails
            job_emails = await self.filter_job_related_emails(email_batch)
            
            # Preprocess emails
            preprocessed_emails = await self.preprocess_emails(job_emails)
            
            # Analyze patterns
            pattern_analysis = await self.analyze_email_patterns(job_emails)
            
            # Get batch statistics
            batch_stats = email_batch.get_statistics()
            
            # Compile results
            results = {
                'batch_id': email_batch.batch_id,
                'total_emails': len(email_batch.emails),
                'job_related_emails': len(job_emails),
                'preprocessed_emails': preprocessed_emails,
                'pattern_analysis': pattern_analysis,
                'batch_statistics': batch_stats,
                'processing_timestamp': datetime.now(),
                'agent_statistics': self.get_statistics()
            }
            
            logger.info(f"Batch processing complete: {len(job_emails)} job-related emails found")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process email batch: {e}")
            self.stats['errors'].append(f"Batch processing failed: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dict[str, Any]: Agent statistics
        """
        return {
            'total_emails_retrieved': self.stats['total_emails_retrieved'],
            'job_related_emails': self.stats['job_related_emails'],
            'processing_time': self.stats['processing_time'],
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else [],
            'job_relevance_rate': (self.stats['job_related_emails'] / self.stats['total_emails_retrieved']) * 100 if self.stats['total_emails_retrieved'] > 0 else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset agent statistics."""
        self.stats = {
            'total_emails_retrieved': 0,
            'job_related_emails': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        try:
            await self.gmail_tool._close_session()
            logger.info("Email Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function
def create_email_agent() -> EmailAgent:
    """
    Create and return Email Agent instance.
    
    Returns:
        EmailAgent: Configured Email Agent
    """
    return EmailAgent()

# Test function
async def test_email_agent():
    """Test Email Agent functionality."""
    try:
        agent = create_email_agent()
        
        # Test initialization
        success = await agent.initialize()
        if not success:
            print("❌ Email Agent initialization failed")
            return
        
        print("✅ Email Agent initialization successful")
        
        # Test email retrieval (small batch)
        batch = await agent.retrieve_recent_emails(days_back=7)
        print(f"✅ Retrieved {len(batch.emails)} emails from last 7 days")
        
        if batch.emails:
            # Test filtering
            job_emails = await agent.filter_job_related_emails(batch)
            print(f"✅ Found {len(job_emails)} job-related emails")
            
            if job_emails:
                # Test preprocessing
                preprocessed = await agent.preprocess_emails(job_emails[:3])  # Limit to 3
                print(f"✅ Preprocessed {len(preprocessed)} emails")
                
                # Show sample result
                if preprocessed:
                    sample = preprocessed[0]
                    print(f"📧 Sample result: {sample['company']} - {sample['position']} (confidence: {sample['confidence']:.2f})")
            
            # Test pattern analysis
            patterns = await agent.analyze_email_patterns(batch.emails[:10])  # Limit to 10
            print(f"✅ Pattern analysis: {patterns['unique_domains']} unique domains")
        
        # Test statistics
        stats = agent.get_statistics()
        print(f"📊 Agent stats: {stats['total_emails_retrieved']} emails, {stats['job_relevance_rate']:.1f}% job-related")
        
    except Exception as e:
        print(f"❌ Email Agent test failed: {e}")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(test_email_agent())