"""
Classification Agent for job application tracking system.

Handles job application email classification using LLM analysis to
identify job applications and extract relevant information.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from models.email_models import GmailEmail
from models.job_models import JobApplication, ApplicationStatus
from config.settings import get_settings
from utils import logger, call_llm, parse_yaml_response, clean_text, calculate_confidence_score

class ClassificationAgent:
    """
    Classification Agent for job application email analysis.
    
    Uses LLM analysis to classify emails as job applications and extract
    relevant information like company, position, and application status.
    """
    
    def __init__(self):
        """Initialize Classification Agent with settings."""
        self.settings = get_settings()
        
        # Processing statistics
        self.stats = {
            'total_emails_classified': 0,
            'job_applications_identified': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
        
        # Classification prompts
        self.classification_prompt_template = """
        Analyze this email to determine if it's related to a job application. The email may be in English or French.
        
        Email Details:
        From: {sender}
        Subject: {subject}
        Date: {date}
        Content: {content}
        
        Please analyze this email and determine:
        1. Is this email related to a job application?
        2. If yes, extract the following information:
           - Company name
           - Position/role title
           - Application status (applied, acknowledged, interviewing, rejected, offer, unknown)
           - Key details or context
        3. Provide confidence score (0.0-1.0)
        
        Consider these factors for both English and French emails:
        
        ENGLISH keywords to look for:
        - Sender domains: jobs, careers, hr, talent, recruiting, recruitment, noreply
        - Subject keywords: application, position, interview, role, job, career, hiring, candidate
        - Content keywords: apply, applied, application, interview, position, role, offer, rejection, acknowledged
        - Status indicators: confirmed, received, reviewing, scheduled, invitation, declined, accepted
        
        FRENCH keywords to look for:
        - Sender domains: emplois, carrieres, rh, recrutement, candidature
        - Subject keywords: candidature, poste, entretien, emploi, carrière, embauche, candidat
        - Content keywords: postuler, candidature, entretien, poste, emploi, offre, refus, reçu
        - Status indicators: confirmé, reçu, en cours d'examen, programmé, invitation, refusé, accepté
        
        Also consider:
        - Email tone and structure (formal business communication)
        - Automated vs. personal emails
        - Job board domains (LinkedIn, Indeed, Glassdoor, etc.)
        
        Always respond in English, even if the email is in French.
        
        Output in YAML format:
        ```yaml
        is_job_application: true/false
        confidence: 0.95
        company: "Company Name"
        position: "Job Title"
        status: "applied/acknowledged/interviewing/rejected/offer/unknown"
        details: "Brief description of email content"
        reasoning: "Why this was classified as job-related or not"
        ```
        """
    
    async def classify_email(self, email: GmailEmail) -> Optional[Dict[str, Any]]:
        """
        Classify a single email for job application relevance.
        
        Args:
            email (GmailEmail): Email to classify
            
        Returns:
            Optional[Dict[str, Any]]: Classification result or None if not job-related
        """
        start_time = datetime.now()
        
        try:
            # Prepare content for analysis
            content = self._prepare_email_content(email)
            
            # Create classification prompt
            prompt = self.classification_prompt_template.format(
                sender=email.sender,
                subject=email.subject,
                date=email.received_date.strftime('%Y-%m-%d %H:%M:%S'),
                content=content
            )
            
            logger.debug(f"Classifying email: {email.subject[:50]}...")
            
            # Call LLM for classification
            response = call_llm(prompt)
            
            # Parse response
            classification = parse_yaml_response(response)
            
            # Validate classification
            if not self._validate_classification(classification):
                logger.warning(f"Invalid classification response for email {email.id}")
                self.stats['failed_extractions'] += 1
                return None
            
            # Update statistics
            self.stats['total_emails_classified'] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_time'] += processing_time
            
            # Check if job application
            if not classification.get('is_job_application', False):
                logger.debug(f"Email not classified as job application: {email.subject[:50]}...")
                return None
            
            # Check confidence threshold
            confidence = classification.get('confidence', 0.0)
            if confidence < self.settings.classification_confidence_threshold:
                logger.debug(f"Classification confidence too low: {confidence:.2f} for {email.subject[:50]}...")
                return None
            
            # Extract job application information
            job_info = self._extract_job_information(email, classification)
            
            self.stats['job_applications_identified'] += 1
            self.stats['successful_extractions'] += 1
            
            logger.info(f"Job application identified: {job_info['company']} - {job_info['position']} (confidence: {confidence:.2f})")
            
            return job_info
            
        except Exception as e:
            logger.error(f"Failed to classify email {email.id}: {e}")
            self.stats['failed_extractions'] += 1
            self.stats['errors'].append(f"Classification failed for {email.id}: {e}")
            return None
    
    def _prepare_email_content(self, email: GmailEmail) -> str:
        """
        Prepare email content for LLM analysis.
        
        Args:
            email (GmailEmail): Email to prepare
            
        Returns:
            str: Prepared content
        """
        # Clean and truncate body content
        body = clean_text(email.body)
        
        # Truncate to reasonable length for LLM
        max_length = 2000
        if len(body) > max_length:
            body = body[:max_length] + "..."
        
        # Include relevant headers if available
        headers_info = ""
        if email.raw_headers:
            reply_to = email.raw_headers.get('reply-to', '')
            if reply_to:
                headers_info = f"Reply-To: {reply_to}\n"
        
        return f"{headers_info}{body}"
    
    def _validate_classification(self, classification: Dict[str, Any]) -> bool:
        """
        Validate classification response structure.
        
        Args:
            classification (Dict[str, Any]): Classification response
            
        Returns:
            bool: True if valid
        """
        required_fields = ['is_job_application', 'confidence']
        
        # Check required fields
        for field in required_fields:
            if field not in classification:
                return False
        
        # Validate confidence score
        confidence = classification.get('confidence', 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            return False
        
        # If it's a job application, check additional fields
        if classification.get('is_job_application', False):
            job_fields = ['company', 'position', 'status']
            for field in job_fields:
                if field not in classification or not classification[field]:
                    return False
        
        return True
    
    def _extract_job_information(self, email: GmailEmail, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract job application information from classification.
        
        Args:
            email (GmailEmail): Original email
            classification (Dict[str, Any]): Classification result
            
        Returns:
            Dict[str, Any]: Job application information
        """
        # Extract basic information
        company = classification.get('company', 'Unknown Company').strip()
        position = classification.get('position', 'Unknown Position').strip()
        status_str = classification.get('status', 'unknown').lower()
        confidence = classification.get('confidence', 0.0)
        details = classification.get('details', '')
        reasoning = classification.get('reasoning', '')
        
        # Parse status
        try:
            status = ApplicationStatus(status_str)
        except ValueError:
            status = ApplicationStatus.UNKNOWN
        
        # Additional context factors
        context_factors = {
            'sender_domain': email.sender_domain or 'unknown',
            'is_automated': email.is_automated(),
            'is_job_related_domain': email.is_job_related_domain(),
            'email_labels': email.labels,
            'classification_reasoning': reasoning
        }
        
        return {
            'email_id': email.id,
            'thread_id': email.thread_id,
            'company': company,
            'position': position,
            'status': status,
            'confidence': confidence,
            'details': details,
            'applied_date': email.received_date,
            'last_updated': datetime.now(),
            'context_factors': context_factors,
            'original_subject': email.subject,
            'original_sender': email.sender,
            'classification_timestamp': datetime.now()
        }
    
    async def classify_email_batch(self, emails: List[GmailEmail]) -> List[Dict[str, Any]]:
        """
        Classify a batch of emails for job applications.
        
        Args:
            emails (List[GmailEmail]): Emails to classify
            
        Returns:
            List[Dict[str, Any]]: Job application information for classified emails
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Classifying batch of {len(emails)} emails")
            
            job_applications = []
            
            # Process emails with rate limiting
            for i, email in enumerate(emails):
                try:
                    # Add delay between requests to avoid rate limiting
                    if i > 0:
                        await asyncio.sleep(0.5)  # 500ms delay between requests
                    
                    # Classify email
                    job_info = await self.classify_email(email)
                    
                    if job_info:
                        job_applications.append(job_info)
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(emails)} emails, found {len(job_applications)} job applications")
                
                except Exception as e:
                    logger.error(f"Error processing email {i+1}: {e}")
                    continue
            
            # Update statistics
            self.stats['last_run'] = datetime.now()
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Batch classification complete: {len(job_applications)} job applications found in {processing_time:.2f}s")
            
            return job_applications
            
        except Exception as e:
            logger.error(f"Failed to classify email batch: {e}")
            self.stats['errors'].append(f"Batch classification failed: {e}")
            raise
    
    async def create_job_applications(self, job_infos: List[Dict[str, Any]]) -> List[JobApplication]:
        """
        Create JobApplication objects from classification results.
        
        Args:
            job_infos (List[Dict[str, Any]]): Job application information
            
        Returns:
            List[JobApplication]: Created job applications
        """
        try:
            logger.info(f"Creating {len(job_infos)} job application objects")
            
            job_applications = []
            
            for job_info in job_infos:
                try:
                    # Create job application
                    job_app = JobApplication(
                        email_id=job_info['email_id'],
                        company=job_info['company'],
                        position=job_info['position'],
                        status=job_info['status'],
                        applied_date=job_info['applied_date'],
                        last_updated=job_info['last_updated'],
                        confidence_score=job_info['confidence'],
                        notes=job_info.get('details', ''),
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    # Add classification context as tags
                    context = job_info.get('context_factors', {})
                    if context.get('is_job_related_domain'):
                        job_app.add_tag('job_board')
                    if context.get('is_automated'):
                        job_app.add_tag('automated')
                    
                    job_applications.append(job_app)
                    
                    logger.debug(f"Created job application: {job_app.company} - {job_app.position}")
                    
                except Exception as e:
                    logger.error(f"Failed to create job application from info: {e}")
                    continue
            
            logger.info(f"Successfully created {len(job_applications)} job application objects")
            
            return job_applications
            
        except Exception as e:
            logger.error(f"Failed to create job applications: {e}")
            self.stats['errors'].append(f"Job application creation failed: {e}")
            raise
    
    async def refine_classification(self, job_info: Dict[str, Any], additional_context: str = "") -> Dict[str, Any]:
        """
        Refine classification with additional context.
        
        Args:
            job_info (Dict[str, Any]): Original job information
            additional_context (str): Additional context for refinement
            
        Returns:
            Dict[str, Any]: Refined job information
        """
        try:
            refinement_prompt = f"""
            Refine the classification of this job application with additional context:
            
            Current Classification:
            Company: {job_info['company']}
            Position: {job_info['position']}
            Status: {job_info['status'].value}
            Confidence: {job_info['confidence']}
            Details: {job_info.get('details', '')}
            
            Additional Context:
            {additional_context}
            
            Please provide a refined classification with improved accuracy:
            
            Output in YAML format:
            ```yaml
            company: "Refined Company Name"
            position: "Refined Position Title"
            status: "applied/acknowledged/interviewing/rejected/offer/unknown"
            confidence: 0.95
            details: "Updated details"
            changes_made: "Description of changes made"
            ```
            """
            
            response = await call_llm(refinement_prompt)
            refinement = parse_yaml_response(response)
            
            # Update job info with refinements
            if refinement.get('company'):
                job_info['company'] = refinement['company']
            if refinement.get('position'):
                job_info['position'] = refinement['position']
            if refinement.get('status'):
                try:
                    job_info['status'] = ApplicationStatus(refinement['status'])
                except ValueError:
                    pass
            if refinement.get('confidence'):
                job_info['confidence'] = refinement['confidence']
            if refinement.get('details'):
                job_info['details'] = refinement['details']
            
            # Add refinement notes
            changes = refinement.get('changes_made', '')
            if changes:
                job_info['refinement_notes'] = changes
                job_info['refined_at'] = datetime.now()
            
            logger.info(f"Refined classification for {job_info['company']} - {job_info['position']}")
            
            return job_info
            
        except Exception as e:
            logger.error(f"Failed to refine classification: {e}")
            return job_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dict[str, Any]: Agent statistics
        """
        return {
            'total_emails_classified': self.stats['total_emails_classified'],
            'job_applications_identified': self.stats['job_applications_identified'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'processing_time': self.stats['processing_time'],
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else [],
            'success_rate': (self.stats['successful_extractions'] / self.stats['total_emails_classified']) * 100 if self.stats['total_emails_classified'] > 0 else 0,
            'job_identification_rate': (self.stats['job_applications_identified'] / self.stats['total_emails_classified']) * 100 if self.stats['total_emails_classified'] > 0 else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset agent statistics."""
        self.stats = {
            'total_emails_classified': 0,
            'job_applications_identified': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
    
    async def validate_classification_accuracy(self, job_infos: List[Dict[str, Any]], 
                                             sample_size: int = 5) -> Dict[str, Any]:
        """
        Validate classification accuracy on a sample.
        
        Args:
            job_infos (List[Dict[str, Any]]): Job application information
            sample_size (int): Number of samples to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            import random
            
            # Sample for validation
            sample = random.sample(job_infos, min(sample_size, len(job_infos)))
            
            validation_results = []
            
            for job_info in sample:
                validation_prompt = f"""
                Validate this job application classification:
                
                Company: {job_info['company']}
                Position: {job_info['position']}
                Status: {job_info['status'].value}
                Confidence: {job_info['confidence']}
                
                Original Email Subject: {job_info['original_subject']}
                Original Sender: {job_info['original_sender']}
                
                Rate the accuracy of this classification:
                - Company accuracy (0-10)
                - Position accuracy (0-10)
                - Status accuracy (0-10)
                - Overall accuracy (0-10)
                
                Output in YAML format:
                ```yaml
                company_accuracy: 8
                position_accuracy: 9
                status_accuracy: 7
                overall_accuracy: 8
                feedback: "Brief feedback on classification quality"
                ```
                """
                
                response = await call_llm(validation_prompt)
                validation = parse_yaml_response(response)
                
                validation_results.append({
                    'job_info': job_info,
                    'validation': validation
                })
                
                # Add delay
                await asyncio.sleep(0.5)
            
            # Calculate average scores
            avg_company = sum(r['validation'].get('company_accuracy', 0) for r in validation_results) / len(validation_results)
            avg_position = sum(r['validation'].get('position_accuracy', 0) for r in validation_results) / len(validation_results)
            avg_status = sum(r['validation'].get('status_accuracy', 0) for r in validation_results) / len(validation_results)
            avg_overall = sum(r['validation'].get('overall_accuracy', 0) for r in validation_results) / len(validation_results)
            
            return {
                'sample_size': len(validation_results),
                'average_scores': {
                    'company_accuracy': avg_company,
                    'position_accuracy': avg_position,
                    'status_accuracy': avg_status,
                    'overall_accuracy': avg_overall
                },
                'validation_results': validation_results
            }
            
        except Exception as e:
            logger.error(f"Failed to validate classification accuracy: {e}")
            return {}

# Factory function
def create_classification_agent() -> ClassificationAgent:
    """
    Create and return Classification Agent instance.
    
    Returns:
        ClassificationAgent: Configured Classification Agent
    """
    return ClassificationAgent()

# Test function
async def test_classification_agent():
    """Test Classification Agent functionality."""
    try:
        agent = create_classification_agent()
        
        # Create test email
        from datetime import datetime, timezone
        test_email = GmailEmail(
            id="test_123",
            thread_id="thread_456",
            sender="recruiter@google.com",
            subject="Thank you for your application - Software Engineer Position",
            body="Dear Candidate, We have received your application for the Software Engineer position at Google. We will review your application and get back to you within 2 weeks.",
            received_date=datetime.now(timezone.utc),
            labels=["INBOX"],
            format="plain",
            raw_headers={"from": "recruiter@google.com"}
        )
        
        print("✅ Test email created")
        
        # Test classification
        job_info = await agent.classify_email(test_email)
        
        if job_info:
            print("✅ Email classified as job application")
            print(f"🏢 Company: {job_info['company']}")
            print(f"💼 Position: {job_info['position']}")
            print(f"📊 Status: {job_info['status'].value}")
            print(f"🎯 Confidence: {job_info['confidence']:.2f}")
            
            # Test job application creation
            job_apps = await agent.create_job_applications([job_info])
            print(f"✅ Created {len(job_apps)} job application objects")
            
            if job_apps:
                job_app = job_apps[0]
                print(f"📋 Job application: {job_app.company} - {job_app.position}")
        else:
            print("❌ Email not classified as job application")
        
        # Test statistics
        stats = agent.get_statistics()
        print(f"📊 Agent stats: {stats['total_emails_classified']} classified, {stats['success_rate']:.1f}% success rate")
        
    except Exception as e:
        print(f"❌ Classification Agent test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_classification_agent())