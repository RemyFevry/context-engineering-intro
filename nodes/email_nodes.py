"""
PocketFlow nodes for email processing in job application tracking system.

Contains PocketFlow node implementations that wrap the Email Agent
for use in the job application tracking workflow.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from pocketflow import Node, AsyncNode
from agents.email_agent import EmailAgent, create_email_agent
from models.email_models import EmailBatch, create_email_batch
from config.settings import get_settings
from utils import logger

class EmailRetrievalNode(AsyncNode):
    """
    PocketFlow node for retrieving emails from Gmail.
    
    Handles email retrieval using the Email Agent and stores
    results in shared state for downstream processing.
    """
    
    def __init__(self, node_id: str = "email_retrieval"):
        """
        Initialize EmailRetrievalNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.settings = get_settings()
        self.email_agent = None
        
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for email retrieval.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Initialize email agent if not already done
            if not self.email_agent:
                self.email_agent = create_email_agent()
                await self.email_agent.initialize()
            
            # Get parameters from shared state
            days_back = shared.get('days_back', self.settings.email_days_back)
            
            logger.info(f"EmailRetrievalNode: Preparing to retrieve emails from last {days_back} days")
            
            return {
                'days_back': days_back,
                'agent_ready': True
            }
            
        except Exception as e:
            logger.error(f"EmailRetrievalNode prep failed: {e}")
            return {
                'error': str(e),
                'agent_ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute email retrieval.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('agent_ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Agent not ready'),
                'email_batch': None
            }
        
        try:
            days_back = inputs['days_back']
            
            logger.info(f"EmailRetrievalNode: Retrieving emails from last {days_back} days")
            
            # Retrieve emails using email agent
            email_batch = await self.email_agent.retrieve_recent_emails(days_back)
            
            logger.info(f"EmailRetrievalNode: Retrieved {len(email_batch.emails)} emails")
            
            return {
                'success': True,
                'email_batch': email_batch,
                'total_emails': len(email_batch.emails),
                'batch_id': email_batch.batch_id
            }
            
        except Exception as e:
            logger.error(f"EmailRetrievalNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'email_batch': None
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process email retrieval results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store email batch in shared state
                shared['email_batch'] = exec_res['email_batch']
                shared['total_emails_retrieved'] = exec_res['total_emails']
                shared['email_batch_id'] = exec_res['batch_id']
                
                # Store processing metadata
                shared['email_retrieval_completed'] = True
                shared['email_retrieval_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.email_agent:
                    shared['email_agent_stats'] = self.email_agent.get_statistics()
                
                logger.info(f"EmailRetrievalNode: Completed successfully, retrieved {exec_res['total_emails']} emails")
                
                return "success"
            else:
                # Store error information
                shared['email_retrieval_error'] = exec_res.get('error', 'Unknown error')
                shared['email_retrieval_completed'] = False
                
                logger.error(f"EmailRetrievalNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"EmailRetrievalNode post-processing failed: {e}")
            shared['email_retrieval_error'] = str(e)
            shared['email_retrieval_completed'] = False
            return "error"

class EmailFilteringNode(AsyncNode):
    """
    PocketFlow node for filtering job-related emails.
    
    Filters emails from email batch to identify job-related content
    using the Email Agent's filtering capabilities.
    """
    
    def __init__(self, node_id: str = "email_filtering"):
        """
        Initialize EmailFilteringNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.settings = get_settings()
        self.email_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for email filtering.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get email batch from shared state
            email_batch = shared.get('email_batch')
            
            if not email_batch:
                return {
                    'error': 'No email batch available for filtering',
                    'ready': False
                }
            
            # Initialize email agent if not already done
            if not self.email_agent:
                self.email_agent = create_email_agent()
                await self.email_agent.initialize()
            
            logger.info(f"EmailFilteringNode: Preparing to filter {len(email_batch.emails)} emails")
            
            return {
                'email_batch': email_batch,
                'confidence_threshold': self.settings.classification_confidence_threshold,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"EmailFilteringNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute email filtering.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for filtering'),
                'job_related_emails': []
            }
        
        try:
            email_batch = inputs['email_batch']
            
            logger.info(f"EmailFilteringNode: Filtering {len(email_batch.emails)} emails for job-related content")
            
            # Filter job-related emails
            job_related_emails = await self.email_agent.filter_job_related_emails(email_batch)
            
            logger.info(f"EmailFilteringNode: Found {len(job_related_emails)} job-related emails")
            
            return {
                'success': True,
                'job_related_emails': job_related_emails,
                'total_job_emails': len(job_related_emails),
                'filter_ratio': len(job_related_emails) / len(email_batch.emails) if email_batch.emails else 0
            }
            
        except Exception as e:
            logger.error(f"EmailFilteringNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_related_emails': []
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process email filtering results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store filtered emails in shared state
                shared['job_related_emails'] = exec_res['job_related_emails']
                shared['total_job_emails'] = exec_res['total_job_emails']
                shared['filter_ratio'] = exec_res['filter_ratio']
                
                # Store processing metadata
                shared['email_filtering_completed'] = True
                shared['email_filtering_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.email_agent:
                    shared['email_agent_stats'] = self.email_agent.get_statistics()
                
                logger.info(f"EmailFilteringNode: Completed successfully, found {exec_res['total_job_emails']} job-related emails")
                
                return "success"
            else:
                # Store error information
                shared['email_filtering_error'] = exec_res.get('error', 'Unknown error')
                shared['email_filtering_completed'] = False
                
                logger.error(f"EmailFilteringNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"EmailFilteringNode post-processing failed: {e}")
            shared['email_filtering_error'] = str(e)
            shared['email_filtering_completed'] = False
            return "error"

class EmailPreprocessingNode(AsyncNode):
    """
    PocketFlow node for preprocessing job-related emails.
    
    Preprocesses job-related emails to extract job application information
    using the Email Agent's preprocessing capabilities.
    """
    
    def __init__(self, node_id: str = "email_preprocessing"):
        """
        Initialize EmailPreprocessingNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.email_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for email preprocessing.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get job-related emails from shared state
            job_related_emails = shared.get('job_related_emails', [])
            
            if not job_related_emails:
                return {
                    'error': 'No job-related emails available for preprocessing',
                    'ready': False
                }
            
            # Initialize email agent if not already done
            if not self.email_agent:
                self.email_agent = create_email_agent()
                await self.email_agent.initialize()
            
            logger.info(f"EmailPreprocessingNode: Preparing to preprocess {len(job_related_emails)} job-related emails")
            
            return {
                'job_related_emails': job_related_emails,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"EmailPreprocessingNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute email preprocessing.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for preprocessing'),
                'preprocessed_emails': []
            }
        
        try:
            job_related_emails = inputs['job_related_emails']
            
            logger.info(f"EmailPreprocessingNode: Preprocessing {len(job_related_emails)} job-related emails")
            
            # Preprocess emails
            preprocessed_emails = await self.email_agent.preprocess_emails(job_related_emails)
            
            logger.info(f"EmailPreprocessingNode: Preprocessed {len(preprocessed_emails)} emails")
            
            return {
                'success': True,
                'preprocessed_emails': preprocessed_emails,
                'total_preprocessed': len(preprocessed_emails)
            }
            
        except Exception as e:
            logger.error(f"EmailPreprocessingNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'preprocessed_emails': []
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process email preprocessing results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store preprocessed emails in shared state
                shared['preprocessed_emails'] = exec_res['preprocessed_emails']
                shared['total_preprocessed_emails'] = exec_res['total_preprocessed']
                
                # Store processing metadata
                shared['email_preprocessing_completed'] = True
                shared['email_preprocessing_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.email_agent:
                    shared['email_agent_stats'] = self.email_agent.get_statistics()
                
                logger.info(f"EmailPreprocessingNode: Completed successfully, preprocessed {exec_res['total_preprocessed']} emails")
                
                return "success"
            else:
                # Store error information
                shared['email_preprocessing_error'] = exec_res.get('error', 'Unknown error')
                shared['email_preprocessing_completed'] = False
                
                logger.error(f"EmailPreprocessingNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"EmailPreprocessingNode post-processing failed: {e}")
            shared['email_preprocessing_error'] = str(e)
            shared['email_preprocessing_completed'] = False
            return "error"

# Factory functions
def create_email_retrieval_node() -> EmailRetrievalNode:
    """
    Create and return EmailRetrievalNode instance.
    
    Returns:
        EmailRetrievalNode: Configured email retrieval node
    """
    return EmailRetrievalNode()

def create_email_filtering_node() -> EmailFilteringNode:
    """
    Create and return EmailFilteringNode instance.
    
    Returns:
        EmailFilteringNode: Configured email filtering node
    """
    return EmailFilteringNode()

def create_email_preprocessing_node() -> EmailPreprocessingNode:
    """
    Create and return EmailPreprocessingNode instance.
    
    Returns:
        EmailPreprocessingNode: Configured email preprocessing node
    """
    return EmailPreprocessingNode()

# Test function
async def test_email_nodes():
    """Test email nodes functionality."""
    try:
        # Test email retrieval node
        retrieval_node = create_email_retrieval_node()
        
        # Simulate shared state
        shared = {'days_back': 7}
        
        # Test preparation
        prep_result = await retrieval_node.prep_async(shared)
        print(f"✅ Email retrieval prep: {prep_result.get('agent_ready', False)}")
        
        if prep_result.get('agent_ready', False):
            # Test execution (limited to avoid API calls in test)
            print("📧 Email retrieval node prepared successfully")
            
            # Test filtering node
            filtering_node = create_email_filtering_node()
            print("✅ Email filtering node created")
            
            # Test preprocessing node
            preprocessing_node = create_email_preprocessing_node()
            print("✅ Email preprocessing node created")
        
    except Exception as e:
        print(f"❌ Email nodes test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_email_nodes())