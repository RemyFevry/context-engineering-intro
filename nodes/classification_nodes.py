"""
PocketFlow nodes for email classification in job application tracking system.

Contains PocketFlow node implementations that wrap the Classification Agent
for use in the job application tracking workflow.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from pocketflow import Node, AsyncNode
from agents.classification_agent import ClassificationAgent, create_classification_agent
from models.job_models import JobApplication
from config.settings import get_settings
from utils import logger

class ClassificationNode(AsyncNode):
    """
    PocketFlow node for classifying emails as job applications.
    
    Uses the Classification Agent to analyze emails and identify
    job applications with extracted information.
    """
    
    def __init__(self, node_id: str = "classification"):
        """
        Initialize ClassificationNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.settings = get_settings()
        self.classification_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for email classification.
        
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
                    'error': 'No job-related emails available for classification',
                    'ready': False
                }
            
            # Initialize classification agent if not already done
            if not self.classification_agent:
                self.classification_agent = create_classification_agent()
            
            logger.info(f"ClassificationNode: Preparing to classify {len(job_related_emails)} job-related emails")
            
            return {
                'job_related_emails': job_related_emails,
                'confidence_threshold': self.settings.classification_confidence_threshold,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"ClassificationNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute email classification.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for classification'),
                'job_applications': []
            }
        
        try:
            job_related_emails = inputs['job_related_emails']
            
            logger.info(f"ClassificationNode: Classifying {len(job_related_emails)} emails")
            
            # Classify emails in batch
            job_infos = await self.classification_agent.classify_email_batch(job_related_emails)
            
            logger.info(f"ClassificationNode: Classified {len(job_infos)} emails as job applications")
            
            return {
                'success': True,
                'job_applications_info': job_infos,
                'total_classified': len(job_infos)
            }
            
        except Exception as e:
            logger.error(f"ClassificationNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_applications': []
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process classification results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store classification results in shared state
                shared['job_applications_info'] = exec_res['job_applications_info']
                shared['total_classified'] = exec_res['total_classified']
                
                # Store processing metadata
                shared['classification_completed'] = True
                shared['classification_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.classification_agent:
                    shared['classification_agent_stats'] = self.classification_agent.get_statistics()
                
                logger.info(f"ClassificationNode: Completed successfully, classified {exec_res['total_classified']} job applications")
                
                return "success"
            else:
                # Store error information
                shared['classification_error'] = exec_res.get('error', 'Unknown error')
                shared['classification_completed'] = False
                
                logger.error(f"ClassificationNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"ClassificationNode post-processing failed: {e}")
            shared['classification_error'] = str(e)
            shared['classification_completed'] = False
            return "error"

class JobApplicationCreationNode(AsyncNode):
    """
    PocketFlow node for creating JobApplication objects.
    
    Creates JobApplication objects from classification results
    for use in downstream processing.
    """
    
    def __init__(self, node_id: str = "job_application_creation"):
        """
        Initialize JobApplicationCreationNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.classification_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for job application creation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get classification results from shared state
            job_applications_info = shared.get('job_applications_info', [])
            
            if not job_applications_info:
                return {
                    'error': 'No job application information available for creation',
                    'ready': False
                }
            
            # Initialize classification agent if not already done
            if not self.classification_agent:
                self.classification_agent = create_classification_agent()
            
            logger.info(f"JobApplicationCreationNode: Preparing to create {len(job_applications_info)} job applications")
            
            return {
                'job_applications_info': job_applications_info,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"JobApplicationCreationNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute job application creation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for job application creation'),
                'job_applications': []
            }
        
        try:
            job_applications_info = inputs['job_applications_info']
            
            logger.info(f"JobApplicationCreationNode: Creating {len(job_applications_info)} job application objects")
            
            # Create job applications from classification results
            job_applications = await self.classification_agent.create_job_applications(job_applications_info)
            
            logger.info(f"JobApplicationCreationNode: Created {len(job_applications)} job application objects")
            
            return {
                'success': True,
                'job_applications': job_applications,
                'total_created': len(job_applications)
            }
            
        except Exception as e:
            logger.error(f"JobApplicationCreationNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_applications': []
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process job application creation results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store job applications in shared state
                shared['job_applications'] = exec_res['job_applications']
                shared['total_job_applications'] = exec_res['total_created']
                
                # Store processing metadata
                shared['job_application_creation_completed'] = True
                shared['job_application_creation_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.classification_agent:
                    shared['classification_agent_stats'] = self.classification_agent.get_statistics()
                
                logger.info(f"JobApplicationCreationNode: Completed successfully, created {exec_res['total_created']} job applications")
                
                return "success"
            else:
                # Store error information
                shared['job_application_creation_error'] = exec_res.get('error', 'Unknown error')
                shared['job_application_creation_completed'] = False
                
                logger.error(f"JobApplicationCreationNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"JobApplicationCreationNode post-processing failed: {e}")
            shared['job_application_creation_error'] = str(e)
            shared['job_application_creation_completed'] = False
            return "error"

class ClassificationValidationNode(AsyncNode):
    """
    PocketFlow node for validating classification accuracy.
    
    Validates the accuracy of job application classification
    using the Classification Agent's validation capabilities.
    """
    
    def __init__(self, node_id: str = "classification_validation", sample_size: int = 5):
        """
        Initialize ClassificationValidationNode.
        
        Args:
            node_id (str): Unique identifier for this node
            sample_size (int): Number of samples to validate
        """
        super().__init__()
        self.node_id = node_id
        self.sample_size = sample_size
        self.classification_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for classification validation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get classification results from shared state
            job_applications_info = shared.get('job_applications_info', [])
            
            if not job_applications_info:
                return {
                    'error': 'No job application information available for validation',
                    'ready': False
                }
            
            # Initialize classification agent if not already done
            if not self.classification_agent:
                self.classification_agent = create_classification_agent()
            
            logger.info(f"ClassificationValidationNode: Preparing to validate {min(self.sample_size, len(job_applications_info))} classifications")
            
            return {
                'job_applications_info': job_applications_info,
                'sample_size': self.sample_size,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"ClassificationValidationNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute classification validation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for validation'),
                'validation_results': {}
            }
        
        try:
            job_applications_info = inputs['job_applications_info']
            sample_size = inputs['sample_size']
            
            logger.info(f"ClassificationValidationNode: Validating {min(sample_size, len(job_applications_info))} classifications")
            
            # Validate classification accuracy
            validation_results = await self.classification_agent.validate_classification_accuracy(
                job_applications_info, sample_size
            )
            
            logger.info(f"ClassificationValidationNode: Validation completed")
            
            return {
                'success': True,
                'validation_results': validation_results,
                'samples_validated': validation_results.get('sample_size', 0)
            }
            
        except Exception as e:
            logger.error(f"ClassificationValidationNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'validation_results': {}
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process validation results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store validation results in shared state
                shared['classification_validation_results'] = exec_res['validation_results']
                shared['samples_validated'] = exec_res['samples_validated']
                
                # Store processing metadata
                shared['classification_validation_completed'] = True
                shared['classification_validation_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.classification_agent:
                    shared['classification_agent_stats'] = self.classification_agent.get_statistics()
                
                logger.info(f"ClassificationValidationNode: Completed successfully, validated {exec_res['samples_validated']} samples")
                
                return "success"
            else:
                # Store error information
                shared['classification_validation_error'] = exec_res.get('error', 'Unknown error')
                shared['classification_validation_completed'] = False
                
                logger.error(f"ClassificationValidationNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"ClassificationValidationNode post-processing failed: {e}")
            shared['classification_validation_error'] = str(e)
            shared['classification_validation_completed'] = False
            return "error"

# Factory functions
def create_classification_node() -> ClassificationNode:
    """
    Create and return ClassificationNode instance.
    
    Returns:
        ClassificationNode: Configured classification node
    """
    return ClassificationNode()

def create_job_application_creation_node() -> JobApplicationCreationNode:
    """
    Create and return JobApplicationCreationNode instance.
    
    Returns:
        JobApplicationCreationNode: Configured job application creation node
    """
    return JobApplicationCreationNode()

def create_classification_validation_node(sample_size: int = 5) -> ClassificationValidationNode:
    """
    Create and return ClassificationValidationNode instance.
    
    Args:
        sample_size (int): Number of samples to validate
        
    Returns:
        ClassificationValidationNode: Configured validation node
    """
    return ClassificationValidationNode(sample_size=sample_size)

# Test function
async def test_classification_nodes():
    """Test classification nodes functionality."""
    try:
        # Test classification node
        classification_node = create_classification_node()
        
        # Test job application creation node
        job_app_node = create_job_application_creation_node()
        
        # Test validation node
        validation_node = create_classification_validation_node()
        
        print("✅ Classification nodes created successfully")
        print(f"🔍 Classification node ID: {classification_node.node_id}")
        print(f"📋 Job application creation node ID: {job_app_node.node_id}")
        print(f"✅ Validation node ID: {validation_node.node_id}")
        
    except Exception as e:
        print(f"❌ Classification nodes test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_classification_nodes())