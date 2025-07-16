"""
Main flow configuration for job application tracking system.

Orchestrates the PocketFlow nodes to create a complete workflow for
retrieving, analyzing, and reporting on job applications.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from pocketflow import Flow

# Import all nodes
from nodes.email_nodes import (
    create_email_retrieval_node,
    create_email_filtering_node,
    create_email_preprocessing_node
)
from nodes.classification_nodes import (
    create_classification_node,
    create_job_application_creation_node,
    create_classification_validation_node
)
from nodes.research_nodes import (
    create_research_node,
    create_company_insights_node,
    create_research_aggregation_node
)
from nodes.status_nodes import (
    create_status_node,
    create_insights_generation_node,
    create_final_report_node
)

from config.settings import get_settings
from utils import logger


class JobApplicationTrackingFlow:
    """
    Main flow orchestrator for job application tracking.
    
    Manages the complete workflow from email retrieval to final reporting,
    coordinating all agents and nodes in the proper sequence.
    """
    
    def __init__(self):
        """Initialize the flow with all nodes and configuration."""
        self.settings = get_settings()
        # Flow configuration will be handled directly
        
        # Initialize all nodes
        self.nodes = {
            # Email processing nodes
            'email_retrieval': create_email_retrieval_node(),
            'email_filtering': create_email_filtering_node(),
            'email_preprocessing': create_email_preprocessing_node(),
            
            # Classification nodes
            'classification': create_classification_node(),
            'job_application_creation': create_job_application_creation_node(),
            'classification_validation': create_classification_validation_node(),
            
            # Research nodes
            'research': create_research_node(),
            'company_insights': create_company_insights_node(),
            'research_aggregation': create_research_aggregation_node(),
            
            # Status and reporting nodes
            'status': create_status_node(),
            'insights_generation': create_insights_generation_node(),
            'final_report': create_final_report_node()
        }
        
        # Flow statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'average_runtime': 0.0,
            'errors': []
        }
    
    async def run_full_workflow(self, days_back: int = None) -> Dict[str, Any]:
        """
        Execute the complete job application tracking workflow.
        
        Args:
            days_back (int, optional): Number of days back to retrieve emails
            
        Returns:
            Dict[str, Any]: Complete workflow results
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting job application tracking workflow")
            
            # Initialize shared state
            shared_state = {
                'workflow_started': datetime.now(),
                'days_back': days_back or self.settings.email_days_back,
                'workflow_id': f"job_tracking_{int(datetime.now().timestamp())}"
            }
            
            # Phase 1: Email Processing
            logger.info("Phase 1: Email Processing")
            email_result = await self._run_email_processing_phase(shared_state)
            
            if not email_result.get('success', False):
                return self._create_error_result("Email processing failed", email_result, shared_state)
            
            # Phase 2: Classification
            logger.info("Phase 2: Email Classification")
            classification_result = await self._run_classification_phase(shared_state)
            
            if not classification_result.get('success', False):
                return self._create_error_result("Classification failed", classification_result, shared_state)
            
            # Phase 3: Research (optional, depends on settings)
            research_result = {'success': True}
            if self.settings.brave_api_key:
                logger.info("Phase 3: Job Research")
                research_result = await self._run_research_phase(shared_state)
            else:
                logger.info("Phase 3: Skipping research (no Brave API key)")
            
            # Phase 4: Status and Reporting
            logger.info("Phase 4: Status and Reporting")
            reporting_result = await self._run_reporting_phase(shared_state)
            
            if not reporting_result.get('success', False):
                return self._create_error_result("Reporting failed", reporting_result, shared_state)
            
            # Compile final results
            final_result = await self._compile_final_results(shared_state)
            
            # Update statistics
            self._update_success_stats(start_time)
            
            logger.info("Job application tracking workflow completed successfully")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self._update_error_stats(start_time, str(e))
            return self._create_error_result("Workflow execution failed", {'error': str(e)}, shared_state)
    
    async def _run_email_processing_phase(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute email processing phase."""
        try:
            # Step 1: Email Retrieval
            logger.info("Step 1: Retrieving emails")
            
            retrieval_prep = await self.nodes['email_retrieval'].prep_async(shared_state)
            retrieval_exec = await self.nodes['email_retrieval'].exec_async(retrieval_prep)
            retrieval_action = await self.nodes['email_retrieval'].post_async(
                shared_state, retrieval_prep, retrieval_exec
            )
            
            if retrieval_action != "success":
                return {'success': False, 'error': 'Email retrieval failed', 'step': 'retrieval'}
            
            # Step 2: Email Filtering
            logger.info("Step 2: Filtering job-related emails")
            
            filtering_prep = await self.nodes['email_filtering'].prep_async(shared_state)
            filtering_exec = await self.nodes['email_filtering'].exec_async(filtering_prep)
            filtering_action = await self.nodes['email_filtering'].post_async(
                shared_state, filtering_prep, filtering_exec
            )
            
            if filtering_action != "success":
                return {'success': False, 'error': 'Email filtering failed', 'step': 'filtering'}
            
            # Step 3: Email Preprocessing
            logger.info("Step 3: Preprocessing emails")
            
            preprocessing_prep = await self.nodes['email_preprocessing'].prep_async(shared_state)
            preprocessing_exec = await self.nodes['email_preprocessing'].exec_async(preprocessing_prep)
            preprocessing_action = await self.nodes['email_preprocessing'].post_async(
                shared_state, preprocessing_prep, preprocessing_exec
            )
            
            if preprocessing_action != "success":
                return {'success': False, 'error': 'Email preprocessing failed', 'step': 'preprocessing'}
            
            return {'success': True, 'phase': 'email_processing'}
            
        except Exception as e:
            logger.error(f"Email processing phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'email_processing'}
    
    async def _run_classification_phase(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute classification phase."""
        try:
            # Step 1: Email Classification
            logger.info("Step 1: Classifying emails as job applications")
            
            classification_prep = await self.nodes['classification'].prep_async(shared_state)
            classification_exec = await self.nodes['classification'].exec_async(classification_prep)
            classification_action = await self.nodes['classification'].post_async(
                shared_state, classification_prep, classification_exec
            )
            
            if classification_action != "success":
                return {'success': False, 'error': 'Email classification failed', 'step': 'classification'}
            
            # Step 2: Job Application Creation
            logger.info("Step 2: Creating job application objects")
            
            creation_prep = await self.nodes['job_application_creation'].prep_async(shared_state)
            creation_exec = await self.nodes['job_application_creation'].exec_async(creation_prep)
            creation_action = await self.nodes['job_application_creation'].post_async(
                shared_state, creation_prep, creation_exec
            )
            
            if creation_action != "success":
                return {'success': False, 'error': 'Job application creation failed', 'step': 'creation'}
            
            # Step 3: Classification Validation (optional)
            if self.settings.enable_validation:
                logger.info("Step 3: Validating classification accuracy")
                
                validation_prep = await self.nodes['classification_validation'].prep_async(shared_state)
                validation_exec = await self.nodes['classification_validation'].exec_async(validation_prep)
                validation_action = await self.nodes['classification_validation'].post_async(
                    shared_state, validation_prep, validation_exec
                )
                
                if validation_action != "success":
                    logger.warning("Classification validation failed, continuing workflow")
            
            return {'success': True, 'phase': 'classification'}
            
        except Exception as e:
            logger.error(f"Classification phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'classification'}
    
    async def _run_research_phase(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research phase."""
        try:
            # Step 1: Job Research
            logger.info("Step 1: Researching job applications")
            
            research_prep = await self.nodes['research'].prep_async(shared_state)
            research_exec = await self.nodes['research'].exec_async(research_prep)
            research_action = await self.nodes['research'].post_async(
                shared_state, research_prep, research_exec
            )
            
            if research_action != "success":
                logger.warning("Job research failed, continuing without research data")
                return {'success': True, 'phase': 'research', 'warning': 'Research failed'}
            
            # Step 2: Company Insights
            logger.info("Step 2: Gathering company insights")
            
            insights_prep = await self.nodes['company_insights'].prep_async(shared_state)
            insights_exec = await self.nodes['company_insights'].exec_async(insights_prep)
            insights_action = await self.nodes['company_insights'].post_async(
                shared_state, insights_prep, insights_exec
            )
            
            if insights_action != "success":
                logger.warning("Company insights failed, continuing with available data")
            
            # Step 3: Research Aggregation
            logger.info("Step 3: Aggregating research results")
            
            aggregation_prep = await self.nodes['research_aggregation'].prep_async(shared_state)
            aggregation_exec = await self.nodes['research_aggregation'].exec_async(aggregation_prep)
            aggregation_action = await self.nodes['research_aggregation'].post_async(
                shared_state, aggregation_prep, aggregation_exec
            )
            
            if aggregation_action != "success":
                logger.warning("Research aggregation failed, continuing with raw research data")
            
            return {'success': True, 'phase': 'research'}
            
        except Exception as e:
            logger.error(f"Research phase failed: {e}")
            return {'success': True, 'phase': 'research', 'error': str(e), 'warning': 'Research phase failed'}
    
    async def _run_reporting_phase(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reporting phase."""
        try:
            # Step 1: Status Reporting
            logger.info("Step 1: Generating status reports")
            
            status_prep = await self.nodes['status'].prep_async(shared_state)
            status_exec = await self.nodes['status'].exec_async(status_prep)
            status_action = await self.nodes['status'].post_async(
                shared_state, status_prep, status_exec
            )
            
            if status_action != "success":
                return {'success': False, 'error': 'Status reporting failed', 'step': 'status'}
            
            # Step 2: Insights Generation
            if self.settings.openai_api_key:
                logger.info("Step 2: Generating AI insights")
                
                insights_prep = await self.nodes['insights_generation'].prep_async(shared_state)
                insights_exec = await self.nodes['insights_generation'].exec_async(insights_prep)
                insights_action = await self.nodes['insights_generation'].post_async(
                    shared_state, insights_prep, insights_exec
                )
                
                if insights_action != "success":
                    logger.warning("AI insights generation failed, continuing with basic reporting")
            else:
                logger.info("Step 2: Skipping AI insights (no OpenAI API key)")
            
            # Step 3: Final Report Generation
            logger.info("Step 3: Creating final report")
            
            final_prep = await self.nodes['final_report'].prep_async(shared_state)
            final_exec = await self.nodes['final_report'].exec_async(final_prep)
            final_action = await self.nodes['final_report'].post_async(
                shared_state, final_prep, final_exec
            )
            
            if final_action != "success":
                return {'success': False, 'error': 'Final report generation failed', 'step': 'final_report'}
            
            return {'success': True, 'phase': 'reporting'}
            
        except Exception as e:
            logger.error(f"Reporting phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'reporting'}
    
    async def _compile_final_results(self, shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final workflow results."""
        return {
            'success': True,
            'workflow_id': shared_state.get('workflow_id'),
            'execution_time': (datetime.now() - shared_state['workflow_started']).total_seconds(),
            'summary': {
                'total_emails_retrieved': shared_state.get('total_emails_retrieved', 0),
                'job_related_emails': shared_state.get('total_job_emails', 0),
                'job_applications_created': shared_state.get('total_job_applications', 0),
                'research_completed': shared_state.get('research_completed', False),
                'final_report_generated': shared_state.get('final_report_completed', False)
            },
            'results': {
                'job_applications': shared_state.get('job_applications', []),
                'status_report': shared_state.get('status_report', {}),
                'ai_insights': shared_state.get('ai_insights', {}),
                'research_insights': shared_state.get('aggregated_research_insights', {}),
                'final_report': shared_state.get('final_report', {})
            },
            'metadata': {
                'workflow_completed': shared_state.get('workflow_completed', False),
                'completion_timestamp': shared_state.get('workflow_completion_timestamp'),
                'agent_statistics': {
                    'email_agent': shared_state.get('email_agent_stats', {}),
                    'classification_agent': shared_state.get('classification_agent_stats', {}),
                    'research_agent': shared_state.get('research_agent_stats', {}),
                    'status_agent': shared_state.get('status_agent_stats', {})
                }
            }
        }
    
    def _create_error_result(self, message: str, error_details: Dict[str, Any], 
                           shared_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'success': False,
            'error': message,
            'error_details': error_details,
            'workflow_id': shared_state.get('workflow_id'),
            'execution_time': (datetime.now() - shared_state['workflow_started']).total_seconds(),
            'partial_results': {
                'total_emails_retrieved': shared_state.get('total_emails_retrieved', 0),
                'job_related_emails': shared_state.get('total_job_emails', 0),
                'job_applications_created': shared_state.get('total_job_applications', 0)
            }
        }
    
    def _update_success_stats(self, start_time: datetime) -> None:
        """Update success statistics."""
        self.stats['total_runs'] += 1
        self.stats['successful_runs'] += 1
        self.stats['last_run'] = datetime.now()
        
        runtime = (datetime.now() - start_time).total_seconds()
        self.stats['average_runtime'] = (
            (self.stats['average_runtime'] * (self.stats['total_runs'] - 1) + runtime) / 
            self.stats['total_runs']
        )
    
    def _update_error_stats(self, start_time: datetime, error: str) -> None:
        """Update error statistics."""
        self.stats['total_runs'] += 1
        self.stats['failed_runs'] += 1
        self.stats['last_run'] = datetime.now()
        self.stats['errors'].append({
            'timestamp': datetime.now(),
            'error': error,
            'runtime': (datetime.now() - start_time).total_seconds()
        })
        
        # Keep only last 10 errors
        if len(self.stats['errors']) > 10:
            self.stats['errors'] = self.stats['errors'][-10:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get flow statistics.
        
        Returns:
            Dict[str, Any]: Flow execution statistics
        """
        return {
            'total_runs': self.stats['total_runs'],
            'successful_runs': self.stats['successful_runs'],
            'failed_runs': self.stats['failed_runs'],
            'success_rate': (self.stats['successful_runs'] / self.stats['total_runs']) 
                          if self.stats['total_runs'] > 0 else 0.0,
            'average_runtime': self.stats['average_runtime'],
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else []
        }
    
    def reset_statistics(self) -> None:
        """Reset flow statistics."""
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'last_run': None,
            'average_runtime': 0.0,
            'errors': []
        }


# Factory function
def create_job_application_flow() -> JobApplicationTrackingFlow:
    """
    Create and return JobApplicationTrackingFlow instance.
    
    Returns:
        JobApplicationTrackingFlow: Configured flow instance
    """
    return JobApplicationTrackingFlow()


# Convenience function for direct execution
async def run_job_application_tracking(days_back: int = None) -> Dict[str, Any]:
    """
    Run job application tracking workflow.
    
    Args:
        days_back (int, optional): Number of days back to retrieve emails
        
    Returns:
        Dict[str, Any]: Workflow results
    """
    flow = create_job_application_flow()
    return await flow.run_full_workflow(days_back)


# Test function
async def test_flow():
    """Test the job application tracking flow."""
    try:
        logger.info("Testing job application tracking flow")
        
        # Create flow instance
        flow = create_job_application_flow()
        
        # Check node initialization
        assert len(flow.nodes) == 12, f"Expected 12 nodes, got {len(flow.nodes)}"
        
        # Check node types
        required_nodes = [
            'email_retrieval', 'email_filtering', 'email_preprocessing',
            'classification', 'job_application_creation', 'classification_validation',
            'research', 'company_insights', 'research_aggregation',
            'status', 'insights_generation', 'final_report'
        ]
        
        for node_name in required_nodes:
            assert node_name in flow.nodes, f"Missing node: {node_name}"
        
        print("✅ Flow initialization successful")
        print(f"📊 Flow has {len(flow.nodes)} nodes configured")
        
        # Test statistics
        stats = flow.get_statistics()
        print(f"📊 Initial statistics: {stats['total_runs']} runs, {stats['success_rate']:.1%} success rate")
        
        # Note: Full workflow test would require API keys and real data
        print("⚠️  Full workflow test skipped (requires API keys and email data)")
        
    except Exception as e:
        print(f"❌ Flow test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_flow())