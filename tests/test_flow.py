"""
Tests for flow orchestration and workflow management.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone
import asyncio

from flow import (
    JobApplicationTrackingFlow,
    create_job_application_flow,
    run_job_application_tracking
)
from models.job_models import JobApplication, ApplicationStatus
from models.email_models import GmailEmail, EmailBatch


class TestJobApplicationTrackingFlow:
    """Test JobApplicationTrackingFlow class."""
    
    def test_flow_initialization(self):
        """Test flow initialization."""
        flow = JobApplicationTrackingFlow()
        
        # Check that all required nodes are initialized
        expected_nodes = [
            'email_retrieval', 'email_filtering', 'email_preprocessing',
            'classification', 'job_application_creation', 'classification_validation',
            'research', 'company_insights', 'research_aggregation',
            'status', 'insights_generation', 'final_report'
        ]
        
        assert len(flow.nodes) == len(expected_nodes)
        for node_name in expected_nodes:
            assert node_name in flow.nodes
            assert flow.nodes[node_name] is not None
        
        # Check statistics initialization
        assert flow.stats['total_runs'] == 0
        assert flow.stats['successful_runs'] == 0
        assert flow.stats['failed_runs'] == 0
        assert flow.stats['last_run'] is None
        assert flow.stats['average_runtime'] == 0.0
        assert flow.stats['errors'] == []
    
    @pytest.mark.asyncio
    async def test_run_full_workflow_success(self):
        """Test successful full workflow execution."""
        flow = JobApplicationTrackingFlow()
        
        # Mock all phases to return success
        with patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email, \
             patch.object(flow, '_run_classification_phase', new_callable=AsyncMock) as mock_classification, \
             patch.object(flow, '_run_research_phase', new_callable=AsyncMock) as mock_research, \
             patch.object(flow, '_run_reporting_phase', new_callable=AsyncMock) as mock_reporting, \
             patch.object(flow, '_compile_final_results', new_callable=AsyncMock) as mock_compile:
            
            # Set up mock returns
            mock_email.return_value = {'success': True}
            mock_classification.return_value = {'success': True}
            mock_research.return_value = {'success': True}
            mock_reporting.return_value = {'success': True}
            mock_compile.return_value = {'success': True, 'workflow_id': 'test_id'}
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify all phases were called
            mock_email.assert_called_once()
            mock_classification.assert_called_once()
            mock_research.assert_called_once()
            mock_reporting.assert_called_once()
            mock_compile.assert_called_once()
            
            # Verify result
            assert result['success'] is True
            assert result['workflow_id'] == 'test_id'
            
            # Verify statistics were updated
            assert flow.stats['total_runs'] == 1
            assert flow.stats['successful_runs'] == 1
            assert flow.stats['failed_runs'] == 0
            assert flow.stats['last_run'] is not None
    
    @pytest.mark.asyncio
    async def test_run_full_workflow_email_failure(self):
        """Test workflow with email processing failure."""
        flow = JobApplicationTrackingFlow()
        
        # Mock email phase to fail
        with patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email:
            mock_email.return_value = {'success': False, 'error': 'Email processing failed'}
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify result
            assert result['success'] is False
            assert 'Email processing failed' in result['error']
            
            # Verify statistics were updated
            assert flow.stats['total_runs'] == 1
            assert flow.stats['successful_runs'] == 0
            assert flow.stats['failed_runs'] == 1
            assert len(flow.stats['errors']) == 1
    
    @pytest.mark.asyncio
    async def test_run_full_workflow_classification_failure(self):
        """Test workflow with classification failure."""
        flow = JobApplicationTrackingFlow()
        
        # Mock phases
        with patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email, \
             patch.object(flow, '_run_classification_phase', new_callable=AsyncMock) as mock_classification:
            
            mock_email.return_value = {'success': True}
            mock_classification.return_value = {'success': False, 'error': 'Classification failed'}
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify result
            assert result['success'] is False
            assert 'Classification failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_run_full_workflow_research_skip(self):
        """Test workflow with research skipped (no API key)."""
        flow = JobApplicationTrackingFlow()
        
        # Mock settings to have no Brave API key
        with patch.object(flow.settings, 'brave_api_key', None), \
             patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email, \
             patch.object(flow, '_run_classification_phase', new_callable=AsyncMock) as mock_classification, \
             patch.object(flow, '_run_reporting_phase', new_callable=AsyncMock) as mock_reporting, \
             patch.object(flow, '_compile_final_results', new_callable=AsyncMock) as mock_compile:
            
            mock_email.return_value = {'success': True}
            mock_classification.return_value = {'success': True}
            mock_reporting.return_value = {'success': True}
            mock_compile.return_value = {'success': True, 'workflow_id': 'test_id'}
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify result
            assert result['success'] is True
            
            # Research should have been skipped
            assert flow.stats['successful_runs'] == 1
    
    @pytest.mark.asyncio
    async def test_run_full_workflow_exception(self):
        """Test workflow with unexpected exception."""
        flow = JobApplicationTrackingFlow()
        
        # Mock email phase to raise exception
        with patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email:
            mock_email.side_effect = Exception("Unexpected error")
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify result
            assert result['success'] is False
            assert 'Workflow execution failed' in result['error']
            
            # Verify statistics were updated
            assert flow.stats['failed_runs'] == 1
            assert len(flow.stats['errors']) == 1
    
    @pytest.mark.asyncio
    async def test_run_email_processing_phase_success(self):
        """Test successful email processing phase."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'days_back': 7}
        
        # Mock all nodes to return success
        with patch.object(flow.nodes['email_retrieval'], 'prep_async', new_callable=AsyncMock) as mock_prep, \
             patch.object(flow.nodes['email_retrieval'], 'exec_async', new_callable=AsyncMock) as mock_exec, \
             patch.object(flow.nodes['email_retrieval'], 'post_async', new_callable=AsyncMock) as mock_post, \
             patch.object(flow.nodes['email_filtering'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['email_filtering'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['email_filtering'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['email_preprocessing'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['email_preprocessing'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['email_preprocessing'], 'post_async', new_callable=AsyncMock):
            
            # Set up mock returns
            mock_prep.return_value = {'ready': True}
            mock_exec.return_value = {'success': True}
            mock_post.return_value = 'success'
            
            # Mock all other nodes similarly
            for node_name in ['email_filtering', 'email_preprocessing']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run email processing phase
            result = await flow._run_email_processing_phase(shared_state)
            
            # Verify result
            assert result['success'] is True
            assert result['phase'] == 'email_processing'
    
    @pytest.mark.asyncio
    async def test_run_email_processing_phase_failure(self):
        """Test email processing phase with failure."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'days_back': 7}
        
        # Mock retrieval node to fail
        with patch.object(flow.nodes['email_retrieval'], 'prep_async', new_callable=AsyncMock) as mock_prep, \
             patch.object(flow.nodes['email_retrieval'], 'exec_async', new_callable=AsyncMock) as mock_exec, \
             patch.object(flow.nodes['email_retrieval'], 'post_async', new_callable=AsyncMock) as mock_post:
            
            mock_prep.return_value = {'ready': True}
            mock_exec.return_value = {'success': True}
            mock_post.return_value = 'error'  # Failure
            
            # Run email processing phase
            result = await flow._run_email_processing_phase(shared_state)
            
            # Verify result
            assert result['success'] is False
            assert 'Email retrieval failed' in result['error']
            assert result['step'] == 'retrieval'
    
    @pytest.mark.asyncio
    async def test_run_classification_phase_success(self):
        """Test successful classification phase."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'job_related_emails': ['email1', 'email2']}
        
        # Mock all classification nodes to return success
        with patch.object(flow.nodes['classification'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['classification'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['classification'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['job_application_creation'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['job_application_creation'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['job_application_creation'], 'post_async', new_callable=AsyncMock):
            
            # Set up mock returns
            for node_name in ['classification', 'job_application_creation']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run classification phase
            result = await flow._run_classification_phase(shared_state)
            
            # Verify result
            assert result['success'] is True
            assert result['phase'] == 'classification'
    
    @pytest.mark.asyncio
    async def test_run_research_phase_success(self):
        """Test successful research phase."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'job_applications': ['app1', 'app2']}
        
        # Mock all research nodes to return success
        with patch.object(flow.nodes['research'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'post_async', new_callable=AsyncMock):
            
            # Set up mock returns
            for node_name in ['research', 'company_insights', 'research_aggregation']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run research phase
            result = await flow._run_research_phase(shared_state)
            
            # Verify result
            assert result['success'] is True
            assert result['phase'] == 'research'
    
    @pytest.mark.asyncio
    async def test_run_research_phase_with_warnings(self):
        """Test research phase with warnings (partial failures)."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'job_applications': ['app1', 'app2']}
        
        # Mock research node to fail but others succeed
        with patch.object(flow.nodes['research'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['company_insights'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['research_aggregation'], 'post_async', new_callable=AsyncMock):
            
            # Research node fails
            flow.nodes['research'].prep_async.return_value = {'ready': True}
            flow.nodes['research'].exec_async.return_value = {'success': True}
            flow.nodes['research'].post_async.return_value = 'error'
            
            # Others succeed
            for node_name in ['company_insights', 'research_aggregation']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run research phase
            result = await flow._run_research_phase(shared_state)
            
            # Verify result (should still succeed with warnings)
            assert result['success'] is True
            assert result['phase'] == 'research'
    
    @pytest.mark.asyncio
    async def test_run_reporting_phase_success(self):
        """Test successful reporting phase."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'job_applications': ['app1', 'app2']}
        
        # Mock reporting nodes to return success
        with patch.object(flow.nodes['status'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['status'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['status'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['insights_generation'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['insights_generation'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['insights_generation'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'post_async', new_callable=AsyncMock):
            
            # Set up mock returns
            for node_name in ['status', 'insights_generation', 'final_report']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run reporting phase
            result = await flow._run_reporting_phase(shared_state)
            
            # Verify result
            assert result['success'] is True
            assert result['phase'] == 'reporting'
    
    @pytest.mark.asyncio
    async def test_run_reporting_phase_no_openai_key(self):
        """Test reporting phase with no OpenAI API key."""
        flow = JobApplicationTrackingFlow()
        shared_state = {'job_applications': ['app1', 'app2']}
        
        # Mock settings to have no OpenAI API key
        with patch.object(flow.settings, 'openai_api_key', None), \
             patch.object(flow.nodes['status'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['status'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['status'], 'post_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'prep_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'exec_async', new_callable=AsyncMock), \
             patch.object(flow.nodes['final_report'], 'post_async', new_callable=AsyncMock):
            
            # Set up mock returns
            for node_name in ['status', 'final_report']:
                flow.nodes[node_name].prep_async.return_value = {'ready': True}
                flow.nodes[node_name].exec_async.return_value = {'success': True}
                flow.nodes[node_name].post_async.return_value = 'success'
            
            # Run reporting phase
            result = await flow._run_reporting_phase(shared_state)
            
            # Verify result
            assert result['success'] is True
            assert result['phase'] == 'reporting'
    
    @pytest.mark.asyncio
    async def test_compile_final_results(self):
        """Test compiling final results."""
        flow = JobApplicationTrackingFlow()
        shared_state = {
            'workflow_id': 'test_workflow',
            'workflow_started': datetime.now(timezone.utc),
            'total_emails_retrieved': 100,
            'total_job_emails': 20,
            'total_job_applications': 15,
            'research_completed': True,
            'final_report_completed': True,
            'job_applications': ['app1', 'app2'],
            'status_report': {'summary': 'test'},
            'ai_insights': {'insights': 'test'},
            'aggregated_research_insights': {'insights': 'test'},
            'final_report': {'report': 'test'},
            'workflow_completed': True,
            'workflow_completion_timestamp': datetime.now(timezone.utc),
            'email_agent_stats': {'stat': 'test'},
            'classification_agent_stats': {'stat': 'test'},
            'research_agent_stats': {'stat': 'test'},
            'status_agent_stats': {'stat': 'test'}
        }
        
        result = await flow._compile_final_results(shared_state)
        
        assert result['success'] is True
        assert result['workflow_id'] == 'test_workflow'
        assert result['summary']['total_emails_retrieved'] == 100
        assert result['summary']['job_related_emails'] == 20
        assert result['summary']['job_applications_created'] == 15
        assert result['summary']['research_completed'] is True
        assert result['summary']['final_report_generated'] is True
        assert 'execution_time' in result
        assert 'results' in result
        assert 'metadata' in result
    
    def test_create_error_result(self):
        """Test error result creation."""
        flow = JobApplicationTrackingFlow()
        shared_state = {
            'workflow_id': 'test_workflow',
            'workflow_started': datetime.now(timezone.utc),
            'total_emails_retrieved': 50,
            'total_job_emails': 10,
            'total_job_applications': 5
        }
        
        error_result = flow._create_error_result(
            "Test error", 
            {"error_code": "TEST_ERROR"}, 
            shared_state
        )
        
        assert error_result['success'] is False
        assert error_result['error'] == "Test error"
        assert error_result['error_details']['error_code'] == "TEST_ERROR"
        assert error_result['workflow_id'] == 'test_workflow'
        assert error_result['partial_results']['total_emails_retrieved'] == 50
        assert error_result['partial_results']['job_related_emails'] == 10
        assert error_result['partial_results']['job_applications_created'] == 5
    
    def test_update_success_stats(self):
        """Test updating success statistics."""
        flow = JobApplicationTrackingFlow()
        start_time = datetime.now(timezone.utc)
        
        # Initial stats
        assert flow.stats['total_runs'] == 0
        assert flow.stats['successful_runs'] == 0
        assert flow.stats['average_runtime'] == 0.0
        
        # Update stats
        flow._update_success_stats(start_time)
        
        assert flow.stats['total_runs'] == 1
        assert flow.stats['successful_runs'] == 1
        assert flow.stats['average_runtime'] > 0.0
        assert flow.stats['last_run'] is not None
    
    def test_update_error_stats(self):
        """Test updating error statistics."""
        flow = JobApplicationTrackingFlow()
        start_time = datetime.now(timezone.utc)
        
        # Initial stats
        assert flow.stats['total_runs'] == 0
        assert flow.stats['failed_runs'] == 0
        assert len(flow.stats['errors']) == 0
        
        # Update stats
        flow._update_error_stats(start_time, "Test error")
        
        assert flow.stats['total_runs'] == 1
        assert flow.stats['failed_runs'] == 1
        assert len(flow.stats['errors']) == 1
        assert flow.stats['errors'][0]['error'] == "Test error"
        assert flow.stats['last_run'] is not None
    
    def test_update_error_stats_limit(self):
        """Test error statistics limit."""
        flow = JobApplicationTrackingFlow()
        start_time = datetime.now(timezone.utc)
        
        # Add 15 errors (more than the limit of 10)
        for i in range(15):
            flow._update_error_stats(start_time, f"Error {i}")
        
        # Should only keep the last 10 errors
        assert len(flow.stats['errors']) == 10
        assert flow.stats['errors'][0]['error'] == "Error 5"  # First kept error
        assert flow.stats['errors'][-1]['error'] == "Error 14"  # Last error
    
    def test_get_statistics(self):
        """Test getting flow statistics."""
        flow = JobApplicationTrackingFlow()
        
        # Update some stats
        flow.stats['total_runs'] = 5
        flow.stats['successful_runs'] = 3
        flow.stats['failed_runs'] = 2
        flow.stats['average_runtime'] = 120.5
        flow.stats['last_run'] = datetime.now(timezone.utc)
        flow.stats['errors'] = [
            {'error': 'Error 1', 'timestamp': datetime.now(timezone.utc)},
            {'error': 'Error 2', 'timestamp': datetime.now(timezone.utc)}
        ]
        
        stats = flow.get_statistics()
        
        assert stats['total_runs'] == 5
        assert stats['successful_runs'] == 3
        assert stats['failed_runs'] == 2
        assert stats['success_rate'] == 0.6  # 3/5
        assert stats['average_runtime'] == 120.5
        assert stats['last_run'] is not None
        assert len(stats['recent_errors']) == 2
    
    def test_get_statistics_no_runs(self):
        """Test getting statistics with no runs."""
        flow = JobApplicationTrackingFlow()
        
        stats = flow.get_statistics()
        
        assert stats['total_runs'] == 0
        assert stats['successful_runs'] == 0
        assert stats['failed_runs'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_runtime'] == 0.0
        assert stats['last_run'] is None
        assert stats['recent_errors'] == []
    
    def test_reset_statistics(self):
        """Test resetting flow statistics."""
        flow = JobApplicationTrackingFlow()
        
        # Add some stats
        flow.stats['total_runs'] = 10
        flow.stats['successful_runs'] = 8
        flow.stats['failed_runs'] = 2
        flow.stats['average_runtime'] = 150.0
        flow.stats['last_run'] = datetime.now(timezone.utc)
        flow.stats['errors'] = [{'error': 'Test error'}]
        
        # Reset stats
        flow.reset_statistics()
        
        assert flow.stats['total_runs'] == 0
        assert flow.stats['successful_runs'] == 0
        assert flow.stats['failed_runs'] == 0
        assert flow.stats['average_runtime'] == 0.0
        assert flow.stats['last_run'] is None
        assert flow.stats['errors'] == []


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_job_application_flow(self):
        """Test create_job_application_flow factory."""
        flow = create_job_application_flow()
        
        assert isinstance(flow, JobApplicationTrackingFlow)
        assert len(flow.nodes) == 12  # Expected number of nodes
        assert flow.stats['total_runs'] == 0
    
    @pytest.mark.asyncio
    async def test_run_job_application_tracking(self):
        """Test run_job_application_tracking convenience function."""
        with patch('flow.create_job_application_flow') as mock_create_flow:
            mock_flow = MagicMock()
            mock_flow.run_full_workflow = AsyncMock(return_value={'success': True})
            mock_create_flow.return_value = mock_flow
            
            result = await run_job_application_tracking(days_back=7)
            
            mock_create_flow.assert_called_once()
            mock_flow.run_full_workflow.assert_called_once_with(7)
            assert result['success'] is True


class TestFlowIntegration:
    """Test flow integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_mock(self):
        """Test end-to-end workflow with mocked components."""
        flow = JobApplicationTrackingFlow()
        
        # Mock all node operations
        mock_nodes = {}
        for node_name in flow.nodes:
            mock_nodes[node_name] = {
                'prep_async': AsyncMock(return_value={'ready': True}),
                'exec_async': AsyncMock(return_value={'success': True}),
                'post_async': AsyncMock(return_value='success')
            }
            
            # Patch node methods
            flow.nodes[node_name].prep_async = mock_nodes[node_name]['prep_async']
            flow.nodes[node_name].exec_async = mock_nodes[node_name]['exec_async']
            flow.nodes[node_name].post_async = mock_nodes[node_name]['post_async']
        
        # Mock settings
        with patch.object(flow.settings, 'brave_api_key', 'test_key'), \
             patch.object(flow.settings, 'openai_api_key', 'test_key'):
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Verify result
            assert result['success'] is True
            
            # Verify all nodes were called
            for node_name in ['email_retrieval', 'email_filtering', 'email_preprocessing',
                             'classification', 'job_application_creation',
                             'research', 'company_insights', 'research_aggregation',
                             'status', 'insights_generation', 'final_report']:
                assert mock_nodes[node_name]['prep_async'].called
                assert mock_nodes[node_name]['exec_async'].called
                assert mock_nodes[node_name]['post_async'].called
    
    @pytest.mark.asyncio
    async def test_partial_workflow_recovery(self):
        """Test workflow recovery from partial failures."""
        flow = JobApplicationTrackingFlow()
        
        # Mock email and classification to succeed, research to fail partially
        with patch.object(flow, '_run_email_processing_phase', new_callable=AsyncMock) as mock_email, \
             patch.object(flow, '_run_classification_phase', new_callable=AsyncMock) as mock_classification, \
             patch.object(flow, '_run_research_phase', new_callable=AsyncMock) as mock_research, \
             patch.object(flow, '_run_reporting_phase', new_callable=AsyncMock) as mock_reporting, \
             patch.object(flow, '_compile_final_results', new_callable=AsyncMock) as mock_compile:
            
            mock_email.return_value = {'success': True}
            mock_classification.return_value = {'success': True}
            mock_research.return_value = {'success': True, 'warning': 'Partial failure'}
            mock_reporting.return_value = {'success': True}
            mock_compile.return_value = {'success': True, 'workflow_id': 'test_id'}
            
            # Run workflow
            result = await flow.run_full_workflow(days_back=7)
            
            # Should still succeed with warnings
            assert result['success'] is True
            assert result['workflow_id'] == 'test_id'
    
    def test_node_names_consistency(self):
        """Test that node names are consistent across the workflow."""
        flow = JobApplicationTrackingFlow()
        
        # Expected node names based on the workflow
        expected_nodes = [
            'email_retrieval', 'email_filtering', 'email_preprocessing',
            'classification', 'job_application_creation', 'classification_validation',
            'research', 'company_insights', 'research_aggregation',
            'status', 'insights_generation', 'final_report'
        ]
        
        # Check that all expected nodes exist
        for node_name in expected_nodes:
            assert node_name in flow.nodes, f"Missing node: {node_name}"
        
        # Check that no unexpected nodes exist
        for node_name in flow.nodes:
            assert node_name in expected_nodes, f"Unexpected node: {node_name}"


if __name__ == "__main__":
    pytest.main([__file__])