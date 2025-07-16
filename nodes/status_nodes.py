"""
PocketFlow nodes for status reporting in job application tracking system.

Contains PocketFlow node implementations that wrap the Status Agent
for use in the job application tracking workflow.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from pocketflow import Node, AsyncNode
from agents.status_agent import StatusAgent, create_status_agent
from models.job_models import JobApplication, ApplicationSummary
from models.research_models import ResearchResult
from config.settings import get_settings
from utils import logger

class StatusNode(AsyncNode):
    """
    PocketFlow node for generating status reports.
    
    Uses the Status Agent to generate comprehensive status reports
    and analytics for job application tracking.
    """
    
    def __init__(self, node_id: str = "status"):
        """
        Initialize StatusNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.status_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for status report generation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get job applications from shared state
            job_applications = shared.get('job_applications', [])
            
            if not job_applications:
                return {
                    'error': 'No job applications available for status reporting',
                    'ready': False
                }
            
            # Get optional research results
            research_results = shared.get('research_results', [])
            
            # Initialize status agent if not already done
            if not self.status_agent:
                self.status_agent = create_status_agent()
            
            logger.info(f"StatusNode: Preparing to generate status report for {len(job_applications)} job applications")
            
            return {
                'job_applications': job_applications,
                'research_results': research_results,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"StatusNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute status report generation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for status reporting'),
                'status_report': {}
            }
        
        try:
            job_applications = inputs['job_applications']
            research_results = inputs['research_results']
            
            logger.info(f"StatusNode: Generating status report for {len(job_applications)} applications")
            
            # Generate detailed status report
            status_report = self.status_agent.generate_detailed_report(
                job_applications, research_results
            )
            
            logger.info(f"StatusNode: Status report generated successfully")
            
            return {
                'success': True,
                'status_report': status_report,
                'total_applications': len(job_applications),
                'report_sections': len(status_report)
            }
            
        except Exception as e:
            logger.error(f"StatusNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_report': {}
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process status report results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store status report in shared state
                shared['status_report'] = exec_res['status_report']
                shared['total_applications_reported'] = exec_res['total_applications']
                
                # Store processing metadata
                shared['status_reporting_completed'] = True
                shared['status_reporting_timestamp'] = datetime.now(timezone.utc)
                
                # Get agent statistics
                if self.status_agent:
                    shared['status_agent_stats'] = self.status_agent.get_statistics()
                
                logger.info(f"StatusNode: Completed successfully, generated report for {exec_res['total_applications']} applications")
                
                return "success"
            else:
                # Store error information
                shared['status_reporting_error'] = exec_res.get('error', 'Unknown error')
                shared['status_reporting_completed'] = False
                
                logger.error(f"StatusNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"StatusNode post-processing failed: {e}")
            shared['status_reporting_error'] = str(e)
            shared['status_reporting_completed'] = False
            return "error"

class InsightsGenerationNode(AsyncNode):
    """
    PocketFlow node for generating AI-powered insights.
    
    Uses the Status Agent to generate AI-powered insights and
    recommendations for job search improvement.
    """
    
    def __init__(self, node_id: str = "insights_generation"):
        """
        Initialize InsightsGenerationNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.status_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for insights generation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get job applications from shared state
            job_applications = shared.get('job_applications', [])
            
            if not job_applications:
                return {
                    'error': 'No job applications available for insights generation',
                    'ready': False
                }
            
            # Initialize status agent if not already done
            if not self.status_agent:
                self.status_agent = create_status_agent()
            
            logger.info(f"InsightsGenerationNode: Preparing to generate insights for {len(job_applications)} job applications")
            
            return {
                'job_applications': job_applications,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"InsightsGenerationNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute insights generation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for insights generation'),
                'insights': {}
            }
        
        try:
            job_applications = inputs['job_applications']
            
            logger.info(f"InsightsGenerationNode: Generating AI insights for {len(job_applications)} applications")
            
            # Generate AI-powered insights
            insights = await self.status_agent.generate_insights(job_applications)
            
            logger.info(f"InsightsGenerationNode: AI insights generated successfully")
            
            return {
                'success': True,
                'insights': insights,
                'applications_analyzed': len(job_applications)
            }
            
        except Exception as e:
            logger.error(f"InsightsGenerationNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'insights': {}
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process insights generation results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store insights in shared state
                shared['ai_insights'] = exec_res['insights']
                shared['applications_analyzed'] = exec_res['applications_analyzed']
                
                # Store processing metadata
                shared['insights_generation_completed'] = True
                shared['insights_generation_timestamp'] = datetime.now(timezone.utc)
                
                # Get agent statistics
                if self.status_agent:
                    shared['status_agent_stats'] = self.status_agent.get_statistics()
                
                logger.info(f"InsightsGenerationNode: Completed successfully, generated insights for {exec_res['applications_analyzed']} applications")
                
                return "success"
            else:
                # Store error information
                shared['insights_generation_error'] = exec_res.get('error', 'Unknown error')
                shared['insights_generation_completed'] = False
                
                logger.error(f"InsightsGenerationNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"InsightsGenerationNode post-processing failed: {e}")
            shared['insights_generation_error'] = str(e)
            shared['insights_generation_completed'] = False
            return "error"

class FinalReportNode(AsyncNode):
    """
    PocketFlow node for generating final consolidated report.
    
    Consolidates all processing results into a final comprehensive
    report for the job application tracking workflow.
    """
    
    def __init__(self, node_id: str = "final_report"):
        """
        Initialize FinalReportNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for final report generation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Gather all processing results from shared state
            job_applications = shared.get('job_applications', [])
            status_report = shared.get('status_report', {})
            ai_insights = shared.get('ai_insights', {})
            research_results = shared.get('research_results', [])
            aggregated_research_insights = shared.get('aggregated_research_insights', {})
            
            # Processing metadata
            processing_metadata = {
                'email_retrieval_completed': shared.get('email_retrieval_completed', False),
                'classification_completed': shared.get('classification_completed', False),
                'research_completed': shared.get('research_completed', False),
                'status_reporting_completed': shared.get('status_reporting_completed', False),
                'insights_generation_completed': shared.get('insights_generation_completed', False),
                'total_emails_retrieved': shared.get('total_emails_retrieved', 0),
                'total_job_emails': shared.get('total_job_emails', 0),
                'total_classified': shared.get('total_classified', 0),
                'total_researched': shared.get('total_researched', 0),
                'total_postings_found': shared.get('total_postings_found', 0)
            }
            
            # Agent statistics
            agent_stats = {
                'email_agent_stats': shared.get('email_agent_stats', {}),
                'classification_agent_stats': shared.get('classification_agent_stats', {}),
                'research_agent_stats': shared.get('research_agent_stats', {}),
                'status_agent_stats': shared.get('status_agent_stats', {})
            }
            
            logger.info(f"FinalReportNode: Preparing to generate final report")
            
            return {
                'job_applications': job_applications,
                'status_report': status_report,
                'ai_insights': ai_insights,
                'research_results': research_results,
                'aggregated_research_insights': aggregated_research_insights,
                'processing_metadata': processing_metadata,
                'agent_stats': agent_stats,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"FinalReportNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final report generation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for final report generation'),
                'final_report': {}
            }
        
        try:
            logger.info(f"FinalReportNode: Generating final consolidated report")
            
            # Create final report
            final_report = await self._create_final_report(inputs)
            
            logger.info(f"FinalReportNode: Final report generated successfully")
            
            return {
                'success': True,
                'final_report': final_report,
                'report_sections': len(final_report)
            }
            
        except Exception as e:
            logger.error(f"FinalReportNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'final_report': {}
            }
    
    async def _create_final_report(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the final consolidated report.
        
        Args:
            inputs (Dict[str, Any]): Input data
            
        Returns:
            Dict[str, Any]: Final report
        """
        # Extract inputs
        job_applications = inputs['job_applications']
        status_report = inputs['status_report']
        ai_insights = inputs['ai_insights']
        research_results = inputs['research_results']
        aggregated_research_insights = inputs['aggregated_research_insights']
        processing_metadata = inputs['processing_metadata']
        agent_stats = inputs['agent_stats']
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            job_applications, status_report, ai_insights, processing_metadata
        )
        
        # Create processing summary
        processing_summary = self._create_processing_summary(processing_metadata, agent_stats)
        
        # Create key findings
        key_findings = self._create_key_findings(
            job_applications, status_report, aggregated_research_insights
        )
        
        # Create recommendations
        recommendations = self._create_recommendations(ai_insights, status_report)
        
        # Create next steps
        next_steps = self._create_next_steps(ai_insights, status_report)
        
        # Create final report
        final_report = {
            'executive_summary': executive_summary,
            'processing_summary': processing_summary,
            'key_findings': key_findings,
            'recommendations': recommendations,
            'next_steps': next_steps,
            'detailed_data': {
                'job_applications': [app.to_dict() for app in job_applications],
                'status_report': status_report,
                'ai_insights': ai_insights,
                'research_insights': aggregated_research_insights,
                'research_results_count': len(research_results)
            },
            'agent_performance': agent_stats,
            'metadata': {
                'generated_at': datetime.now(timezone.utc),
                'total_applications': len(job_applications),
                'processing_completed': all([
                    processing_metadata.get('email_retrieval_completed', False),
                    processing_metadata.get('classification_completed', False),
                    processing_metadata.get('status_reporting_completed', False)
                ]),
                'workflow_version': '1.0.0'
            }
        }
        
        return final_report
    
    def _create_executive_summary(self, job_applications: List[JobApplication], 
                                status_report: Dict[str, Any], 
                                ai_insights: Dict[str, Any], 
                                processing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary."""
        if not job_applications:
            return {'message': 'No job applications to summarize'}
        
        summary = status_report.get('summary', {})
        
        return {
            'total_applications': len(job_applications),
            'response_rate': getattr(summary, 'response_rate', 0.0),
            'interview_rate': getattr(summary, 'interview_rate', 0.0),
            'offer_rate': getattr(summary, 'offer_rate', 0.0),
            'emails_processed': processing_metadata.get('total_emails_retrieved', 0),
            'job_emails_identified': processing_metadata.get('total_job_emails', 0),
            'research_completed': processing_metadata.get('research_completed', False),
            'overall_assessment': ai_insights.get('overall_assessment', 'Assessment not available'),
            'key_insight': ai_insights.get('strengths', ['No insights available'])[0] if ai_insights.get('strengths') else 'No insights available'
        }
    
    def _create_processing_summary(self, processing_metadata: Dict[str, Any], 
                                 agent_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Create processing summary."""
        return {
            'workflow_completion': {
                'email_retrieval': processing_metadata.get('email_retrieval_completed', False),
                'classification': processing_metadata.get('classification_completed', False),
                'research': processing_metadata.get('research_completed', False),
                'status_reporting': processing_metadata.get('status_reporting_completed', False),
                'insights_generation': processing_metadata.get('insights_generation_completed', False)
            },
            'data_processing': {
                'total_emails_retrieved': processing_metadata.get('total_emails_retrieved', 0),
                'job_related_emails': processing_metadata.get('total_job_emails', 0),
                'applications_classified': processing_metadata.get('total_classified', 0),
                'applications_researched': processing_metadata.get('total_researched', 0),
                'job_postings_found': processing_metadata.get('total_postings_found', 0)
            },
            'agent_performance': {
                'email_agent': {
                    'success_rate': agent_stats.get('email_agent_stats', {}).get('job_relevance_rate', 0),
                    'processing_time': agent_stats.get('email_agent_stats', {}).get('processing_time', 0)
                },
                'classification_agent': {
                    'success_rate': agent_stats.get('classification_agent_stats', {}).get('success_rate', 0),
                    'job_identification_rate': agent_stats.get('classification_agent_stats', {}).get('job_identification_rate', 0)
                },
                'research_agent': {
                    'success_rate': agent_stats.get('research_agent_stats', {}).get('success_rate', 0),
                    'average_postings_per_research': agent_stats.get('research_agent_stats', {}).get('average_postings_per_research', 0)
                }
            }
        }
    
    def _create_key_findings(self, job_applications: List[JobApplication], 
                           status_report: Dict[str, Any], 
                           aggregated_research_insights: Dict[str, Any]) -> List[str]:
        """Create key findings list."""
        findings = []
        
        if job_applications:
            findings.append(f"Analyzed {len(job_applications)} job applications")
        
        # Status findings
        if status_report.get('summary'):
            summary = status_report['summary']
            response_rate = getattr(summary, 'response_rate', 0.0)
            if response_rate > 0:
                findings.append(f"Response rate: {response_rate:.1%}")
        
        # Research findings
        if aggregated_research_insights:
            market_activity = aggregated_research_insights.get('market_activity', {})
            if market_activity.get('market_temperature'):
                findings.append(f"Market temperature: {market_activity['market_temperature']}")
            
            if aggregated_research_insights.get('total_job_postings', 0) > 0:
                findings.append(f"Found {aggregated_research_insights['total_job_postings']} related job postings")
        
        if not findings:
            findings.append("No significant findings to report")
        
        return findings
    
    def _create_recommendations(self, ai_insights: Dict[str, Any], 
                              status_report: Dict[str, Any]) -> List[str]:
        """Create recommendations list."""
        recommendations = []
        
        # AI-generated recommendations
        if ai_insights.get('specific_recommendations'):
            recommendations.extend(ai_insights['specific_recommendations'])
        
        # Status-based recommendations
        if status_report.get('stale_analysis', {}).get('follow_up_recommendations'):
            recommendations.extend(status_report['stale_analysis']['follow_up_recommendations'])
        
        if not recommendations:
            recommendations.append("Continue monitoring job application progress")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _create_next_steps(self, ai_insights: Dict[str, Any], 
                         status_report: Dict[str, Any]) -> List[str]:
        """Create next steps list."""
        next_steps = []
        
        # AI-generated next steps
        if ai_insights.get('next_steps'):
            next_steps.extend(ai_insights['next_steps'])
        
        # Stale application follow-ups
        stale_analysis = status_report.get('stale_analysis', {})
        if stale_analysis.get('stale_counts', {}).get('follow_up_needed', 0) > 0:
            next_steps.append("Follow up on applications needing attention")
        
        if not next_steps:
            next_steps.append("Continue regular job application tracking")
        
        return next_steps[:3]  # Top 3 next steps
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process final report results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store final report in shared state
                shared['final_report'] = exec_res['final_report']
                
                # Store processing metadata
                shared['final_report_completed'] = True
                shared['final_report_timestamp'] = datetime.now(timezone.utc)
                
                # Mark workflow as completed
                shared['workflow_completed'] = True
                shared['workflow_completion_timestamp'] = datetime.now(timezone.utc)
                
                logger.info(f"FinalReportNode: Completed successfully, generated final report")
                
                return "success"
            else:
                # Store error information
                shared['final_report_error'] = exec_res.get('error', 'Unknown error')
                shared['final_report_completed'] = False
                
                logger.error(f"FinalReportNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"FinalReportNode post-processing failed: {e}")
            shared['final_report_error'] = str(e)
            shared['final_report_completed'] = False
            return "error"

# Factory functions
def create_status_node() -> StatusNode:
    """
    Create and return StatusNode instance.
    
    Returns:
        StatusNode: Configured status node
    """
    return StatusNode()

def create_insights_generation_node() -> InsightsGenerationNode:
    """
    Create and return InsightsGenerationNode instance.
    
    Returns:
        InsightsGenerationNode: Configured insights generation node
    """
    return InsightsGenerationNode()

def create_final_report_node() -> FinalReportNode:
    """
    Create and return FinalReportNode instance.
    
    Returns:
        FinalReportNode: Configured final report node
    """
    return FinalReportNode()

# Test function
async def test_status_nodes():
    """Test status nodes functionality."""
    try:
        # Test status node
        status_node = create_status_node()
        
        # Test insights generation node
        insights_node = create_insights_generation_node()
        
        # Test final report node
        final_report_node = create_final_report_node()
        
        print("✅ Status nodes created successfully")
        print(f"📊 Status node ID: {status_node.node_id}")
        print(f"💡 Insights generation node ID: {insights_node.node_id}")
        print(f"📋 Final report node ID: {final_report_node.node_id}")
        
    except Exception as e:
        print(f"❌ Status nodes test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_status_nodes())