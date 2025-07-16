"""
PocketFlow nodes for research operations in job application tracking system.

Contains PocketFlow node implementations that wrap the Research Agent
for use in the job application tracking workflow.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from pocketflow import Node, AsyncNode
from agents.research_agent import ResearchAgent, create_research_agent
from models.job_models import JobApplication
from models.research_models import ResearchResult
from config.settings import get_settings
from utils import logger

class ResearchNode(AsyncNode):
    """
    PocketFlow node for researching job applications.
    
    Uses the Research Agent to find related job postings and
    enrich job application data with market insights.
    """
    
    def __init__(self, node_id: str = "research"):
        """
        Initialize ResearchNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.settings = get_settings()
        self.research_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for job application research.
        
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
                    'error': 'No job applications available for research',
                    'ready': False
                }
            
            # Initialize research agent if not already done
            if not self.research_agent:
                self.research_agent = create_research_agent()
            
            logger.info(f"ResearchNode: Preparing to research {len(job_applications)} job applications")
            
            return {
                'job_applications': job_applications,
                'max_results': self.settings.research_max_results,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"ResearchNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute job application research.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for research'),
                'research_results': []
            }
        
        try:
            job_applications = inputs['job_applications']
            
            logger.info(f"ResearchNode: Researching {len(job_applications)} job applications")
            
            # Research job applications in batch
            research_results = await self.research_agent.research_batch(job_applications)
            
            logger.info(f"ResearchNode: Completed research for {len(research_results)} applications")
            
            return {
                'success': True,
                'research_results': research_results,
                'total_researched': len(research_results),
                'total_postings_found': sum(result.total_results for result in research_results)
            }
            
        except Exception as e:
            logger.error(f"ResearchNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'research_results': []
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process research results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store research results in shared state
                shared['research_results'] = exec_res['research_results']
                shared['total_researched'] = exec_res['total_researched']
                shared['total_postings_found'] = exec_res['total_postings_found']
                
                # Store processing metadata
                shared['research_completed'] = True
                shared['research_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.research_agent:
                    shared['research_agent_stats'] = self.research_agent.get_statistics()
                
                logger.info(f"ResearchNode: Completed successfully, researched {exec_res['total_researched']} applications")
                
                return "success"
            else:
                # Store error information
                shared['research_error'] = exec_res.get('error', 'Unknown error')
                shared['research_completed'] = False
                
                logger.error(f"ResearchNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"ResearchNode post-processing failed: {e}")
            shared['research_error'] = str(e)
            shared['research_completed'] = False
            return "error"

class CompanyInsightsNode(AsyncNode):
    """
    PocketFlow node for gathering company insights.
    
    Uses the Research Agent to gather insights about companies
    that users have applied to.
    """
    
    def __init__(self, node_id: str = "company_insights"):
        """
        Initialize CompanyInsightsNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
        self.research_agent = None
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for company insights gathering.
        
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
                    'error': 'No job applications available for company insights',
                    'ready': False
                }
            
            # Extract unique companies
            companies = list(set(app.company for app in job_applications))
            
            # Initialize research agent if not already done
            if not self.research_agent:
                self.research_agent = create_research_agent()
            
            logger.info(f"CompanyInsightsNode: Preparing to gather insights for {len(companies)} companies")
            
            return {
                'companies': companies,
                'job_applications': job_applications,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"CompanyInsightsNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute company insights gathering.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for company insights'),
                'company_insights': {}
            }
        
        try:
            companies = inputs['companies']
            
            logger.info(f"CompanyInsightsNode: Gathering insights for {len(companies)} companies")
            
            # Gather insights for each company
            company_insights = {}
            
            for company in companies:
                try:
                    insights = await self.research_agent.get_company_insights(company)
                    company_insights[company] = insights
                    
                    # Rate limiting between companies
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.warning(f"Failed to get insights for {company}: {e}")
                    company_insights[company] = {
                        'company': company,
                        'error': str(e)
                    }
            
            logger.info(f"CompanyInsightsNode: Gathered insights for {len(company_insights)} companies")
            
            return {
                'success': True,
                'company_insights': company_insights,
                'total_companies': len(company_insights)
            }
            
        except Exception as e:
            logger.error(f"CompanyInsightsNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'company_insights': {}
            }
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process company insights results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store company insights in shared state
                shared['company_insights'] = exec_res['company_insights']
                shared['total_companies_researched'] = exec_res['total_companies']
                
                # Store processing metadata
                shared['company_insights_completed'] = True
                shared['company_insights_timestamp'] = datetime.now()
                
                # Get agent statistics
                if self.research_agent:
                    shared['research_agent_stats'] = self.research_agent.get_statistics()
                
                logger.info(f"CompanyInsightsNode: Completed successfully, gathered insights for {exec_res['total_companies']} companies")
                
                return "success"
            else:
                # Store error information
                shared['company_insights_error'] = exec_res.get('error', 'Unknown error')
                shared['company_insights_completed'] = False
                
                logger.error(f"CompanyInsightsNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"CompanyInsightsNode post-processing failed: {e}")
            shared['company_insights_error'] = str(e)
            shared['company_insights_completed'] = False
            return "error"

class ResearchAggregationNode(AsyncNode):
    """
    PocketFlow node for aggregating research results.
    
    Aggregates research results from multiple sources and generates
    consolidated insights for the job search.
    """
    
    def __init__(self, node_id: str = "research_aggregation"):
        """
        Initialize ResearchAggregationNode.
        
        Args:
            node_id (str): Unique identifier for this node
        """
        super().__init__()
        self.node_id = node_id
    
    async def prep_async(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare for research aggregation.
        
        Args:
            shared (Dict[str, Any]): Shared state
            
        Returns:
            Dict[str, Any]: Prepared inputs
        """
        try:
            # Get research results from shared state
            research_results = shared.get('research_results', [])
            company_insights = shared.get('company_insights', {})
            job_applications = shared.get('job_applications', [])
            
            if not research_results and not company_insights:
                return {
                    'error': 'No research data available for aggregation',
                    'ready': False
                }
            
            logger.info(f"ResearchAggregationNode: Preparing to aggregate {len(research_results)} research results and {len(company_insights)} company insights")
            
            return {
                'research_results': research_results,
                'company_insights': company_insights,
                'job_applications': job_applications,
                'ready': True
            }
            
        except Exception as e:
            logger.error(f"ResearchAggregationNode prep failed: {e}")
            return {
                'error': str(e),
                'ready': False
            }
    
    async def exec_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research aggregation.
        
        Args:
            inputs (Dict[str, Any]): Prepared inputs
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if not inputs.get('ready', False):
            return {
                'success': False,
                'error': inputs.get('error', 'Not ready for research aggregation'),
                'aggregated_insights': {}
            }
        
        try:
            research_results = inputs['research_results']
            company_insights = inputs['company_insights']
            job_applications = inputs['job_applications']
            
            logger.info(f"ResearchAggregationNode: Aggregating research data")
            
            # Aggregate research results
            aggregated_insights = await self._aggregate_research_data(
                research_results, company_insights, job_applications
            )
            
            logger.info(f"ResearchAggregationNode: Aggregation completed")
            
            return {
                'success': True,
                'aggregated_insights': aggregated_insights,
                'total_job_postings': aggregated_insights.get('total_job_postings', 0),
                'average_relevance': aggregated_insights.get('average_relevance', 0.0)
            }
            
        except Exception as e:
            logger.error(f"ResearchAggregationNode execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'aggregated_insights': {}
            }
    
    async def _aggregate_research_data(self, research_results: List[ResearchResult], 
                                     company_insights: Dict[str, Any], 
                                     job_applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Aggregate research data into consolidated insights.
        
        Args:
            research_results (List[ResearchResult]): Research results
            company_insights (Dict[str, Any]): Company insights
            job_applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Aggregated insights
        """
        # Research results aggregation
        total_postings = sum(result.total_results for result in research_results)
        total_researched = len(research_results)
        average_relevance = sum(result.average_relevance for result in research_results) / len(research_results) if research_results else 0.0
        
        # Best matches
        best_matches = []
        for result in research_results:
            if result.best_match:
                best_matches.append({
                    'company': result.target_company,
                    'position': result.target_position,
                    'match_title': result.best_match.title,
                    'match_company': result.best_match.company,
                    'relevance_score': result.best_match.relevance_score,
                    'url': result.best_match.url
                })
        
        # Company insights aggregation
        company_summary = {}
        for company, insights in company_insights.items():
            if not insights.get('error'):
                company_summary[company] = {
                    'total_postings': insights.get('total_postings', 0),
                    'active_recruitment': insights.get('active_recruitment', 0),
                    'common_positions': insights.get('common_positions', []),
                    'hiring_locations': insights.get('hiring_locations', []),
                    'activity_level': insights.get('recent_activity', {}).get('activity_level', 'unknown')
                }
        
        # Market analysis
        market_activity = self._calculate_market_activity(research_results, company_insights)
        
        # Application-to-posting ratio
        application_companies = set(app.company for app in job_applications)
        researched_companies = set(company_insights.keys())
        coverage_ratio = len(researched_companies & application_companies) / len(application_companies) if application_companies else 0.0
        
        return {
            'total_job_postings': total_postings,
            'total_researched': total_researched,
            'average_relevance': average_relevance,
            'best_matches': best_matches[:10],  # Top 10 matches
            'company_summary': company_summary,
            'market_activity': market_activity,
            'coverage_ratio': coverage_ratio,
            'aggregation_timestamp': datetime.now(),
            'insights': self._generate_market_insights(research_results, company_insights)
        }
    
    def _calculate_market_activity(self, research_results: List[ResearchResult], 
                                 company_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market activity metrics."""
        # Activity from research results
        recent_postings = 0
        for result in research_results:
            for posting in result.job_postings:
                if posting.is_recent(30):
                    recent_postings += 1
        
        # Activity from company insights
        active_companies = 0
        for company, insights in company_insights.items():
            if not insights.get('error'):
                activity_level = insights.get('recent_activity', {}).get('activity_level', 'low')
                if activity_level in ['medium', 'high']:
                    active_companies += 1
        
        total_companies = len([c for c in company_insights.values() if not c.get('error')])
        
        return {
            'recent_postings': recent_postings,
            'active_companies': active_companies,
            'total_companies': total_companies,
            'activity_rate': active_companies / total_companies if total_companies > 0 else 0.0,
            'market_temperature': self._assess_market_temperature(active_companies, total_companies)
        }
    
    def _assess_market_temperature(self, active_companies: int, total_companies: int) -> str:
        """Assess market temperature based on activity."""
        if total_companies == 0:
            return 'unknown'
        
        activity_rate = active_companies / total_companies
        
        if activity_rate > 0.7:
            return 'hot'
        elif activity_rate > 0.4:
            return 'warm'
        elif activity_rate > 0.2:
            return 'cool'
        else:
            return 'cold'
    
    def _generate_market_insights(self, research_results: List[ResearchResult], 
                                company_insights: Dict[str, Any]) -> List[str]:
        """Generate market insights from research data."""
        insights = []
        
        # Research coverage
        if research_results:
            avg_postings = sum(r.total_results for r in research_results) / len(research_results)
            if avg_postings > 10:
                insights.append(f"High market activity detected with average {avg_postings:.1f} postings per application")
            elif avg_postings > 5:
                insights.append(f"Moderate market activity with average {avg_postings:.1f} postings per application")
            else:
                insights.append(f"Limited market activity with average {avg_postings:.1f} postings per application")
        
        # Company activity
        active_companies = sum(1 for insights in company_insights.values() 
                             if not insights.get('error') and 
                             insights.get('recent_activity', {}).get('activity_level') in ['medium', 'high'])
        
        if active_companies > 0:
            insights.append(f"{active_companies} companies showing active hiring patterns")
        
        # Market temperature
        total_companies = len([c for c in company_insights.values() if not c.get('error')])
        if total_companies > 0:
            activity_rate = active_companies / total_companies
            if activity_rate > 0.5:
                insights.append("Market conditions appear favorable for job seekers")
            else:
                insights.append("Market conditions suggest a more competitive environment")
        
        return insights
    
    async def post_async(self, shared: Dict[str, Any], prep_res: Dict[str, Any], 
                        exec_res: Dict[str, Any]) -> str:
        """
        Post-process research aggregation results.
        
        Args:
            shared (Dict[str, Any]): Shared state
            prep_res (Dict[str, Any]): Preparation results
            exec_res (Dict[str, Any]): Execution results
            
        Returns:
            str: Next flow action
        """
        try:
            if exec_res.get('success', False):
                # Store aggregated insights in shared state
                shared['aggregated_research_insights'] = exec_res['aggregated_insights']
                shared['total_job_postings_found'] = exec_res['total_job_postings']
                shared['average_research_relevance'] = exec_res['average_relevance']
                
                # Store processing metadata
                shared['research_aggregation_completed'] = True
                shared['research_aggregation_timestamp'] = datetime.now()
                
                logger.info(f"ResearchAggregationNode: Completed successfully")
                
                return "success"
            else:
                # Store error information
                shared['research_aggregation_error'] = exec_res.get('error', 'Unknown error')
                shared['research_aggregation_completed'] = False
                
                logger.error(f"ResearchAggregationNode: Failed with error: {exec_res.get('error')}")
                
                return "error"
                
        except Exception as e:
            logger.error(f"ResearchAggregationNode post-processing failed: {e}")
            shared['research_aggregation_error'] = str(e)
            shared['research_aggregation_completed'] = False
            return "error"

# Factory functions
def create_research_node() -> ResearchNode:
    """
    Create and return ResearchNode instance.
    
    Returns:
        ResearchNode: Configured research node
    """
    return ResearchNode()

def create_company_insights_node() -> CompanyInsightsNode:
    """
    Create and return CompanyInsightsNode instance.
    
    Returns:
        CompanyInsightsNode: Configured company insights node
    """
    return CompanyInsightsNode()

def create_research_aggregation_node() -> ResearchAggregationNode:
    """
    Create and return ResearchAggregationNode instance.
    
    Returns:
        ResearchAggregationNode: Configured research aggregation node
    """
    return ResearchAggregationNode()

# Test function
async def test_research_nodes():
    """Test research nodes functionality."""
    try:
        # Test research node
        research_node = create_research_node()
        
        # Test company insights node
        company_insights_node = create_company_insights_node()
        
        # Test research aggregation node
        aggregation_node = create_research_aggregation_node()
        
        print("✅ Research nodes created successfully")
        print(f"🔬 Research node ID: {research_node.node_id}")
        print(f"🏢 Company insights node ID: {company_insights_node.node_id}")
        print(f"📊 Research aggregation node ID: {aggregation_node.node_id}")
        
    except Exception as e:
        print(f"❌ Research nodes test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_research_nodes())