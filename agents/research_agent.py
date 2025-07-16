"""
Research Agent for job application tracking system.

Handles job posting research using Brave Search API to find related
job postings and enrich job application data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from tools.brave_search_tool import BraveSearchTool, create_brave_search_tool
from models.job_models import JobApplication
from models.research_models import JobPosting, ResearchResult, create_research_result
from config.settings import get_settings
from utils import logger, call_llm, parse_yaml_response

class ResearchAgent:
    """
    Research Agent for job posting research.
    
    Uses Brave Search API to find related job postings and enrich
    job application data with market insights and additional information.
    """
    
    def __init__(self):
        """Initialize Research Agent with tools and settings."""
        self.settings = get_settings()
        self.brave_tool = create_brave_search_tool()
        
        # Processing statistics
        self.stats = {
            'total_applications_researched': 0,
            'successful_researches': 0,
            'failed_researches': 0,
            'total_postings_found': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
        
        # Research analysis prompt
        self.analysis_prompt_template = """
        Analyze these job postings found for a job application and provide insights:
        
        Job Application:
        Company: {company}
        Position: {position}
        Applied Date: {applied_date}
        Status: {status}
        
        Found Job Postings:
        {postings}
        
        Please analyze these findings and provide:
        1. Market insights about this role at this company
        2. Salary range estimates based on similar postings
        3. Common requirements and qualifications
        4. Competition level assessment
        5. Application strategy recommendations
        
        Output in YAML format:
        ```yaml
        market_insights: "Analysis of the job market for this role"
        salary_estimate: "Estimated salary range"
        common_requirements: ["requirement1", "requirement2"]
        competition_level: "low/medium/high"
        application_strategy: "Recommendations for the application"
        confidence: 0.85
        ```
        """
    
    async def research_job_application(self, job_app: JobApplication) -> ResearchResult:
        """
        Research job postings for a specific job application.
        
        Args:
            job_app (JobApplication): Job application to research
            
        Returns:
            ResearchResult: Research results with job postings and analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Researching job application: {job_app.company} - {job_app.position}")
            
            # Create research queries
            queries = self._generate_research_queries(job_app)
            
            # Search for job postings
            all_postings = []
            
            for query in queries:
                try:
                    # Search with current query
                    postings = await self.brave_tool.search_job_postings(
                        query, 
                        max_results=self.settings.research_max_results // len(queries)
                    )
                    
                    all_postings.extend(postings)
                    
                    # Rate limiting
                    await asyncio.sleep(self.settings.brave_rate_limit_delay)
                    
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")
                    continue
            
            # Remove duplicates and filter relevant postings
            unique_postings = self._deduplicate_postings(all_postings)
            relevant_postings = self._filter_relevant_postings(unique_postings, job_app)
            
            # Create research result
            research_result = create_research_result(
                query=" OR ".join(queries),
                target_company=job_app.company,
                target_position=job_app.position,
                source_used="Brave Search API"
            )
            
            research_result.job_postings = relevant_postings
            
            # Analyze results
            research_result.analyze_results()
            
            # Add market analysis
            if relevant_postings:
                market_analysis = await self._analyze_market_insights(job_app, relevant_postings)
                research_result.market_analysis = market_analysis
            
            # Update statistics
            self.stats['total_applications_researched'] += 1
            self.stats['total_postings_found'] += len(relevant_postings)
            self.stats['successful_researches'] += 1
            self.stats['processing_time'] += (datetime.now() - start_time).total_seconds()
            self.stats['last_run'] = datetime.now()
            
            logger.info(f"Research complete: {len(relevant_postings)} relevant postings found")
            
            return research_result
            
        except Exception as e:
            logger.error(f"Failed to research job application: {e}")
            self.stats['failed_researches'] += 1
            self.stats['errors'].append(f"Research failed for {job_app.company} - {job_app.position}: {e}")
            raise
    
    def _generate_research_queries(self, job_app: JobApplication) -> List[str]:
        """
        Generate research queries for job application.
        
        Args:
            job_app (JobApplication): Job application
            
        Returns:
            List[str]: List of search queries
        """
        queries = []
        
        # Base query
        base_query = f"{job_app.company} {job_app.position}"
        queries.append(base_query)
        
        # Add job-specific variations
        job_variations = [
            f"{job_app.company} {job_app.position} hiring",
            f"{job_app.company} {job_app.position} careers",
            f"{job_app.company} jobs {job_app.position}",
            f'"{job_app.company}" "{job_app.position}" job'
        ]
        
        queries.extend(job_variations)
        
        # Add location-specific queries if available
        if job_app.location:
            location_queries = [
                f"{job_app.company} {job_app.position} {job_app.location}",
                f"{job_app.position} {job_app.location} {job_app.company}"
            ]
            queries.extend(location_queries)
        
        # Add site-specific searches
        site_queries = [
            f"site:linkedin.com/jobs {job_app.company} {job_app.position}",
            f"site:glassdoor.com {job_app.company} {job_app.position}",
            f"site:indeed.com {job_app.company} {job_app.position}"
        ]
        
        queries.extend(site_queries)
        
        # Limit to top queries to avoid API rate limits
        return queries[:6]
    
    def _deduplicate_postings(self, postings: List[JobPosting]) -> List[JobPosting]:
        """
        Remove duplicate job postings.
        
        Args:
            postings (List[JobPosting]): List of job postings
            
        Returns:
            List[JobPosting]: Deduplicated postings
        """
        seen_urls = set()
        unique_postings = []
        
        for posting in postings:
            if posting.url not in seen_urls:
                unique_postings.append(posting)
                seen_urls.add(posting.url)
        
        return unique_postings
    
    def _filter_relevant_postings(self, postings: List[JobPosting], job_app: JobApplication) -> List[JobPosting]:
        """
        Filter job postings for relevance to application.
        
        Args:
            postings (List[JobPosting]): List of job postings
            job_app (JobApplication): Job application
            
        Returns:
            List[JobPosting]: Filtered relevant postings
        """
        relevant_postings = []
        
        for posting in postings:
            # Calculate relevance scores
            title_similarity = posting.calculate_title_similarity(job_app.position)
            company_similarity = posting.calculate_company_similarity(job_app.company)
            
            # Set relevance factors
            factors = {
                'title_similarity': title_similarity,
                'company_similarity': company_similarity,
                'recency': 1.0 if posting.is_recent(60) else 0.5,  # 60 days recency
                'source_credibility': self._get_source_credibility(posting.source)
            }
            
            # Calculate weighted relevance
            weights = {
                'title_similarity': 0.4,
                'company_similarity': 0.3,
                'recency': 0.2,
                'source_credibility': 0.1
            }
            
            relevance_score = sum(factors[key] * weights[key] for key in factors)
            posting.relevance_score = relevance_score
            posting.match_factors = factors
            
            # Filter by minimum relevance
            if relevance_score >= 0.3:  # Minimum relevance threshold
                relevant_postings.append(posting)
        
        # Sort by relevance score
        relevant_postings.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return relevant_postings
    
    def _get_source_credibility(self, source) -> float:
        """
        Get credibility score for job source.
        
        Args:
            source: Job source
            
        Returns:
            float: Credibility score (0.0-1.0)
        """
        from models.research_models import JobSource
        
        credibility_scores = {
            JobSource.LINKEDIN: 0.9,
            JobSource.GLASSDOOR: 0.8,
            JobSource.COMPANY_WEBSITE: 1.0,
            JobSource.INDEED: 0.7,
            JobSource.MONSTER: 0.6,
            JobSource.DICE: 0.7,
            JobSource.ZIPRECRUITER: 0.6,
            JobSource.CAREERBUILDER: 0.5,
            JobSource.UNKNOWN: 0.3
        }
        
        return credibility_scores.get(source, 0.5)
    
    async def _analyze_market_insights(self, job_app: JobApplication, postings: List[JobPosting]) -> Dict[str, Any]:
        """
        Analyze market insights from job postings.
        
        Args:
            job_app (JobApplication): Job application
            postings (List[JobPosting]): Related job postings
            
        Returns:
            Dict[str, Any]: Market analysis
        """
        try:
            # Prepare postings summary for analysis
            postings_summary = []
            for posting in postings[:5]:  # Limit to top 5 for analysis
                summary = {
                    'title': posting.title,
                    'company': posting.company,
                    'location': posting.location,
                    'salary_range': posting.salary_range,
                    'description': posting.description[:200] + "..." if len(posting.description) > 200 else posting.description,
                    'source': posting.source.value,
                    'relevance_score': posting.relevance_score
                }
                postings_summary.append(summary)
            
            # Create analysis prompt
            postings_text = "\n".join([
                f"- {p['title']} at {p['company']} ({p['location']}) - {p['salary_range'] or 'No salary info'} - Relevance: {p['relevance_score']:.2f}"
                for p in postings_summary
            ])
            
            prompt = self.analysis_prompt_template.format(
                company=job_app.company,
                position=job_app.position,
                applied_date=job_app.applied_date.strftime('%Y-%m-%d'),
                status=job_app.status.value,
                postings=postings_text
            )
            
            # Get analysis from LLM
            response = await call_llm(prompt)
            analysis = parse_yaml_response(response)
            
            # Add statistical analysis
            analysis['statistics'] = self._calculate_posting_statistics(postings)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze market insights: {e}")
            return {
                'market_insights': 'Analysis failed',
                'salary_estimate': 'Unable to estimate',
                'common_requirements': [],
                'competition_level': 'unknown',
                'application_strategy': 'Unable to provide recommendations',
                'confidence': 0.0
            }
    
    def _calculate_posting_statistics(self, postings: List[JobPosting]) -> Dict[str, Any]:
        """
        Calculate statistics from job postings.
        
        Args:
            postings (List[JobPosting]): Job postings
            
        Returns:
            Dict[str, Any]: Statistical analysis
        """
        if not postings:
            return {}
        
        # Source distribution
        source_counts = {}
        for posting in postings:
            source = posting.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Location distribution
        location_counts = {}
        for posting in postings:
            location = posting.location or 'Unknown'
            location_counts[location] = location_counts.get(location, 0) + 1
        
        # Recent postings
        recent_count = sum(1 for posting in postings if posting.is_recent(30))
        
        # Average relevance
        avg_relevance = sum(posting.relevance_score for posting in postings) / len(postings)
        
        # Salary information
        salary_info = []
        for posting in postings:
            if posting.salary_range:
                salary_info.append(posting.salary_range)
        
        return {
            'total_postings': len(postings),
            'source_distribution': source_counts,
            'location_distribution': location_counts,
            'recent_postings': recent_count,
            'recent_percentage': (recent_count / len(postings)) * 100,
            'average_relevance': avg_relevance,
            'salary_mentions': len(salary_info),
            'salary_examples': salary_info[:3]  # Top 3 salary examples
        }
    
    async def research_batch(self, job_apps: List[JobApplication]) -> List[ResearchResult]:
        """
        Research multiple job applications in batch.
        
        Args:
            job_apps (List[JobApplication]): Job applications to research
            
        Returns:
            List[ResearchResult]: Research results
        """
        try:
            logger.info(f"Starting batch research for {len(job_apps)} job applications")
            
            research_results = []
            
            for i, job_app in enumerate(job_apps):
                try:
                    # Research application
                    result = await self.research_job_application(job_app)
                    research_results.append(result)
                    
                    # Progress logging
                    if (i + 1) % 5 == 0:
                        logger.info(f"Completed research for {i + 1}/{len(job_apps)} applications")
                    
                    # Rate limiting between applications
                    await asyncio.sleep(self.settings.brave_rate_limit_delay * 2)
                    
                except Exception as e:
                    logger.error(f"Failed to research application {i+1}: {e}")
                    continue
            
            logger.info(f"Batch research complete: {len(research_results)} successful researches")
            
            return research_results
            
        except Exception as e:
            logger.error(f"Failed to research batch: {e}")
            self.stats['errors'].append(f"Batch research failed: {e}")
            raise
    
    async def get_company_insights(self, company: str) -> Dict[str, Any]:
        """
        Get general insights about a company.
        
        Args:
            company (str): Company name
            
        Returns:
            Dict[str, Any]: Company insights
        """
        try:
            logger.info(f"Getting insights for company: {company}")
            
            # Search for company-specific job postings
            postings = await self.brave_tool.search_company_jobs(company)
            
            if not postings:
                return {
                    'company': company,
                    'total_postings': 0,
                    'insights': 'No job postings found for this company'
                }
            
            # Analyze company postings
            insights = {
                'company': company,
                'total_postings': len(postings),
                'active_recruitment': len([p for p in postings if p.is_recent(30)]),
                'common_positions': self._get_common_positions(postings),
                'hiring_locations': self._get_hiring_locations(postings),
                'salary_ranges': self._extract_salary_ranges(postings),
                'recent_activity': self._analyze_recent_activity(postings)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get company insights: {e}")
            return {
                'company': company,
                'error': str(e)
            }
    
    def _get_common_positions(self, postings: List[JobPosting]) -> List[Tuple[str, int]]:
        """Get most common positions from postings."""
        from collections import Counter
        
        positions = [posting.title for posting in postings]
        return Counter(positions).most_common(5)
    
    def _get_hiring_locations(self, postings: List[JobPosting]) -> List[Tuple[str, int]]:
        """Get hiring locations from postings."""
        from collections import Counter
        
        locations = [posting.location for posting in postings if posting.location]
        return Counter(locations).most_common(5)
    
    def _extract_salary_ranges(self, postings: List[JobPosting]) -> List[str]:
        """Extract salary ranges from postings."""
        salary_ranges = []
        for posting in postings:
            if posting.salary_range:
                salary_ranges.append(posting.salary_range)
        return salary_ranges[:5]  # Return top 5
    
    def _analyze_recent_activity(self, postings: List[JobPosting]) -> Dict[str, Any]:
        """Analyze recent hiring activity."""
        recent_postings = [p for p in postings if p.is_recent(30)]
        
        if not recent_postings:
            return {'activity_level': 'low', 'recent_count': 0}
        
        activity_level = 'high' if len(recent_postings) > 5 else 'medium' if len(recent_postings) > 2 else 'low'
        
        return {
            'activity_level': activity_level,
            'recent_count': len(recent_postings),
            'recent_positions': [p.title for p in recent_postings[:3]]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dict[str, Any]: Agent statistics
        """
        return {
            'total_applications_researched': self.stats['total_applications_researched'],
            'successful_researches': self.stats['successful_researches'],
            'failed_researches': self.stats['failed_researches'],
            'total_postings_found': self.stats['total_postings_found'],
            'processing_time': self.stats['processing_time'],
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else [],
            'success_rate': (self.stats['successful_researches'] / self.stats['total_applications_researched']) * 100 if self.stats['total_applications_researched'] > 0 else 0,
            'average_postings_per_research': self.stats['total_postings_found'] / self.stats['successful_researches'] if self.stats['successful_researches'] > 0 else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset agent statistics."""
        self.stats = {
            'total_applications_researched': 0,
            'successful_researches': 0,
            'failed_researches': 0,
            'total_postings_found': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        try:
            await self.brave_tool._close_session()
            logger.info("Research Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Factory function
def create_research_agent() -> ResearchAgent:
    """
    Create and return Research Agent instance.
    
    Returns:
        ResearchAgent: Configured Research Agent
    """
    return ResearchAgent()

# Test function
async def test_research_agent():
    """Test Research Agent functionality."""
    try:
        agent = create_research_agent()
        
        # Test with sample job application
        from models.job_models import JobApplication, ApplicationStatus
        from datetime import datetime, timezone
        
        test_app = JobApplication(
            email_id="test_123",
            company="Google",
            position="Software Engineer",
            status=ApplicationStatus.APPLIED,
            applied_date=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            confidence_score=0.85
        )
        
        print("✅ Test job application created")
        
        # Test research
        result = await agent.research_job_application(test_app)
        
        print(f"✅ Research completed: {result.total_results} postings found")
        print(f"🎯 Average relevance: {result.average_relevance:.2f}")
        
        if result.best_match:
            print(f"🥇 Best match: {result.best_match.title} at {result.best_match.company}")
        
        # Test company insights
        insights = await agent.get_company_insights("Google")
        print(f"🏢 Company insights: {insights.get('total_postings', 0)} postings found")
        
        # Test statistics
        stats = agent.get_statistics()
        print(f"📊 Agent stats: {stats['total_applications_researched']} researched, {stats['success_rate']:.1f}% success rate")
        
    except Exception as e:
        print(f"❌ Research Agent test failed: {e}")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(test_research_agent())