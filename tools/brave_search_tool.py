"""
Brave Search API integration tool for job application tracking.

Handles job posting research using Brave Search API with proper
rate limiting, error handling, and result parsing.
"""

import asyncio
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import httpx

from config.settings import get_settings
from models.job_models import JobApplication
from models.research_models import JobPosting, JobSource, ResearchResult, create_research_result
from utils import logger, clean_text, safe_get

class BraveSearchTool:
    """
    Brave Search API integration tool.
    
    Handles job posting research using Brave Search API with
    proper rate limiting, error handling, and result parsing.
    """
    
    def __init__(self):
        """Initialize Brave Search tool with settings."""
        self.settings = get_settings()
        self.base_url = self.settings.brave_base_url
        self.api_key = self.settings.brave_api_key
        self.session = None
        
        # Job board domains for source identification
        self.job_board_domains = {
            'linkedin.com': JobSource.LINKEDIN,
            'glassdoor.com': JobSource.GLASSDOOR,
            'indeed.com': JobSource.INDEED,
            'monster.com': JobSource.MONSTER,
            'dice.com': JobSource.DICE,
            'ziprecruiter.com': JobSource.ZIPRECRUITER,
            'careerbuilder.com': JobSource.CAREERBUILDER
        }
        
        # Keywords that indicate job postings
        self.job_keywords = [
            'job', 'career', 'position', 'hiring', 'employment', 'vacancy',
            'opening', 'opportunity', 'apply', 'recruit', 'work', 'role'
        ]
    
    async def _get_session(self) -> httpx.AsyncClient:
        """
        Get or create HTTP session.
        
        Returns:
            httpx.AsyncClient: HTTP session
        """
        if not self.session:
            self.session = httpx.AsyncClient(
                headers={
                    'X-Subscription-Token': self.api_key,
                    'User-Agent': 'JobTracker/1.0 (Python/httpx)'
                },
                timeout=30.0
            )
        return self.session
    
    async def _close_session(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None
    
    def _identify_job_source(self, url: str) -> JobSource:
        """
        Identify job source from URL.
        
        Args:
            url (str): Job posting URL
            
        Returns:
            JobSource: Identified source
        """
        try:
            domain = urlparse(url).netloc.lower()
            
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Check known job board domains
            for job_domain, source in self.job_board_domains.items():
                if job_domain in domain:
                    return source
            
            # Check if it's a company careers page
            if any(keyword in domain for keyword in ['career', 'job', 'talent']):
                return JobSource.COMPANY_WEBSITE
            
            return JobSource.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Failed to identify job source from URL {url}: {e}")
            return JobSource.UNKNOWN
    
    def _extract_job_details(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract job details from search result.
        
        Args:
            result (Dict[str, Any]): Search result from Brave API
            
        Returns:
            Optional[Dict[str, Any]]: Extracted job details
        """
        try:
            title = safe_get(result, 'title', '')
            url = safe_get(result, 'url', '')
            description = safe_get(result, 'description', '')
            
            if not title or not url:
                return None
            
            # Check if result looks like a job posting
            if not self._is_job_posting(title, description, url):
                return None
            
            # Extract company name
            company = self._extract_company_name(title, description, url)
            
            # Extract location
            location = self._extract_location(title, description)
            
            # Extract posted date
            posted_date = self._extract_posted_date(description)
            
            # Extract salary range
            salary_range = self._extract_salary_range(description)
            
            # Identify source
            source = self._identify_job_source(url)
            
            return {
                'title': clean_text(title),
                'company': company,
                'location': location,
                'url': url,
                'description': clean_text(description),
                'source': source,
                'posted_date': posted_date,
                'salary_range': salary_range
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract job details from result: {e}")
            return None
    
    def _is_job_posting(self, title: str, description: str, url: str) -> bool:
        """
        Check if search result is a job posting.
        
        Args:
            title (str): Result title
            description (str): Result description
            url (str): Result URL
            
        Returns:
            bool: True if result appears to be a job posting
        """
        # Check for job keywords in title
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in self.job_keywords):
            return True
        
        # Check for job board domains
        if self._identify_job_source(url) != JobSource.UNKNOWN:
            return True
        
        # Check description for job-related content
        description_lower = description.lower()
        job_indicators = [
            'apply', 'hiring', 'position', 'role', 'opportunity',
            'salary', 'benefits', 'qualifications', 'requirements',
            'experience', 'skills', 'remote', 'full-time', 'part-time'
        ]
        
        indicator_count = sum(1 for indicator in job_indicators if indicator in description_lower)
        
        # If multiple indicators present, likely a job posting
        return indicator_count >= 2
    
    def _extract_company_name(self, title: str, description: str, url: str) -> str:
        """
        Extract company name from search result.
        
        Args:
            title (str): Result title
            description (str): Result description
            url (str): Result URL
            
        Returns:
            str: Extracted company name
        """
        # Try to extract from title (common pattern: "Job Title at Company Name")
        title_patterns = [
            r'at\s+([A-Z][a-zA-Z\s&\.,]+?)(?:\s*[-|]|$)',
            r'@\s+([A-Z][a-zA-Z\s&\.,]+?)(?:\s*[-|]|$)',
            r'•\s+([A-Z][a-zA-Z\s&\.,]+?)(?:\s*[-|]|$)',
            r'-\s+([A-Z][a-zA-Z\s&\.,]+?)(?:\s*[-|]|$)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, title)
            if match:
                company = match.group(1).strip()
                if len(company) > 2 and company not in ['Job', 'Career', 'Hiring']:
                    return company
        
        # Try to extract from description
        desc_patterns = [
            r'(?:Company|Organization|Employer):\s*([A-Z][a-zA-Z\s&\.,]+)',
            r'(?:at|@)\s+([A-Z][a-zA-Z\s&\.,]+?)(?:\s+is|,)',
            r'([A-Z][a-zA-Z\s&\.,]+?)\s+is\s+(?:looking|seeking|hiring)'
        ]
        
        for pattern in desc_patterns:
            match = re.search(pattern, description)
            if match:
                company = match.group(1).strip()
                if len(company) > 2:
                    return company
        
        # Try to extract from URL domain
        try:
            domain = urlparse(url).netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Remove common extensions
            domain = domain.replace('.com', '').replace('.org', '').replace('.net', '')
            
            # Skip job boards
            if domain not in ['linkedin', 'glassdoor', 'indeed', 'monster', 'dice']:
                return domain.replace('-', ' ').title()
        
        except Exception:
            pass
        
        return "Unknown Company"
    
    def _extract_location(self, title: str, description: str) -> str:
        """
        Extract location from search result.
        
        Args:
            title (str): Result title
            description (str): Result description
            
        Returns:
            str: Extracted location
        """
        # Common location patterns
        location_patterns = [
            r'(?:in|@)\s+([A-Z][a-zA-Z\s,]+(?:,\s*[A-Z]{2})?)',
            r'Location:\s*([A-Z][a-zA-Z\s,]+(?:,\s*[A-Z]{2})?)',
            r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})',  # City, State
            r'Remote|Work from home|WFH'
        ]
        
        text = f"{title} {description}"
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip() if match.lastindex else match.group(0)
                if len(location) > 2:
                    return location
        
        return "Location not specified"
    
    def _extract_posted_date(self, description: str) -> Optional[datetime]:
        """
        Extract posted date from description.
        
        Args:
            description (str): Result description
            
        Returns:
            Optional[datetime]: Extracted date
        """
        # Date patterns
        date_patterns = [
            r'Posted\s+(\d+)\s+days?\s+ago',
            r'(\d+)\s+days?\s+ago',
            r'Posted\s+(\d+)\s+hours?\s+ago',
            r'(\d+)\s+hours?\s+ago',
            r'Posted\s+yesterday',
            r'yesterday',
            r'Posted\s+today',
            r'today'
        ]
        
        description_lower = description.lower()
        
        for pattern in date_patterns:
            match = re.search(pattern, description_lower)
            if match:
                if 'day' in pattern:
                    try:
                        days = int(match.group(1))
                        return datetime.now() - timedelta(days=days)
                    except (ValueError, IndexError):
                        pass
                elif 'hour' in pattern:
                    try:
                        hours = int(match.group(1))
                        return datetime.now() - timedelta(hours=hours)
                    except (ValueError, IndexError):
                        pass
                elif 'yesterday' in pattern:
                    return datetime.now() - timedelta(days=1)
                elif 'today' in pattern:
                    return datetime.now()
        
        return None
    
    def _extract_salary_range(self, description: str) -> Optional[str]:
        """
        Extract salary range from description.
        
        Args:
            description (str): Result description
            
        Returns:
            Optional[str]: Extracted salary range
        """
        # Salary patterns
        salary_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*[-–]\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per|/)\s*(?:year|annum|yr)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*k\s*[-–]\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*k',
            r'Salary:\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        return None
    
    async def search_job_postings(self, query: str, max_results: int = None) -> List[JobPosting]:
        """
        Search for job postings using Brave Search API.
        
        Args:
            query (str): Search query
            max_results (int, optional): Maximum results to return
            
        Returns:
            List[JobPosting]: List of job postings
        """
        max_results = max_results or self.settings.research_max_results
        
        try:
            session = await self._get_session()
            
            # Prepare search parameters
            params = {
                'q': query,
                'count': min(max_results, 20),  # Brave API limit
                'safesearch': 'off',
                'search_lang': 'en',
                'country': 'US',
                'text_decorations': 'false',
                'spellcheck': 'false'
            }
            
            logger.info(f"Searching Brave API with query: {query}")
            
            # Make API request
            response = await session.get(
                f"{self.base_url}/web/search",
                params=params
            )
            
            # Rate limiting
            await asyncio.sleep(self.settings.brave_rate_limit_delay)
            
            if response.status_code != 200:
                logger.error(f"Brave API error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            web_results = safe_get(data, 'web', {}).get('results', [])
            
            logger.info(f"Received {len(web_results)} search results")
            
            # Process results
            job_postings = []
            
            for result in web_results:
                job_details = self._extract_job_details(result)
                if job_details:
                    try:
                        job_posting = JobPosting(**job_details)
                        job_postings.append(job_posting)
                    except Exception as e:
                        logger.warning(f"Failed to create JobPosting: {e}")
                        continue
            
            logger.info(f"Extracted {len(job_postings)} job postings")
            return job_postings
            
        except httpx.TimeoutException:
            logger.error(f"Timeout searching for: {query}")
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error searching for {query}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching for {query}: {e}")
            return []
    
    async def research_job_application(self, job_app: JobApplication) -> ResearchResult:
        """
        Research job postings for a specific job application.
        
        Args:
            job_app (JobApplication): Job application to research
            
        Returns:
            ResearchResult: Research results
        """
        # Create search query
        base_query = f"{job_app.company} {job_app.position}"
        
        # Add location if available
        if job_app.location:
            location_query = f"{base_query} {job_app.location}"
        else:
            location_query = base_query
        
        # Add job-specific keywords
        job_query = f"{location_query} job hiring"
        
        logger.info(f"Researching job application: {job_app.company} - {job_app.position}")
        
        # Search for job postings
        job_postings = await self.search_job_postings(job_query)
        
        # Create research result
        research_result = create_research_result(
            query=job_query,
            target_company=job_app.company,
            target_position=job_app.position,
            source_used="Brave Search API"
        )
        
        research_result.job_postings = job_postings
        
        # Analyze results
        research_result.analyze_results()
        
        logger.info(f"Research complete: {len(job_postings)} postings found, best match score: {research_result.best_match.relevance_score if research_result.best_match else 0:.2f}")
        
        return research_result
    
    async def search_company_jobs(self, company: str, position_type: str = None) -> List[JobPosting]:
        """
        Search for jobs at a specific company.
        
        Args:
            company (str): Company name
            position_type (str, optional): Type of position
            
        Returns:
            List[JobPosting]: Job postings at the company
        """
        query = f"{company} jobs careers"
        
        if position_type:
            query += f" {position_type}"
        
        # Add site-specific searches for better results
        site_queries = [
            f"site:linkedin.com/jobs {company}",
            f"site:glassdoor.com {company} jobs",
            f"site:{company.lower().replace(' ', '')}.com careers"
        ]
        
        all_postings = []
        
        for site_query in site_queries:
            try:
                postings = await self.search_job_postings(site_query, max_results=5)
                all_postings.extend(postings)
                
                # Add delay between searches
                await asyncio.sleep(self.settings.brave_rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Failed to search with query '{site_query}': {e}")
                continue
        
        # Remove duplicates based on URL
        unique_postings = []
        seen_urls = set()
        
        for posting in all_postings:
            if posting.url not in seen_urls:
                unique_postings.append(posting)
                seen_urls.add(posting.url)
        
        return unique_postings
    
    async def test_connection(self) -> bool:
        """
        Test Brave Search API connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            session = await self._get_session()
            
            # Test with simple query
            params = {
                'q': 'software engineer jobs',
                'count': 1,
                'safesearch': 'off'
            }
            
            response = await session.get(
                f"{self.base_url}/web/search",
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                results = safe_get(data, 'web', {}).get('results', [])
                
                logger.info(f"Brave API connection successful. Test query returned {len(results)} results")
                return True
            else:
                logger.error(f"Brave API connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Brave API connection test failed: {e}")
            return False
        finally:
            await self._close_session()
    
    async def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get search suggestions for a query.
        
        Args:
            query (str): Base query
            
        Returns:
            List[str]: List of suggested queries
        """
        # Generate common job search variations
        base_terms = query.split()
        
        suggestions = []
        
        # Add location variations
        locations = ['remote', 'san francisco', 'new york', 'seattle', 'austin']
        for location in locations:
            suggestions.append(f"{query} {location}")
        
        # Add job type variations
        job_types = ['full time', 'part time', 'contract', 'internship']
        for job_type in job_types:
            suggestions.append(f"{query} {job_type}")
        
        # Add experience level variations
        levels = ['entry level', 'senior', 'junior', 'lead']
        for level in levels:
            suggestions.append(f"{level} {query}")
        
        return suggestions[:10]  # Limit to top 10 suggestions

# Factory function
def create_brave_search_tool() -> BraveSearchTool:
    """
    Create and return Brave Search tool instance.
    
    Returns:
        BraveSearchTool: Configured Brave Search tool
    """
    return BraveSearchTool()

# Test function
async def test_brave_search_tool():
    """Test Brave Search tool functionality."""
    try:
        tool = create_brave_search_tool()
        
        # Test connection
        success = await tool.test_connection()
        if not success:
            print("❌ Brave Search API connection failed")
            return
        
        print("✅ Brave Search API connection successful")
        
        # Test job search
        query = "Google software engineer"
        postings = await tool.search_job_postings(query, max_results=3)
        
        print(f"✅ Job search returned {len(postings)} results")
        
        if postings:
            sample_posting = postings[0]
            print(f"💼 Sample posting: {sample_posting.title}")
            print(f"🏢 Company: {sample_posting.company}")
            print(f"📍 Location: {sample_posting.location}")
            print(f"🔗 Source: {sample_posting.source.value}")
            print(f"🎯 Relevance: {sample_posting.relevance_score:.2f}")
        
        # Test suggestions
        suggestions = await tool.get_search_suggestions("python developer")
        print(f"💡 Search suggestions: {suggestions[:3]}")
        
    except Exception as e:
        print(f"❌ Brave Search tool test failed: {e}")
    finally:
        await tool._close_session()

if __name__ == "__main__":
    asyncio.run(test_brave_search_tool())