"""
Research result data models for job application tracking system.

Contains Pydantic models for job posting research results from
external sources like LinkedIn, Glassdoor, and other job boards.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator, root_validator, model_validator
from urllib.parse import urlparse, parse_qs
import re

class JobSource(str, Enum):
    """Job posting source platforms."""
    LINKEDIN = "linkedin"
    GLASSDOOR = "glassdoor"
    INDEED = "indeed"
    MONSTER = "monster"
    DICE = "dice"
    ZIPRECRUITER = "ziprecruiter"
    CAREERBUILDER = "careerbuilder"
    COMPANY_WEBSITE = "company_website"
    UNKNOWN = "unknown"

class JobType(str, Enum):
    """Job types."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"
    UNKNOWN = "unknown"

class ExperienceLevel(str, Enum):
    """Experience levels."""
    ENTRY_LEVEL = "entry_level"
    ASSOCIATE = "associate"
    MID_LEVEL = "mid_level"
    SENIOR_LEVEL = "senior_level"
    DIRECTOR = "director"
    EXECUTIVE = "executive"
    UNKNOWN = "unknown"

class JobPosting(BaseModel):
    """
    Job posting model for research results.
    
    Represents a job posting found during research with all relevant
    metadata, relevance scoring, and analysis capabilities.
    """
    
    # Core information
    title: str = Field(..., description="Job posting title")
    company: str = Field(..., description="Company name")
    location: str = Field(..., description="Job location")
    url: str = Field(..., description="Job posting URL")
    description: str = Field(..., description="Job description excerpt")
    source: JobSource = Field(..., description="Source platform")
    
    # Additional details
    posted_date: Optional[datetime] = Field(None, description="Date job was posted")
    salary_range: Optional[str] = Field(None, description="Salary range if available")
    job_type: JobType = Field(JobType.UNKNOWN, description="Type of job")
    experience_level: ExperienceLevel = Field(ExperienceLevel.UNKNOWN, description="Required experience level")
    
    # Requirements and qualifications
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    education_requirements: Optional[str] = Field(None, description="Education requirements")
    
    # Company information
    company_size: Optional[str] = Field(None, description="Company size")
    industry: Optional[str] = Field(None, description="Industry sector")
    company_description: Optional[str] = Field(None, description="Company description")
    
    # Scoring and analysis
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Relevance score")
    match_factors: Dict[str, float] = Field(default_factory=dict, description="Matching factors")
    
    # Metadata
    scraped_at: datetime = Field(default_factory=datetime.now, description="When data was scraped")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    
    # Application tracking
    applied: bool = Field(False, description="Whether user has applied")
    application_date: Optional[datetime] = Field(None, description="Date of application")
    application_status: Optional[str] = Field(None, description="Application status")
    
    @validator('title')
    def validate_title(cls, v):
        """
        Validate and clean job title.
        
        Args:
            v (str): Job title
            
        Returns:
            str: Cleaned title
            
        Raises:
            ValueError: If title is empty
        """
        if not v or v.strip() == "":
            raise ValueError("Job title cannot be empty")
        
        # Clean title
        title = v.strip()
        
        # Remove excessive whitespace
        title = re.sub(r'\s+', ' ', title)
        
        # Remove common job board prefixes
        prefixes = ['Job:', 'Position:', 'Opening:', 'Opportunity:']
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        return title
    
    @validator('company')
    def validate_company(cls, v):
        """
        Validate and normalize company name.
        
        Args:
            v (str): Company name
            
        Returns:
            str: Normalized company name
        """
        if not v or v.strip() == "":
            raise ValueError("Company name cannot be empty")
        
        # Clean company name
        company = v.strip()
        
        # Remove common suffixes
        suffixes = [', Inc.', ', LLC', ', Corp.', ', Ltd.', ' Inc.', ' LLC', ' Corp.', ' Ltd.']
        for suffix in suffixes:
            if company.endswith(suffix):
                company = company[:-len(suffix)]
                break
        
        return company.title()
    
    @validator('url')
    def validate_url(cls, v):
        """
        Validate job posting URL.
        
        Args:
            v (str): Job posting URL
            
        Returns:
            str: Validated URL
            
        Raises:
            ValueError: If URL is invalid
        """
        if not v:
            raise ValueError("Job posting URL cannot be empty")
        
        try:
            parsed = urlparse(v)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Invalid URL format")
            return v
        except Exception:
            raise ValueError("Invalid URL format")
    
    @validator('description')
    def validate_description(cls, v):
        """
        Validate and clean job description.
        
        Args:
            v (str): Job description
            
        Returns:
            str: Cleaned description
        """
        if not v:
            return ""
        
        # Clean description
        description = v.strip()
        
        # Remove excessive whitespace
        description = re.sub(r'\s+', ' ', description)
        
        # Truncate if too long
        if len(description) > 2000:
            description = description[:2000] + "..."
        
        return description
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        """
        Validate relevance score range.
        
        Args:
            v (float): Relevance score
            
        Returns:
            float: Validated score
        """
        return max(0.0, min(1.0, v))
    
    @model_validator(mode='before')
    @classmethod
    def set_derived_fields(cls, values):
        """
        Set derived fields based on other values.
        
        Args:
            values (dict): Field values
            
        Returns:
            dict: Updated values with derived fields
        """
        url = values.get('url', '')
        title = values.get('title', '').lower()
        
        # Infer source from URL
        if url and values.get('source') == JobSource.UNKNOWN:
            if 'linkedin.com' in url:
                values['source'] = JobSource.LINKEDIN
            elif 'glassdoor.com' in url:
                values['source'] = JobSource.GLASSDOOR
            elif 'indeed.com' in url:
                values['source'] = JobSource.INDEED
            elif 'monster.com' in url:
                values['source'] = JobSource.MONSTER
            elif 'dice.com' in url:
                values['source'] = JobSource.DICE
            elif 'ziprecruiter.com' in url:
                values['source'] = JobSource.ZIPRECRUITER
            elif 'careerbuilder.com' in url:
                values['source'] = JobSource.CAREERBUILDER
            else:
                values['source'] = JobSource.COMPANY_WEBSITE
        
        # Infer job type from title
        if values.get('job_type') == JobType.UNKNOWN:
            if any(word in title for word in ['intern', 'internship']):
                values['job_type'] = JobType.INTERNSHIP
            elif any(word in title for word in ['contract', 'contractor', 'freelance']):
                values['job_type'] = JobType.CONTRACT
            elif any(word in title for word in ['part-time', 'part time']):
                values['job_type'] = JobType.PART_TIME
            elif any(word in title for word in ['temporary', 'temp']):
                values['job_type'] = JobType.TEMPORARY
            else:
                values['job_type'] = JobType.FULL_TIME
        
        # Infer experience level from title
        if values.get('experience_level') == ExperienceLevel.UNKNOWN:
            if any(word in title for word in ['senior', 'sr', 'lead', 'principal']):
                values['experience_level'] = ExperienceLevel.SENIOR_LEVEL
            elif any(word in title for word in ['junior', 'jr', 'entry', 'graduate']):
                values['experience_level'] = ExperienceLevel.ENTRY_LEVEL
            elif any(word in title for word in ['director', 'vp', 'vice president']):
                values['experience_level'] = ExperienceLevel.DIRECTOR
            elif any(word in title for word in ['ceo', 'cto', 'cfo', 'chief']):
                values['experience_level'] = ExperienceLevel.EXECUTIVE
            else:
                values['experience_level'] = ExperienceLevel.MID_LEVEL
        
        return values
    
    def extract_skills_from_description(self) -> List[str]:
        """
        Extract skills from job description using keyword matching.
        
        Returns:
            List[str]: List of extracted skills
        """
        if not self.description:
            return []
        
        # Common tech skills to look for
        tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'terraform',
            'sql', 'postgresql', 'mysql', 'mongodb', 'redis',
            'git', 'github', 'gitlab', 'jenkins', 'ci/cd',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'agile', 'scrum', 'kanban', 'devops', 'microservices'
        ]
        
        description_lower = self.description.lower()
        found_skills = []
        
        for skill in tech_skills:
            if skill in description_lower:
                found_skills.append(skill.title())
        
        return found_skills
    
    def calculate_title_similarity(self, target_title: str) -> float:
        """
        Calculate similarity between job title and target title.
        
        Args:
            target_title (str): Target title to compare against
            
        Returns:
            float: Similarity score (0.0-1.0)
        """
        def normalize_title(title: str) -> set:
            """Normalize title to set of words."""
            title_lower = title.lower()
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\b\w+\b', title_lower)
            return set(word for word in words if word not in stop_words and len(word) > 2)
        
        title_words = normalize_title(self.title)
        target_words = normalize_title(target_title)
        
        if not title_words or not target_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = title_words.intersection(target_words)
        union = title_words.union(target_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_company_similarity(self, target_company: str) -> float:
        """
        Calculate similarity between company names.
        
        Args:
            target_company (str): Target company to compare against
            
        Returns:
            float: Similarity score (0.0-1.0)
        """
        company_lower = self.company.lower()
        target_lower = target_company.lower()
        
        # Exact match
        if company_lower == target_lower:
            return 1.0
        
        # Contains match
        if target_lower in company_lower or company_lower in target_lower:
            return 0.8
        
        # Word overlap
        company_words = set(re.findall(r'\b\w+\b', company_lower))
        target_words = set(re.findall(r'\b\w+\b', target_lower))
        
        if not company_words or not target_words:
            return 0.0
        
        intersection = company_words.intersection(target_words)
        union = company_words.union(target_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_location_match(self, target_location: str) -> float:
        """
        Calculate location match score.
        
        Args:
            target_location (str): Target location
            
        Returns:
            float: Location match score (0.0-1.0)
        """
        if not self.location or not target_location:
            return 0.0
        
        location_lower = self.location.lower()
        target_lower = target_location.lower()
        
        # Remote work
        if 'remote' in location_lower or 'remote' in target_lower:
            return 1.0
        
        # Exact match
        if location_lower == target_lower:
            return 1.0
        
        # City/state matching
        location_parts = re.findall(r'\b\w+\b', location_lower)
        target_parts = re.findall(r'\b\w+\b', target_lower)
        
        matches = sum(1 for part in location_parts if part in target_parts)
        total = max(len(location_parts), len(target_parts))
        
        return matches / total if total > 0 else 0.0
    
    def is_recent(self, days: int = 30) -> bool:
        """
        Check if job posting is recent.
        
        Args:
            days (int): Number of days to consider recent
            
        Returns:
            bool: True if posting is recent
        """
        if not self.posted_date:
            return True  # Assume recent if no date available
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        return self.posted_date > cutoff_date
    
    def get_age_in_days(self) -> Optional[int]:
        """
        Get age of job posting in days.
        
        Returns:
            Optional[int]: Age in days, None if posted_date not available
        """
        if not self.posted_date:
            return None
        
        now = datetime.now(timezone.utc)
        posted_date_aware = self.posted_date.replace(tzinfo=timezone.utc) if self.posted_date.tzinfo is None else self.posted_date
        return (now - posted_date_aware).days
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert job posting to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Job posting data as dictionary
        """
        return {
            'title': self.title,
            'company': self.company,
            'location': self.location,
            'url': self.url,
            'description_preview': self.description[:200] + "..." if len(self.description) > 200 else self.description,
            'source': self.source.value,
            'posted_date': self.posted_date.isoformat() if self.posted_date else None,
            'salary_range': self.salary_range,
            'job_type': self.job_type.value,
            'experience_level': self.experience_level.value,
            'relevance_score': self.relevance_score,
            'match_factors': self.match_factors,
            'required_skills': self.required_skills,
            'preferred_skills': self.preferred_skills,
            'is_recent': self.is_recent(),
            'age_in_days': self.get_age_in_days(),
            'applied': self.applied,
            'application_date': self.application_date.isoformat() if self.application_date else None,
            'application_status': self.application_status
        }
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True

class ResearchResult(BaseModel):
    """
    Research result containing multiple job postings.
    
    Represents the result of a job search query with metadata
    about the search and aggregated results.
    """
    
    query: str = Field(..., description="Search query used")
    target_company: str = Field(..., description="Target company name")
    target_position: str = Field(..., description="Target position title")
    
    job_postings: List[JobPosting] = Field(default_factory=list, description="Found job postings")
    
    # Search metadata
    search_timestamp: datetime = Field(default_factory=datetime.now, description="When search was performed")
    source_used: str = Field(..., description="Search source used")
    total_results: int = Field(0, description="Total results found")
    
    # Analysis results
    best_match: Optional[JobPosting] = Field(None, description="Best matching job posting")
    average_relevance: float = Field(0.0, description="Average relevance score")
    
    def analyze_results(self) -> None:
        """
        Analyze job postings and set best match and statistics.
        """
        if not self.job_postings:
            return
        
        # Calculate relevance scores
        for posting in self.job_postings:
            factors = {}
            
            # Title similarity
            title_sim = posting.calculate_title_similarity(self.target_position)
            factors['title_similarity'] = title_sim
            
            # Company similarity
            company_sim = posting.calculate_company_similarity(self.target_company)
            factors['company_similarity'] = company_sim
            
            # Recency factor
            if posting.is_recent():
                factors['recency'] = 1.0
            else:
                factors['recency'] = 0.5
            
            # Source credibility
            source_weights = {
                JobSource.LINKEDIN: 0.9,
                JobSource.GLASSDOOR: 0.8,
                JobSource.INDEED: 0.7,
                JobSource.COMPANY_WEBSITE: 1.0,
                JobSource.UNKNOWN: 0.5
            }
            factors['source_credibility'] = source_weights.get(posting.source, 0.5)
            
            # Calculate weighted relevance score
            weights = {
                'title_similarity': 0.4,
                'company_similarity': 0.3,
                'recency': 0.2,
                'source_credibility': 0.1
            }
            
            relevance = sum(factors[key] * weights[key] for key in factors)
            posting.relevance_score = relevance
            posting.match_factors = factors
        
        # Sort by relevance
        self.job_postings.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Set best match
        if self.job_postings:
            self.best_match = self.job_postings[0]
        
        # Calculate average relevance
        if self.job_postings:
            self.average_relevance = sum(p.relevance_score for p in self.job_postings) / len(self.job_postings)
        
        # Update total results
        self.total_results = len(self.job_postings)
    
    def get_top_matches(self, count: int = 5) -> List[JobPosting]:
        """
        Get top matching job postings.
        
        Args:
            count (int): Number of top matches to return
            
        Returns:
            List[JobPosting]: Top matching postings
        """
        return self.job_postings[:count]
    
    def filter_by_source(self, source: JobSource) -> List[JobPosting]:
        """
        Filter job postings by source.
        
        Args:
            source (JobSource): Source to filter by
            
        Returns:
            List[JobPosting]: Filtered postings
        """
        return [posting for posting in self.job_postings if posting.source == source]
    
    def filter_by_relevance(self, min_score: float = 0.5) -> List[JobPosting]:
        """
        Filter job postings by minimum relevance score.
        
        Args:
            min_score (float): Minimum relevance score
            
        Returns:
            List[JobPosting]: Filtered postings
        """
        return [posting for posting in self.job_postings if posting.relevance_score >= min_score]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get research result statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        if not self.job_postings:
            return {}
        
        # Source distribution
        source_counts = {}
        for posting in self.job_postings:
            source = posting.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Relevance distribution
        high_relevance = len([p for p in self.job_postings if p.relevance_score >= 0.7])
        medium_relevance = len([p for p in self.job_postings if 0.4 <= p.relevance_score < 0.7])
        low_relevance = len([p for p in self.job_postings if p.relevance_score < 0.4])
        
        return {
            'total_results': self.total_results,
            'average_relevance': self.average_relevance,
            'best_match_score': self.best_match.relevance_score if self.best_match else 0.0,
            'source_distribution': source_counts,
            'relevance_distribution': {
                'high': high_relevance,
                'medium': medium_relevance,
                'low': low_relevance
            },
            'recent_postings': len([p for p in self.job_postings if p.is_recent()]),
            'search_timestamp': self.search_timestamp.isoformat()
        }

# Helper functions
def create_job_posting(title: str, company: str, location: str, url: str, 
                      description: str, source: JobSource) -> JobPosting:
    """
    Create job posting with basic information.
    
    Args:
        title (str): Job title
        company (str): Company name
        location (str): Job location
        url (str): Job URL
        description (str): Job description
        source (JobSource): Job source
        
    Returns:
        JobPosting: Created job posting
    """
    return JobPosting(
        title=title,
        company=company,
        location=location,
        url=url,
        description=description,
        source=source
    )

def create_research_result(query: str, target_company: str, target_position: str,
                         source_used: str) -> ResearchResult:
    """
    Create research result container.
    
    Args:
        query (str): Search query
        target_company (str): Target company
        target_position (str): Target position
        source_used (str): Source used for search
        
    Returns:
        ResearchResult: Created research result
    """
    return ResearchResult(
        query=query,
        target_company=target_company,
        target_position=target_position,
        source_used=source_used
    )

if __name__ == "__main__":
    # Test research models
    
    # Test job posting creation
    test_posting = create_job_posting(
        title="Senior Software Engineer",
        company="Google",
        location="Mountain View, CA",
        url="https://linkedin.com/jobs/12345",
        description="We are looking for a senior software engineer to join our team...",
        source=JobSource.LINKEDIN
    )
    
    print("✅ Job posting model created successfully")
    print(f"💼 Title: {test_posting.title}")
    print(f"🏢 Company: {test_posting.company}")
    print(f"📍 Location: {test_posting.location}")
    print(f"🔗 Source: {test_posting.source.value}")
    print(f"💼 Job Type: {test_posting.job_type.value}")
    print(f"📊 Experience Level: {test_posting.experience_level.value}")
    
    # Test similarity calculations
    title_sim = test_posting.calculate_title_similarity("Software Engineer")
    company_sim = test_posting.calculate_company_similarity("Google")
    
    print(f"🎯 Title similarity: {title_sim:.2f}")
    print(f"🎯 Company similarity: {company_sim:.2f}")
    
    # Test research result
    research = create_research_result(
        query="Google Software Engineer",
        target_company="Google",
        target_position="Software Engineer",
        source_used="Brave Search"
    )
    
    research.job_postings = [test_posting]
    research.analyze_results()
    
    print(f"📈 Research results: {research.total_results} postings")
    print(f"📊 Average relevance: {research.average_relevance:.2f}")
    
    if research.best_match:
        print(f"🥇 Best match: {research.best_match.title} at {research.best_match.company}")
    
    print("✅ Research models working correctly")