"""
Status Agent for job application tracking system.

Handles aggregation of job application data and generates comprehensive
status reports, analytics, and insights for job search tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

from models.job_models import JobApplication, ApplicationStatus, ApplicationSummary, create_application_summary
from models.research_models import ResearchResult
from config.settings import get_settings
from utils import logger, format_datetime, call_llm, parse_yaml_response

class StatusAgent:
    """
    Status Agent for job application analytics and reporting.
    
    Aggregates job application data and generates comprehensive status reports,
    analytics, and insights following PocketFlow reduce patterns.
    """
    
    def __init__(self):
        """Initialize Status Agent with settings."""
        self.settings = get_settings()
        
        # Processing statistics
        self.stats = {
            'total_reports_generated': 0,
            'applications_processed': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }
        
        # Insights generation prompt
        self.insights_prompt_template = """
        Analyze this job search data and provide actionable insights:
        
        Job Search Summary:
        - Total Applications: {total_applications}
        - Response Rate: {response_rate:.1%}
        - Interview Rate: {interview_rate:.1%}
        - Offer Rate: {offer_rate:.1%}
        - Average Days Since Applied: {avg_days_applied:.1f}
        
        Status Distribution:
        {status_distribution}
        
        Top Companies Applied To:
        {top_companies}
        
        Top Positions Applied For:
        {top_positions}
        
        Recent Application Activity:
        {recent_activity}
        
        Please provide actionable insights and recommendations for improving the job search:
        
        Output in YAML format:
        ```yaml
        overall_assessment: "Brief assessment of job search performance"
        strengths: ["strength1", "strength2"]
        areas_for_improvement: ["area1", "area2"]
        specific_recommendations: ["recommendation1", "recommendation2"]
        next_steps: ["step1", "step2"]
        market_outlook: "Assessment of current job market conditions"
        confidence: 0.85
        ```
        """
    
    def generate_status_summary(self, applications: List[JobApplication]) -> ApplicationSummary:
        """
        Generate comprehensive status summary from job applications.
        
        Args:
            applications (List[JobApplication]): Job applications to analyze
            
        Returns:
            ApplicationSummary: Comprehensive summary with statistics
        """
        try:
            logger.info(f"Generating status summary for {len(applications)} applications")
            
            # Create summary using the model method
            summary = create_application_summary(applications)
            
            logger.info(f"Status summary generated: {summary.total_applications} applications, {summary.response_rate:.1%} response rate")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate status summary: {e}")
            self.stats['errors'].append(f"Status summary generation failed: {e}")
            raise
    
    def generate_detailed_report(self, applications: List[JobApplication], 
                               research_results: List[ResearchResult] = None) -> Dict[str, Any]:
        """
        Generate detailed status report with comprehensive analytics.
        
        Args:
            applications (List[JobApplication]): Job applications
            research_results (List[ResearchResult], optional): Research results
            
        Returns:
            Dict[str, Any]: Detailed report with analytics
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            logger.info(f"Generating detailed report for {len(applications)} applications")
            
            # Basic summary
            summary = self.generate_status_summary(applications)
            
            # Time-based analysis
            time_analysis = self._analyze_application_timeline(applications)
            
            # Company analysis
            company_analysis = self._analyze_company_patterns(applications)
            
            # Position analysis
            position_analysis = self._analyze_position_patterns(applications)
            
            # Status flow analysis
            status_flow = self._analyze_status_flow(applications)
            
            # Success metrics
            success_metrics = self._calculate_success_metrics(applications)
            
            # Stale applications analysis
            stale_analysis = self._analyze_stale_applications(applications)
            
            # Research insights (if available)
            research_insights = {}
            if research_results:
                research_insights = self._analyze_research_results(research_results)
            
            # Compile detailed report
            detailed_report = {
                'summary': summary,
                'time_analysis': time_analysis,
                'company_analysis': company_analysis,
                'position_analysis': position_analysis,
                'status_flow': status_flow,
                'success_metrics': success_metrics,
                'stale_analysis': stale_analysis,
                'research_insights': research_insights,
                'report_metadata': {
                    'generated_at': datetime.now(timezone.utc),
                    'total_applications': len(applications),
                    'report_type': 'detailed',
                    'processing_time': (datetime.now(timezone.utc) - start_time).total_seconds()
                }
            }
            
            # Update statistics
            self.stats['total_reports_generated'] += 1
            self.stats['applications_processed'] += len(applications)
            self.stats['processing_time'] += (datetime.now(timezone.utc) - start_time).total_seconds()
            self.stats['last_run'] = datetime.now(timezone.utc)
            
            logger.info(f"Detailed report generated successfully")
            
            return detailed_report
            
        except Exception as e:
            logger.error(f"Failed to generate detailed report: {e}")
            self.stats['errors'].append(f"Detailed report generation failed: {e}")
            raise
    
    def _analyze_application_timeline(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Analyze application timeline patterns.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Timeline analysis
        """
        if not applications:
            return {}
        
        # Sort by application date
        sorted_apps = sorted(applications, key=lambda x: x.applied_date)
        
        # Date range (ensure timezone-aware comparison)
        earliest = sorted_apps[0].applied_date
        latest = sorted_apps[-1].applied_date
        
        # Ensure both dates are timezone-aware for comparison
        if earliest.tzinfo is None:
            earliest = earliest.replace(tzinfo=timezone.utc)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
            
        span_days = (latest - earliest).days
        
        # Application frequency
        daily_counts = defaultdict(int)
        weekly_counts = defaultdict(int)
        monthly_counts = defaultdict(int)
        
        for app in applications:
            date_key = app.applied_date.date()
            week_key = app.applied_date.isocalendar()[:2]  # (year, week)
            month_key = (app.applied_date.year, app.applied_date.month)
            
            daily_counts[date_key] += 1
            weekly_counts[week_key] += 1
            monthly_counts[month_key] += 1
        
        # Calculate averages
        avg_per_day = len(applications) / max(span_days, 1)
        avg_per_week = len(applications) / max(len(weekly_counts), 1)
        avg_per_month = len(applications) / max(len(monthly_counts), 1)
        
        # Recent activity (last 30 days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_apps = []
        for app in applications:
            app_date = app.applied_date
            # Ensure timezone-aware comparison
            if app_date.tzinfo is None:
                app_date = app_date.replace(tzinfo=timezone.utc)
            if app_date > recent_cutoff:
                recent_apps.append(app)
        
        # Peak periods
        peak_day = max(daily_counts.items(), key=lambda x: x[1]) if daily_counts else None
        peak_week = max(weekly_counts.items(), key=lambda x: x[1]) if weekly_counts else None
        peak_month = max(monthly_counts.items(), key=lambda x: x[1]) if monthly_counts else None
        
        return {
            'date_range': {
                'earliest': earliest,
                'latest': latest,
                'span_days': span_days
            },
            'frequency': {
                'avg_per_day': avg_per_day,
                'avg_per_week': avg_per_week,
                'avg_per_month': avg_per_month
            },
            'recent_activity': {
                'last_30_days': len(recent_apps),
                'recent_rate': len(recent_apps) / 30.0
            },
            'peak_periods': {
                'peak_day': {'date': peak_day[0], 'count': peak_day[1]} if peak_day else None,
                'peak_week': {'week': peak_week[0], 'count': peak_week[1]} if peak_week else None,
                'peak_month': {'month': peak_month[0], 'count': peak_month[1]} if peak_month else None
            },
            'distribution': {
                'daily_variance': self._calculate_variance([count for count in daily_counts.values()]),
                'consistency_score': self._calculate_consistency_score(daily_counts)
            }
        }
    
    def _analyze_company_patterns(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Analyze company application patterns.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Company analysis
        """
        if not applications:
            return {}
        
        # Company counts
        company_counts = Counter(app.company for app in applications)
        
        # Company success rates
        company_success = defaultdict(lambda: {'total': 0, 'responded': 0, 'interviewed': 0, 'offered': 0})
        
        for app in applications:
            company = app.company
            company_success[company]['total'] += 1
            
            if app.status in [ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER]:
                company_success[company]['responded'] += 1
            
            if app.status in [ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER]:
                company_success[company]['interviewed'] += 1
            
            if app.status == ApplicationStatus.OFFER:
                company_success[company]['offered'] += 1
        
        # Calculate rates
        company_rates = {}
        for company, stats in company_success.items():
            if stats['total'] > 0:
                company_rates[company] = {
                    'applications': stats['total'],
                    'response_rate': stats['responded'] / stats['total'],
                    'interview_rate': stats['interviewed'] / stats['total'],
                    'offer_rate': stats['offered'] / stats['total']
                }
        
        # Top performing companies
        top_companies = sorted(company_rates.items(), 
                             key=lambda x: (x[1]['response_rate'], x[1]['applications']), 
                             reverse=True)[:5]
        
        return {
            'total_companies': len(company_counts),
            'company_distribution': dict(company_counts.most_common(10)),
            'company_success_rates': company_rates,
            'top_performing_companies': top_companies,
            'diversification_score': self._calculate_diversification_score(company_counts),
            'concentration_risk': self._calculate_concentration_risk(company_counts)
        }
    
    def _analyze_position_patterns(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Analyze position application patterns.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Position analysis
        """
        if not applications:
            return {}
        
        # Position counts
        position_counts = Counter(app.position for app in applications)
        
        # Position success rates
        position_success = defaultdict(lambda: {'total': 0, 'responded': 0, 'interviewed': 0, 'offered': 0})
        
        for app in applications:
            position = app.position
            position_success[position]['total'] += 1
            
            if app.status in [ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER]:
                position_success[position]['responded'] += 1
            
            if app.status in [ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER]:
                position_success[position]['interviewed'] += 1
            
            if app.status == ApplicationStatus.OFFER:
                position_success[position]['offered'] += 1
        
        # Calculate rates
        position_rates = {}
        for position, stats in position_success.items():
            if stats['total'] > 0:
                position_rates[position] = {
                    'applications': stats['total'],
                    'response_rate': stats['responded'] / stats['total'],
                    'interview_rate': stats['interviewed'] / stats['total'],
                    'offer_rate': stats['offered'] / stats['total']
                }
        
        # Top performing positions
        top_positions = sorted(position_rates.items(), 
                             key=lambda x: (x[1]['response_rate'], x[1]['applications']), 
                             reverse=True)[:5]
        
        return {
            'total_positions': len(position_counts),
            'position_distribution': dict(position_counts.most_common(10)),
            'position_success_rates': position_rates,
            'top_performing_positions': top_positions,
            'focus_score': self._calculate_focus_score(position_counts)
        }
    
    def _analyze_status_flow(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Analyze status transition patterns.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Status flow analysis
        """
        if not applications:
            return {}
        
        # Status distribution
        status_counts = Counter(app.status for app in applications)
        
        # Days in each status
        status_days = defaultdict(list)
        for app in applications:
            days_since_applied = app.get_days_since_applied()
            status_days[app.status].append(days_since_applied)
        
        # Average days in each status
        avg_days_per_status = {}
        for status, days_list in status_days.items():
            if days_list:
                avg_days_per_status[status] = sum(days_list) / len(days_list)
        
        # Status progression rates
        total_apps = len(applications)
        progression_rates = {}
        
        for status in ApplicationStatus:
            count = status_counts.get(status, 0)
            progression_rates[status] = count / total_apps if total_apps > 0 else 0
        
        return {
            'status_distribution': dict(status_counts),
            'status_percentages': {
                (status.value if hasattr(status, 'value') else str(status)): (count / total_apps) * 100 
                for status, count in status_counts.items()
            },
            'average_days_per_status': {
                (status.value if hasattr(status, 'value') else str(status)): days 
                for status, days in avg_days_per_status.items()
            },
            'progression_rates': {
                (status.value if hasattr(status, 'value') else str(status)): rate 
                for status, rate in progression_rates.items()
            },
            'conversion_funnel': self._calculate_conversion_funnel(applications)
        }
    
    def _calculate_success_metrics(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Calculate success metrics and KPIs.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Success metrics
        """
        if not applications:
            return {}
        
        total_apps = len(applications)
        
        # Response metrics
        responded = len([app for app in applications if app.status in [
            ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER
        ]])
        
        interviewed = len([app for app in applications if app.status in [
            ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER
        ]])
        
        offers = len([app for app in applications if app.status == ApplicationStatus.OFFER])
        
        rejected = len([app for app in applications if app.status == ApplicationStatus.REJECTED])
        
        # Time metrics
        avg_response_time = self._calculate_avg_response_time(applications)
        
        # Quality metrics
        avg_confidence = sum(app.confidence_score for app in applications) / total_apps
        
        # Efficiency metrics
        applications_per_offer = total_apps / offers if offers > 0 else float('inf')
        
        return {
            'response_rate': responded / total_apps,
            'interview_rate': interviewed / total_apps,
            'offer_rate': offers / total_apps,
            'rejection_rate': rejected / total_apps,
            'average_response_time_days': avg_response_time,
            'average_confidence_score': avg_confidence,
            'applications_per_offer': applications_per_offer,
            'success_velocity': self._calculate_success_velocity(applications),
            'efficiency_score': self._calculate_efficiency_score(applications)
        }
    
    def _analyze_stale_applications(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Analyze stale applications that need follow-up.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: Stale applications analysis
        """
        if not applications:
            return {}
        
        # Define staleness thresholds
        thresholds = {
            'follow_up_needed': 14,  # 2 weeks
            'stale': 30,  # 1 month
            'very_stale': 60  # 2 months
        }
        
        now = datetime.now(timezone.utc)
        stale_categories = {
            'follow_up_needed': [],
            'stale': [],
            'very_stale': []
        }
        
        for app in applications:
            # Ensure timezone-aware comparison
            last_updated = app.last_updated
            if last_updated.tzinfo is None:
                last_updated = last_updated.replace(tzinfo=timezone.utc)
            days_since_update = (now - last_updated).days
            
            if days_since_update >= thresholds['very_stale']:
                stale_categories['very_stale'].append(app)
            elif days_since_update >= thresholds['stale']:
                stale_categories['stale'].append(app)
            elif days_since_update >= thresholds['follow_up_needed']:
                stale_categories['follow_up_needed'].append(app)
        
        # Stale applications by status
        stale_by_status = defaultdict(list)
        for category, apps in stale_categories.items():
            for app in apps:
                stale_by_status[app.status].append(app)
        
        return {
            'stale_counts': {category: len(apps) for category, apps in stale_categories.items()},
            'stale_applications': {
                category: [self._summarize_application(app) for app in apps[:5]]  # Top 5
                for category, apps in stale_categories.items()
            },
            'stale_by_status': {
                (status.value if hasattr(status, 'value') else str(status)): len(apps) for status, apps in stale_by_status.items()
            },
            'follow_up_recommendations': self._generate_follow_up_recommendations(stale_categories)
        }
    
    def _analyze_research_results(self, research_results: List[ResearchResult]) -> Dict[str, Any]:
        """
        Analyze research results for insights.
        
        Args:
            research_results (List[ResearchResult]): Research results
            
        Returns:
            Dict[str, Any]: Research insights
        """
        if not research_results:
            return {}
        
        # Market insights
        total_postings = sum(result.total_results for result in research_results)
        avg_relevance = sum(result.average_relevance for result in research_results) / len(research_results)
        
        # Best matches
        best_matches = []
        for result in research_results:
            if result.best_match:
                best_matches.append({
                    'company': result.target_company,
                    'position': result.target_position,
                    'best_match_title': result.best_match.title,
                    'best_match_company': result.best_match.company,
                    'relevance_score': result.best_match.relevance_score
                })
        
        return {
            'total_research_conducted': len(research_results),
            'total_postings_found': total_postings,
            'average_relevance': avg_relevance,
            'best_matches': best_matches[:10],  # Top 10
            'market_activity_score': self._calculate_market_activity_score(research_results)
        }
    
    async def generate_insights(self, applications: List[JobApplication]) -> Dict[str, Any]:
        """
        Generate AI-powered insights and recommendations.
        
        Args:
            applications (List[JobApplication]): Job applications
            
        Returns:
            Dict[str, Any]: AI-generated insights
        """
        try:
            if not applications:
                return {
                    'insights': 'No applications to analyze',
                    'recommendations': []
                }
            
            # Generate summary statistics
            summary = self.generate_status_summary(applications)
            
            # Prepare data for analysis
            status_dist = '\n'.join([f"- {(status.value if hasattr(status, 'value') else str(status))}: {count}" for status, count in summary.status_counts.items()])
            
            companies = Counter(app.company for app in applications)
            top_companies = '\n'.join([f"- {company}: {count}" for company, count in companies.most_common(5)])
            
            positions = Counter(app.position for app in applications)
            top_positions = '\n'.join([f"- {position}: {count}" for position, count in positions.most_common(5)])
            
            # Recent activity
            recent_apps = [app for app in applications if app.get_days_since_applied() <= 30]
            recent_activity = f"{len(recent_apps)} applications in the last 30 days"
            
            # Average days since applied
            avg_days_applied = sum(app.get_days_since_applied() for app in applications) / len(applications)
            
            # Create insights prompt
            prompt = self.insights_prompt_template.format(
                total_applications=summary.total_applications,
                response_rate=summary.response_rate,
                interview_rate=summary.interview_rate,
                offer_rate=summary.offer_rate,
                avg_days_applied=avg_days_applied,
                status_distribution=status_dist,
                top_companies=top_companies,
                top_positions=top_positions,
                recent_activity=recent_activity
            )
            
            # Get AI insights
            response = await call_llm(prompt)
            insights = parse_yaml_response(response)
            
            # Add timestamp and metadata
            insights['generated_at'] = datetime.now(timezone.utc)
            insights['applications_analyzed'] = len(applications)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {
                'insights': f'Failed to generate insights: {e}',
                'recommendations': []
            }
    
    # Helper methods
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _calculate_consistency_score(self, daily_counts: Dict) -> float:
        """Calculate consistency score for application frequency."""
        if not daily_counts:
            return 0.0
        
        values = list(daily_counts.values())
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = self._calculate_variance(values)
        
        # Consistency score: higher is more consistent
        return 1.0 / (1.0 + variance) if variance > 0 else 1.0
    
    def _calculate_diversification_score(self, company_counts: Counter) -> float:
        """Calculate diversification score (lower is more diversified)."""
        if not company_counts:
            return 0.0
        
        total = sum(company_counts.values())
        # Herfindahl-Hirschman Index
        hhi = sum((count / total) ** 2 for count in company_counts.values())
        return hhi
    
    def _calculate_concentration_risk(self, company_counts: Counter) -> float:
        """Calculate concentration risk (percentage of applications to top company)."""
        if not company_counts:
            return 0.0
        
        total = sum(company_counts.values())
        top_company_count = company_counts.most_common(1)[0][1]
        return top_company_count / total
    
    def _calculate_focus_score(self, position_counts: Counter) -> float:
        """Calculate focus score (concentration in specific positions)."""
        if not position_counts:
            return 0.0
        
        total = sum(position_counts.values())
        # Top 3 positions as percentage of total
        top_3_count = sum(count for _, count in position_counts.most_common(3))
        return top_3_count / total
    
    def _calculate_conversion_funnel(self, applications: List[JobApplication]) -> Dict[str, float]:
        """Calculate conversion funnel metrics."""
        if not applications:
            return {}
        
        total = len(applications)
        acknowledged = len([app for app in applications if app.status in [
            ApplicationStatus.ACKNOWLEDGED, ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER
        ]])
        interviewing = len([app for app in applications if app.status in [
            ApplicationStatus.INTERVIEWING, ApplicationStatus.OFFER
        ]])
        offers = len([app for app in applications if app.status == ApplicationStatus.OFFER])
        
        return {
            'applied_to_acknowledged': acknowledged / total if total > 0 else 0,
            'acknowledged_to_interviewing': interviewing / acknowledged if acknowledged > 0 else 0,
            'interviewing_to_offer': offers / interviewing if interviewing > 0 else 0
        }
    
    def _calculate_avg_response_time(self, applications: List[JobApplication]) -> float:
        """Calculate average response time."""
        response_times = []
        for app in applications:
            if app.status != ApplicationStatus.APPLIED:
                # Use last_updated as response time
                response_time = (app.last_updated - app.applied_date).days
                response_times.append(response_time)
        
        return sum(response_times) / len(response_times) if response_times else 0.0
    
    def _calculate_success_velocity(self, applications: List[JobApplication]) -> float:
        """Calculate success velocity (offers per unit time)."""
        if not applications:
            return 0.0
        
        offers = len([app for app in applications if app.status == ApplicationStatus.OFFER])
        
        # Calculate time span
        dates = [app.applied_date for app in applications]
        span_days = (max(dates) - min(dates)).days
        
        return offers / max(span_days, 1)  # Offers per day
    
    def _calculate_efficiency_score(self, applications: List[JobApplication]) -> float:
        """Calculate efficiency score."""
        if not applications:
            return 0.0
        
        # Weighted score based on status
        status_weights = {
            ApplicationStatus.APPLIED: 0.1,
            ApplicationStatus.ACKNOWLEDGED: 0.3,
            ApplicationStatus.INTERVIEWING: 0.7,
            ApplicationStatus.OFFER: 1.0,
            ApplicationStatus.REJECTED: 0.0,
            ApplicationStatus.WITHDRAWN: 0.0,
            ApplicationStatus.UNKNOWN: 0.05
        }
        
        total_score = sum(status_weights.get(app.status, 0) for app in applications)
        return total_score / len(applications)
    
    def _calculate_market_activity_score(self, research_results: List[ResearchResult]) -> float:
        """Calculate market activity score."""
        if not research_results:
            return 0.0
        
        # Score based on number of postings found
        total_postings = sum(result.total_results for result in research_results)
        avg_postings = total_postings / len(research_results)
        
        # Normalize to 0-1 scale (assuming 10 postings is high activity)
        return min(avg_postings / 10.0, 1.0)
    
    def _summarize_application(self, app: JobApplication) -> Dict[str, Any]:
        """Create summary of application for reports."""
        return {
            'company': app.company,
            'position': app.position,
            'status': (app.status.value if hasattr(app.status, 'value') else str(app.status)),
            'applied_date': app.applied_date,
            'days_since_applied': app.get_days_since_applied(),
            'days_since_updated': app.get_days_since_updated(),
            'confidence_score': app.confidence_score
        }
    
    def _generate_follow_up_recommendations(self, stale_categories: Dict[str, List[JobApplication]]) -> List[str]:
        """Generate follow-up recommendations."""
        recommendations = []
        
        if stale_categories['follow_up_needed']:
            recommendations.append(f"Follow up on {len(stale_categories['follow_up_needed'])} applications from 2 weeks ago")
        
        if stale_categories['stale']:
            recommendations.append(f"Consider status inquiries for {len(stale_categories['stale'])} applications over 1 month old")
        
        if stale_categories['very_stale']:
            recommendations.append(f"Archive or revisit {len(stale_categories['very_stale'])} applications over 2 months old")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get agent statistics.
        
        Returns:
            Dict[str, Any]: Agent statistics
        """
        return {
            'total_reports_generated': self.stats['total_reports_generated'],
            'applications_processed': self.stats['applications_processed'],
            'processing_time': self.stats['processing_time'],
            'last_run': self.stats['last_run'].isoformat() if self.stats['last_run'] else None,
            'error_count': len(self.stats['errors']),
            'recent_errors': self.stats['errors'][-5:] if self.stats['errors'] else []
        }
    
    def reset_statistics(self) -> None:
        """Reset agent statistics."""
        self.stats = {
            'total_reports_generated': 0,
            'applications_processed': 0,
            'processing_time': 0.0,
            'last_run': None,
            'errors': []
        }

# Factory function
def create_status_agent() -> StatusAgent:
    """
    Create and return Status Agent instance.
    
    Returns:
        StatusAgent: Configured Status Agent
    """
    return StatusAgent()

# Test function
async def test_status_agent():
    """Test Status Agent functionality."""
    try:
        agent = create_status_agent()
        
        # Create test applications
        from models.job_models import JobApplication, ApplicationStatus
        from datetime import datetime, timezone, timedelta
        
        test_apps = [
            JobApplication(
                email_id="test_1",
                company="Google",
                position="Software Engineer",
                status=ApplicationStatus.APPLIED,
                applied_date=datetime.now(timezone.utc) - timedelta(days=10),
                last_updated=datetime.now(timezone.utc) - timedelta(days=8),
                confidence_score=0.85
            ),
            JobApplication(
                email_id="test_2",
                company="Meta",
                position="Software Engineer",
                status=ApplicationStatus.INTERVIEWING,
                applied_date=datetime.now(timezone.utc) - timedelta(days=20),
                last_updated=datetime.now(timezone.utc) - timedelta(days=2),
                confidence_score=0.90
            ),
            JobApplication(
                email_id="test_3",
                company="Netflix",
                position="Senior Software Engineer",
                status=ApplicationStatus.REJECTED,
                applied_date=datetime.now(timezone.utc) - timedelta(days=30),
                last_updated=datetime.now(timezone.utc) - timedelta(days=25),
                confidence_score=0.75
            )
        ]
        
        print("✅ Test applications created")
        
        # Test status summary
        summary = agent.generate_status_summary(test_apps)
        print(f"✅ Status summary: {summary.total_applications} applications")
        print(f"📊 Response rate: {summary.response_rate:.1%}")
        print(f"📊 Interview rate: {summary.interview_rate:.1%}")
        
        # Test detailed report
        report = agent.generate_detailed_report(test_apps)
        print(f"✅ Detailed report generated with {len(report)} sections")
        
        # Test insights
        insights = await agent.generate_insights(test_apps)
        print(f"✅ AI insights generated: {insights.get('overall_assessment', 'N/A')}")
        
        # Test statistics
        stats = agent.get_statistics()
        print(f"📊 Agent stats: {stats['total_reports_generated']} reports generated")
        
    except Exception as e:
        print(f"❌ Status Agent test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_status_agent())