#!/usr/bin/env python3
"""
CLI interface for job application tracking system.

Provides command-line interface for running the job application tracking
workflow with rich formatting and interactive features.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Rich formatting imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.prompt import Prompt, IntPrompt, Confirm
    from rich.tree import Tree
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not available. Install with: pip install rich")

from flow import create_job_application_flow, run_job_application_tracking
from config.settings import get_settings
from utils import logger


class JobApplicationCLI:
    """
    Command-line interface for job application tracking.
    
    Provides interactive commands and rich formatting for the
    job application tracking workflow.
    """
    
    def __init__(self):
        """Initialize CLI with console and settings."""
        self.console = Console() if RICH_AVAILABLE else None
        self.settings = get_settings()
        
        # CLI configuration
        self.config = {
            'output_format': 'rich',  # rich, json, plain
            'show_progress': True,
            'interactive': True,
            'verbose': False
        }
    
    def print_banner(self) -> None:
        """Print application banner."""
        if not RICH_AVAILABLE:
            print("=== Job Application Tracker ===")
            print("Track and analyze your job applications with AI-powered insights")
            return
        
        banner = Panel(
            Align.center(
                "[bold blue]Job Application Tracker[/bold blue]\n"
                "[dim]Track and analyze your job applications with AI-powered insights[/dim]"
            ),
            style="blue",
            padding=(1, 2)
        )
        self.console.print(banner)
    
    def print_usage(self) -> None:
        """Print usage information."""
        usage_text = """
        [bold cyan]Usage:[/bold cyan]
        
        [green]python cli.py track[/green]           - Run job application tracking
        [green]python cli.py track --days 30[/green] - Track last 30 days
        [green]python cli.py status[/green]          - Show system status
        [green]python cli.py setup[/green]           - Interactive setup
        [green]python cli.py config[/green]          - Show configuration
        [green]python cli.py help[/green]            - Show detailed help
        
        [bold cyan]Options:[/bold cyan]
        
        --days, -d       Number of days back to retrieve emails (default: 30)
        --format, -f     Output format: rich, json, plain (default: rich)
        --output, -o     Output file path (optional)
        --verbose, -v    Enable verbose logging
        --no-progress    Disable progress indicators
        --quiet, -q      Quiet mode (minimal output)
        """
        
        if RICH_AVAILABLE:
            self.console.print(Panel(usage_text, title="Usage", style="cyan"))
        else:
            print(usage_text)
    
    async def run_tracking(self, days_back: int = None, output_file: str = None) -> Dict[str, Any]:
        """
        Run job application tracking workflow.
        
        Args:
            days_back (int, optional): Number of days back to retrieve emails
            output_file (str, optional): Output file path
            
        Returns:
            Dict[str, Any]: Workflow results
        """
        if not RICH_AVAILABLE:
            print(f"Starting job application tracking for last {days_back or self.settings.email_days_back} days...")
            result = await run_job_application_tracking(days_back)
            print("Tracking completed!")
            return result
        
        # Rich progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        ) as progress:
            
            # Create progress tasks
            main_task = progress.add_task("Initializing...", total=100)
            
            # Start workflow
            flow = create_job_application_flow()
            
            # Phase 1: Email Processing
            progress.update(main_task, description="Processing emails...", completed=10)
            
            # Phase 2: Classification
            progress.update(main_task, description="Classifying applications...", completed=30)
            
            # Phase 3: Research
            progress.update(main_task, description="Researching job postings...", completed=50)
            
            # Phase 4: Analysis
            progress.update(main_task, description="Analyzing results...", completed=70)
            
            # Phase 5: Reporting
            progress.update(main_task, description="Generating reports...", completed=90)
            
            # Run the actual workflow
            result = await flow.run_full_workflow(days_back)
            
            progress.update(main_task, description="Completed!", completed=100)
        
        # Display results
        await self._display_results(result, output_file)
        
        return result
    
    async def _display_results(self, result: Dict[str, Any], output_file: str = None) -> None:
        """Display workflow results with rich formatting."""
        if not result.get('success'):
            self._display_error(result)
            return
        
        # Save to file if requested
        if output_file:
            await self._save_results(result, output_file)
        
        if not RICH_AVAILABLE:
            self._display_plain_results(result)
            return
        
        # Rich formatted display
        self._display_rich_results(result)
    
    def _display_rich_results(self, result: Dict[str, Any]) -> None:
        """Display results with rich formatting."""
        summary = result.get('summary', {})
        results = result.get('results', {})
        
        # Summary panel
        summary_text = f"""
        [green]✅ Workflow completed successfully![/green]
        
        [bold]Email Processing:[/bold]
        • Total emails retrieved: {summary.get('total_emails_retrieved', 0)}
        • Job-related emails: {summary.get('job_related_emails', 0)}
        • Applications created: {summary.get('job_applications_created', 0)}
        
        [bold]Analysis:[/bold]
        • Research completed: {'✅' if summary.get('research_completed') else '❌'}
        • Final report generated: {'✅' if summary.get('final_report_generated') else '❌'}
        
        [bold]Execution Time:[/bold] {result.get('execution_time', 0):.2f} seconds
        """
        
        self.console.print(Panel(summary_text, title="Summary", style="green"))
        
        # Applications table
        if results.get('job_applications'):
            self._display_applications_table(results['job_applications'])
        
        # Status report
        if results.get('status_report'):
            self._display_status_report(results['status_report'])
        
        # AI insights
        if results.get('ai_insights'):
            self._display_ai_insights(results['ai_insights'])
    
    def _display_applications_table(self, applications: List[Dict[str, Any]]) -> None:
        """Display applications in a table format."""
        table = Table(title="Job Applications", show_header=True, header_style="bold magenta")
        
        table.add_column("Company", style="cyan")
        table.add_column("Position", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Applied", style="blue")
        table.add_column("Confidence", style="red")
        
        for app in applications[:10]:  # Show first 10
            # Handle both dict and object formats
            if isinstance(app, dict):
                company = app.get('company', 'N/A')
                position = app.get('position', 'N/A')
                status = app.get('status', 'N/A')
                applied_date = app.get('applied_date', 'N/A')
                confidence = app.get('confidence_score', 0)
            else:
                company = getattr(app, 'company', 'N/A')
                position = getattr(app, 'position', 'N/A')
                status = getattr(app, 'status', 'N/A')
                applied_date = getattr(app, 'applied_date', 'N/A')
                confidence = getattr(app, 'confidence_score', 0)
            
            # Format date
            if hasattr(applied_date, 'strftime'):
                applied_str = applied_date.strftime('%Y-%m-%d')
            else:
                applied_str = str(applied_date)
            
            # Format confidence
            confidence_str = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else str(confidence)
            
            table.add_row(
                company,
                position,
                status,
                applied_str,
                confidence_str
            )
        
        self.console.print(table)
    
    def _display_status_report(self, status_report: Dict[str, Any]) -> None:
        """Display status report."""
        summary = status_report.get('summary', {})
        
        status_text = f"""
        [bold]Application Summary:[/bold]
        • Total Applications: {getattr(summary, 'total_applications', 0)}
        • Response Rate: {getattr(summary, 'response_rate', 0):.1%}
        • Interview Rate: {getattr(summary, 'interview_rate', 0):.1%}
        • Offer Rate: {getattr(summary, 'offer_rate', 0):.1%}
        
        [bold]Recent Activity:[/bold]
        • Applications in last 30 days: {status_report.get('time_analysis', {}).get('recent_activity', {}).get('last_30_days', 0)}
        • Average per week: {status_report.get('time_analysis', {}).get('frequency', {}).get('avg_per_week', 0):.1f}
        """
        
        self.console.print(Panel(status_text, title="Status Report", style="blue"))
    
    def _display_ai_insights(self, insights: Dict[str, Any]) -> None:
        """Display AI-generated insights."""
        insights_text = f"""
        [bold green]Overall Assessment:[/bold green]
        {insights.get('overall_assessment', 'No assessment available')}
        
        [bold yellow]Strengths:[/bold yellow]
        """
        
        for strength in insights.get('strengths', []):
            insights_text += f"• {strength}\n"
        
        insights_text += f"""
        [bold red]Areas for Improvement:[/bold red]
        """
        
        for area in insights.get('areas_for_improvement', []):
            insights_text += f"• {area}\n"
        
        insights_text += f"""
        [bold blue]Recommendations:[/bold blue]
        """
        
        for rec in insights.get('specific_recommendations', []):
            insights_text += f"• {rec}\n"
        
        self.console.print(Panel(insights_text, title="AI Insights", style="magenta"))
    
    def _display_plain_results(self, result: Dict[str, Any]) -> None:
        """Display results in plain text format."""
        summary = result.get('summary', {})
        print("\n=== Job Application Tracking Results ===")
        print(f"Total emails retrieved: {summary.get('total_emails_retrieved', 0)}")
        print(f"Job-related emails: {summary.get('job_related_emails', 0)}")
        print(f"Applications created: {summary.get('job_applications_created', 0)}")
        print(f"Execution time: {result.get('execution_time', 0):.2f} seconds")
        
        # Show first few applications
        applications = result.get('results', {}).get('job_applications', [])
        if applications:
            print(f"\nFirst {min(5, len(applications))} applications:")
            for i, app in enumerate(applications[:5]):
                if isinstance(app, dict):
                    company = app.get('company', 'N/A')
                    position = app.get('position', 'N/A')
                    status = app.get('status', 'N/A')
                else:
                    company = getattr(app, 'company', 'N/A')
                    position = getattr(app, 'position', 'N/A')
                    status = getattr(app, 'status', 'N/A')
                
                print(f"  {i+1}. {company} - {position} ({status})")
    
    def _display_error(self, result: Dict[str, Any]) -> None:
        """Display error information."""
        error_msg = result.get('error', 'Unknown error')
        error_details = result.get('error_details', {})
        
        if RICH_AVAILABLE:
            error_text = f"""
            [bold red]❌ Workflow failed![/bold red]
            
            [bold]Error:[/bold] {error_msg}
            
            [bold]Details:[/bold]
            {json.dumps(error_details, indent=2)}
            
            [bold]Partial Results:[/bold]
            • Emails retrieved: {result.get('partial_results', {}).get('total_emails_retrieved', 0)}
            • Job-related emails: {result.get('partial_results', {}).get('job_related_emails', 0)}
            • Applications created: {result.get('partial_results', {}).get('job_applications_created', 0)}
            """
            
            self.console.print(Panel(error_text, title="Error", style="red"))
        else:
            print(f"\n❌ Workflow failed: {error_msg}")
            print(f"Details: {error_details}")
    
    async def _save_results(self, result: Dict[str, Any], output_file: str) -> None:
        """Save results to file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            if RICH_AVAILABLE:
                self.console.print(f"[green]Results saved to: {output_file}[/green]")
            else:
                print(f"Results saved to: {output_file}")
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"[red]Failed to save results: {e}[/red]")
            else:
                print(f"Failed to save results: {e}")
    
    def show_status(self) -> None:
        """Show system status and configuration."""
        if not RICH_AVAILABLE:
            print("=== System Status ===")
            print(f"Gmail credentials: {'✅' if os.path.exists(self.settings.gmail_credentials_path) else '❌'}")
            print(f"Brave API key: {'✅' if self.settings.brave_api_key else '❌'}")
            print(f"OpenAI API key: {'✅' if self.settings.openai_api_key else '❌'}")
            return
        
        # Check configuration
        gmail_status = "✅ Configured" if os.path.exists(self.settings.gmail_credentials_path) else "❌ Missing"
        brave_status = "✅ Configured" if self.settings.brave_api_key else "❌ Missing"
        openai_status = "✅ Configured" if self.settings.openai_api_key else "❌ Missing"
        
        status_text = f"""
        [bold]API Configuration:[/bold]
        • Gmail credentials: {gmail_status}
        • Brave API key: {brave_status}
        • OpenAI API key: {openai_status}
        
        [bold]Settings:[/bold]
        • Email days back: {self.settings.email_days_back}
        • Gmail rate limit delay: {self.settings.gmail_rate_limit_delay}
        • Classification threshold: {self.settings.classification_confidence_threshold}
        • Research max results: {self.settings.research_max_results}
        """
        
        self.console.print(Panel(status_text, title="System Status", style="cyan"))
    
    def run_setup(self) -> None:
        """Run interactive setup."""
        if not RICH_AVAILABLE:
            print("=== Interactive Setup ===")
            print("Please configure the following settings:")
            return
        
        self.console.print(Panel("Welcome to Job Application Tracker Setup", style="green"))
        
        # Gmail setup
        if not os.path.exists(self.settings.gmail_credentials_path):
            self.console.print("\n[yellow]Gmail credentials not found.[/yellow]")
            self.console.print("Please follow these steps:")
            self.console.print("1. Go to Google Cloud Console")
            self.console.print("2. Create a new project or select existing")
            self.console.print("3. Enable Gmail API")
            self.console.print("4. Create credentials (OAuth 2.0)")
            self.console.print("5. Download credentials.json")
            self.console.print(f"6. Place it at: {self.settings.gmail_credentials_path}")
        
        # API keys setup
        if not self.settings.brave_api_key:
            self.console.print("\n[yellow]Brave API key not configured.[/yellow]")
            self.console.print("Set BRAVE_API_KEY environment variable")
        
        if not self.settings.openai_api_key:
            self.console.print("\n[yellow]OpenAI API key not configured.[/yellow]")
            self.console.print("Set OPENAI_API_KEY environment variable")
        
        self.console.print("\n[green]Setup complete! Run 'python cli.py status' to verify configuration.[/green]")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Job Application Tracker CLI")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Run job application tracking')
    track_parser.add_argument('--days', '-d', type=int, default=None, help='Number of days back to retrieve emails')
    track_parser.add_argument('--output', '-o', type=str, help='Output file path')
    track_parser.add_argument('--format', '-f', choices=['rich', 'json', 'plain'], default='rich', help='Output format')
    track_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    track_parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    
    # Other commands
    subparsers.add_parser('status', help='Show system status')
    subparsers.add_parser('setup', help='Run interactive setup')
    subparsers.add_parser('config', help='Show configuration')
    subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = JobApplicationCLI()
    
    # Handle commands
    if args.command == 'track':
        cli.print_banner()
        await cli.run_tracking(args.days, args.output)
    elif args.command == 'status':
        cli.show_status()
    elif args.command == 'setup':
        cli.run_setup()
    elif args.command == 'config':
        cli.show_status()
    elif args.command == 'help':
        cli.print_usage()
    else:
        cli.print_banner()
        cli.print_usage()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)