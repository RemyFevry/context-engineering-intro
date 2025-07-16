"""
Gmail API integration tool for job application tracking.

Handles OAuth2 authentication, email retrieval, and email parsing
with proper rate limiting and error handling.
"""

import asyncio
import base64
import email
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import get_settings
from models.email_models import GmailEmail, EmailFormat
from utils import logger, format_datetime, clean_text, truncate_text

class GmailTool:
    """
    Gmail API integration tool.
    
    Handles OAuth2 authentication, email retrieval, and parsing
    with proper rate limiting and error handling.
    """
    
    def __init__(self):
        """Initialize Gmail tool with settings."""
        self.settings = get_settings()
        self.service = None
        self.credentials = None
        
    def _get_credentials(self) -> Credentials:
        """
        Get OAuth2 credentials for Gmail API.
        
        Returns:
            Credentials: Valid OAuth2 credentials
            
        Raises:
            Exception: If authentication fails
        """
        creds = None
        token_path = self.settings.get_gmail_token_path()
        credentials_path = self.settings.get_gmail_credentials_path()
        
        # Load existing token if available
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(
                    str(token_path), 
                    self.settings.gmail_scopes
                )
                logger.info("Loaded existing Gmail credentials")
            except Exception as e:
                logger.warning(f"Failed to load existing token: {e}")
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Refreshed Gmail credentials")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {e}")
                    creds = None
            
            # Run OAuth flow if needed
            if not creds or not creds.valid:
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"Gmail credentials file not found: {credentials_path}. "
                        "Please download credentials.json from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), 
                    self.settings.gmail_scopes
                )
                creds = flow.run_local_server(port=0)
                logger.info("Completed OAuth2 flow")
            
            # Save credentials for next run
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                logger.info(f"Saved credentials to {token_path}")
        
        return creds
    
    def _build_service(self):
        """
        Build Gmail service instance.
        
        Raises:
            Exception: If service creation fails
        """
        try:
            self.credentials = self._get_credentials()
            self.service = build('gmail', 'v1', credentials=self.credentials)
            logger.info("Gmail service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gmail service: {e}")
            raise
    
    def _parse_email_headers(self, headers: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Parse email headers into dictionary.
        
        Args:
            headers (List[Dict[str, str]]): Raw email headers
            
        Returns:
            Dict[str, str]: Parsed headers
        """
        parsed_headers = {}
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            parsed_headers[name] = value
        return parsed_headers
    
    def _extract_email_body(self, payload: Dict[str, Any]) -> tuple[str, EmailFormat]:
        """
        Extract email body content and format.
        
        Args:
            payload (Dict[str, Any]): Email payload
            
        Returns:
            tuple[str, EmailFormat]: Body content and format
        """
        def decode_body(data: str) -> str:
            """Decode base64 email body."""
            if not data:
                return ""
            try:
                # Gmail uses base64url encoding
                decoded = base64.urlsafe_b64decode(data + '==').decode('utf-8')
                return decoded
            except Exception as e:
                logger.warning(f"Failed to decode email body: {e}")
                return ""
        
        mime_type = payload.get('mimeType', '')
        
        # Handle plain text
        if mime_type == 'text/plain':
            body_data = payload.get('body', {}).get('data', '')
            return decode_body(body_data), EmailFormat.PLAIN
        
        # Handle HTML
        elif mime_type == 'text/html':
            body_data = payload.get('body', {}).get('data', '')
            return decode_body(body_data), EmailFormat.HTML
        
        # Handle multipart
        elif mime_type.startswith('multipart/'):
            parts = payload.get('parts', [])
            if not parts:
                return "", EmailFormat.MULTIPART
            
            # Try to find text/plain first, then text/html
            for part in parts:
                part_mime = part.get('mimeType', '')
                if part_mime == 'text/plain':
                    body_data = part.get('body', {}).get('data', '')
                    return decode_body(body_data), EmailFormat.PLAIN
            
            # If no plain text, try HTML
            for part in parts:
                part_mime = part.get('mimeType', '')
                if part_mime == 'text/html':
                    body_data = part.get('body', {}).get('data', '')
                    return decode_body(body_data), EmailFormat.HTML
            
            # If multipart has nested parts, recurse
            for part in parts:
                if part.get('parts'):
                    body, format_type = self._extract_email_body(part)
                    if body:
                        return body, format_type
            
            return "", EmailFormat.MULTIPART
        
        # Unknown format
        else:
            logger.warning(f"Unknown email format: {mime_type}")
            return "", EmailFormat.PLAIN
    
    def _parse_email_message(self, message: Dict[str, Any]) -> GmailEmail:
        """
        Parse Gmail message into GmailEmail model.
        
        Args:
            message (Dict[str, Any]): Raw Gmail message
            
        Returns:
            GmailEmail: Parsed email model
        """
        msg_id = message.get('id', '')
        thread_id = message.get('threadId', '')
        
        payload = message.get('payload', {})
        headers = payload.get('headers', [])
        parsed_headers = self._parse_email_headers(headers)
        
        # Extract key headers
        sender = parsed_headers.get('from', '')
        subject = parsed_headers.get('subject', '')
        date_str = parsed_headers.get('date', '')
        
        # Parse date
        try:
            # Gmail date format: "Wed, 1 Jan 2025 12:00:00 +0000"
            received_date = email.utils.parsedate_to_datetime(date_str)
            if received_date.tzinfo is None:
                received_date = received_date.replace(tzinfo=timezone.utc)
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_str}': {e}")
            received_date = datetime.now(timezone.utc)
        
        # Extract body
        body, email_format = self._extract_email_body(payload)
        
        # Clean and truncate body
        body = clean_text(body)
        body = truncate_text(body, 5000)  # Keep more content for analysis
        
        # Extract labels
        labels = message.get('labelIds', [])
        
        return GmailEmail(
            id=msg_id,
            thread_id=thread_id,
            sender=sender,
            subject=subject,
            body=body,
            received_date=received_date,
            labels=labels,
            format=email_format,
            raw_headers=parsed_headers
        )
    
    async def get_recent_emails(self, days_back: int = None) -> List[GmailEmail]:
        """
        Get recent emails from Gmail.
        
        Args:
            days_back (int, optional): Days back to retrieve. Defaults to settings.
            
        Returns:
            List[GmailEmail]: List of parsed emails
            
        Raises:
            Exception: If email retrieval fails
        """
        if not self.service:
            self._build_service()
        
        days_back = days_back or self.settings.email_days_back
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # Gmail query format: YYYY/MM/DD
        query = f"after:{start_date.strftime('%Y/%m/%d')}"
        
        logger.info(f"Retrieving emails from {format_datetime(start_date)} to {format_datetime(end_date)}")
        
        try:
            all_emails = []
            next_page_token = None
            
            while True:
                # Get message IDs
                results = self.service.users().messages().list(
                    userId='me',
                    q=query,
                    pageToken=next_page_token,
                    maxResults=100
                ).execute()
                
                messages = results.get('messages', [])
                if not messages:
                    break
                
                logger.info(f"Processing {len(messages)} emails...")
                
                # Get full message details
                for msg in messages:
                    try:
                        # Rate limiting
                        await asyncio.sleep(self.settings.gmail_rate_limit_delay)
                        
                        # Get full message
                        full_message = self.service.users().messages().get(
                            userId='me',
                            id=msg['id'],
                            format='full'
                        ).execute()
                        
                        # Parse email
                        email_obj = self._parse_email_message(full_message)
                        all_emails.append(email_obj)
                        
                    except Exception as e:
                        logger.error(f"Failed to process email {msg['id']}: {e}")
                        continue
                
                # Check for next page
                next_page_token = results.get('nextPageToken')
                if not next_page_token:
                    break
                
                logger.info(f"Retrieved {len(all_emails)} emails so far...")
            
            logger.info(f"Successfully retrieved {len(all_emails)} emails")
            return all_emails
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error retrieving emails: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test Gmail API connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            if not self.service:
                self._build_service()
            
            # Test with a simple profile request
            profile = self.service.users().getProfile(userId='me').execute()
            email_address = profile.get('emailAddress', '')
            
            logger.info(f"Successfully connected to Gmail: {email_address}")
            return True
            
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return False
    
    async def search_emails(self, query: str, max_results: int = 100) -> List[GmailEmail]:
        """
        Search emails with custom query.
        
        Args:
            query (str): Gmail search query
            max_results (int): Maximum results to return
            
        Returns:
            List[GmailEmail]: Matching emails
        """
        if not self.service:
            self._build_service()
        
        try:
            logger.info(f"Searching emails with query: {query}")
            
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                try:
                    await asyncio.sleep(self.settings.gmail_rate_limit_delay)
                    
                    full_message = self.service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    
                    email_obj = self._parse_email_message(full_message)
                    emails.append(email_obj)
                    
                except Exception as e:
                    logger.error(f"Failed to process search result {msg['id']}: {e}")
                    continue
            
            logger.info(f"Search returned {len(emails)} emails")
            return emails
            
        except Exception as e:
            logger.error(f"Email search failed: {e}")
            return []

# Factory function for easy instantiation
def create_gmail_tool() -> GmailTool:
    """
    Create and return Gmail tool instance.
    
    Returns:
        GmailTool: Configured Gmail tool
    """
    return GmailTool()

# Test function
async def test_gmail_tool():
    """Test Gmail tool functionality."""
    try:
        tool = create_gmail_tool()
        
        # Test connection
        success = await tool.test_connection()
        if not success:
            print("❌ Gmail connection failed")
            return
        
        print("✅ Gmail connection successful")
        
        # Test email retrieval (small batch)
        emails = await tool.get_recent_emails(days_back=7)
        print(f"✅ Retrieved {len(emails)} emails from last 7 days")
        
        if emails:
            sample_email = emails[0]
            print(f"📧 Sample email: {sample_email.subject[:50]}...")
            print(f"📧 From: {sample_email.sender}")
            print(f"📧 Date: {format_datetime(sample_email.received_date)}")
            print(f"📧 Format: {sample_email.format}")
            print(f"📧 Body preview: {sample_email.body[:100]}...")
        
    except Exception as e:
        print(f"❌ Gmail tool test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_gmail_tool())