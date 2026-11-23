"""
PDF text extraction utilities
"""
import PyPDF2
import pdfplumber
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text from PDF files"""
    
    @staticmethod
    def extract_with_pypdf2(pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    @staticmethod
    def extract_with_pdfplumber(pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, pdf_path: str, method: str = "auto") -> str:
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_path: Path to PDF file
            method: 'pypdf2', 'pdfplumber', or 'auto' (tries both)
        
        Returns:
            Extracted text string
        """
        if method == "pypdf2":
            return cls.extract_with_pypdf2(pdf_path)
        elif method == "pdfplumber":
            return cls.extract_with_pdfplumber(pdf_path)
        else:  # auto
            # Try pdfplumber first (generally better)
            text = cls.extract_with_pdfplumber(pdf_path)
            if not text or len(text) < 100:
                # Fallback to PyPDF2
                text = cls.extract_with_pypdf2(pdf_path)
            return text


class TextCleaner:
    """Clean and normalize extracted text"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,;:()\-@]', '', text)
        
        # Remove multiple consecutive periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        
        return text.strip()
    
    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        """Extract email address from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else None
    
    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        """Extract phone number from text"""
        # Various phone formats
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # +1-234-567-8900
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (234) 567-8900
            r'\d{10}',  # 2345678900
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """Extract candidate name (usually first line or first capitalized words)"""
        # Simple heuristic: first line that looks like a name
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            # Name is usually 2-4 words, all capitalized or title case
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(word[0].isupper() for word in words if word):
                    return line
        
        # Fallback: first line
        return lines[0].strip() if lines else "Unknown"
    
    @classmethod
    def extract_metadata(cls, text: str) -> dict:
        """Extract metadata from CV text"""
        return {
            "name": cls.extract_name(text),
            "email": cls.extract_email(text),
            "phone": cls.extract_phone(text)
        }
