"""
File processing service with support for multiple formats.
Strategy pattern for safe, binary-aware text extraction.
"""

import logging
from io import BytesIO
from typing import Tuple
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class FileProcessor:
    """Base class for file processors."""
    
    def can_process(self, filename: str) -> bool:
        raise NotImplementedError
    
    def extract_text(self, file_bytes: bytes) -> str:
        raise NotImplementedError


class TextFileProcessor(FileProcessor):
    """Handle .txt and .md files."""
    
    EXTENSIONS = {'.txt', '.md', '.markdown'}
    
    def can_process(self, filename: str) -> bool:
        ext = filename.lower().split('.')[-1]
        return f'.{ext}' in self.EXTENSIONS
    
    def extract_text(self, file_bytes: bytes) -> str:
        """Extract text with UTF-8 and latin-1 fallback."""
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                logger.warning("UTF-8 decode failed, falling back to latin-1")
                return file_bytes.decode('latin-1')
            except UnicodeDecodeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Cannot decode file as text: {str(e)}"
                )


class PDFFileProcessor(FileProcessor):
    """Handle .pdf files using pdfplumber."""
    
    EXTENSIONS = {'.pdf'}
    
    def can_process(self, filename: str) -> bool:
        ext = filename.lower().split('.')[-1]
        return f'.{ext}' in self.EXTENSIONS
    
    def extract_text(self, file_bytes: bytes) -> str:
        """Extract text from PDF."""
        try:
            import pdfplumber
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF support not installed"
            )
        
        try:
            text_parts = []
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            if not text_parts:
                raise ValueError("No text found in PDF")
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )


class DocxFileProcessor(FileProcessor):
    """Handle .docx files using python-docx."""
    
    EXTENSIONS = {'.docx'}
    
    def can_process(self, filename: str) -> bool:
        ext = filename.lower().split('.')[-1]
        return f'.{ext}' in self.EXTENSIONS
    
    def extract_text(self, file_bytes: bytes) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DOCX support not installed"
            )
        
        try:
            doc = Document(BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            if not paragraphs:
                raise ValueError("No text found in DOCX")
            
            return "\n\n".join(paragraphs)
        
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text from DOCX: {str(e)}"
            )


class FileProcessingService:
    """Orchestrate file processing with strategy pattern."""
    
    def __init__(self):
        self.processors = [
            TextFileProcessor(),
            PDFFileProcessor(),
            DocxFileProcessor(),
        ]
    
    def extract_text(self, filename: str, file_bytes: bytes) -> str:
        """
        Extract text from any supported file format.
        
        Args:
            filename: Name of uploaded file
            file_bytes: Raw file bytes
        
        Returns:
            Extracted text
        
        Raises:
            HTTPException 400: Unsupported file type or extraction error
        """
        # Find matching processor
        processor = None
        for p in self.processors:
            if p.can_process(filename):
                processor = p
                break
        
        if processor is None:
            supported = ", ".join([".txt", ".md", ".pdf", ".docx"])
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type. Supported: {supported}"
            )
        
        logger.info(f"Processing {filename} with {processor.__class__.__name__}")
        return processor.extract_text(file_bytes)
