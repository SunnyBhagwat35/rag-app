import os
import io
import json
import hashlib

from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Comprehensive document processor for extracting text from various formats
    """
    
    SUPPORTED_FORMATS = {
        '.pdf': 'load_pdf',
        '.txt': 'load_text',
    }
            
    def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Get the extraction method
            method_name = self.SUPPORTED_FORMATS[ext]
            method = getattr(self, method_name)
            loaded_doc = method(file_path)
            
            metadata = self.get_file_metadata(file_path)
            
            return {
                'text': loaded_doc,
                'metadata': metadata,
                'file_type': ext,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'text': '',
                'metadata': {},
                'file_type': ext if 'ext' in locals() else 'unknown',
                'success': False,
                'error': str(e)
            }
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract file metadata"""
        try:
            stat = os.stat(file_path)
            with open(file_path, 'rb') as f:
                content = f.read()
                
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_size': stat.st_size,
                'char_count': len(content.decode('utf-8', errors='ignore'))
            }
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            return {}
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print("Done loadding")
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")            
            return None
    
    def load_text(self, file_path: str) -> str:
        """Extract text from plain text files"""
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            return docs                
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None
        

