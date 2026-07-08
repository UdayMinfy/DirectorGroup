import base64
import logging
from io import BytesIO

from config import MAX_FILE_SIZE_MB, MAX_FILE_PAGES, ALLOWED_FILE_TYPES


LOGGER = logging.getLogger(__name__)


class FileExtractor:
    """Extract text content from PDF and DOCX files."""

    @staticmethod
    def extract_text(file_content_base64, file_type):
        """
        Extract text from base64-encoded file.
        
        Args:
            file_content_base64: Base64 encoded file content
            file_type: File type ('pdf' or 'docx')
            
        Returns:
            Extracted and cleaned text
            
        Raises:
            ValueError: If validation fails or extraction fails
        """
        # Validate file type
        if file_type not in ALLOWED_FILE_TYPES:
            raise ValueError(f"Unsupported file type. Only {', '.join(ALLOWED_FILE_TYPES)} are allowed.")
        
        try:
            # Decode base64
            file_bytes = base64.b64decode(file_content_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {e}")
        
        # Validate file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)")
        
        LOGGER.info("Processing %s file: %.2f MB", file_type, file_size_mb)
        
        # Extract text based on file type
        if file_type == "pdf":
            return FileExtractor._extract_from_pdf(file_bytes)
        elif file_type == "docx":
            return FileExtractor._extract_from_docx(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _extract_from_pdf(file_bytes):
        """Extract text from PDF file."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) library not installed")
        
        try:
            # Open PDF from bytes
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Check page count
            page_count = pdf_doc.page_count
            # Commented out: UI handles page validation, backend just processes
            # if page_count > MAX_FILE_PAGES:
            #     pdf_doc.close()
            #     raise ValueError(f"Document has {page_count} pages, exceeding maximum of {MAX_FILE_PAGES} pages")
            
            LOGGER.info("Extracting text from %d-page PDF", page_count)
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(page_count):
                page = pdf_doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            pdf_doc.close()
            
            if not text_parts:
                raise ValueError("No text content found in PDF")
            
            full_text = "\n\n".join(text_parts)
            cleaned_text = FileExtractor._clean_text(full_text)
            
            LOGGER.info("Extracted %d characters from PDF", len(cleaned_text))
            return cleaned_text
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            LOGGER.exception("Failed to extract text from PDF")
            raise ValueError(f"Failed to extract text from PDF: {e}")

    @staticmethod
    def _extract_from_docx(file_bytes):
        """Extract text from DOCX file."""
        try:
            from docx import Document
        except ImportError:
            raise RuntimeError("python-docx library not installed")
        
        try:
            # Open DOCX from bytes
            docx_file = BytesIO(file_bytes)
            doc = Document(docx_file)
            
            # Estimate page count (rough: 500 words per page)
            total_words = sum(len(para.text.split()) for para in doc.paragraphs)
            estimated_pages = max(1, total_words // 500)
            
            # Commented out: UI handles page validation, backend just processes
            # if estimated_pages > MAX_FILE_PAGES:
            #     raise ValueError(f"Document has approximately {estimated_pages} pages, exceeding maximum of {MAX_FILE_PAGES} pages")
            
            LOGGER.info("Extracting text from DOCX (estimated %d pages)", estimated_pages)
            
            # Extract paragraphs
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            
            # Extract tables
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        table_texts.append(row_text)
            
            if not paragraphs and not table_texts:
                raise ValueError("No text content found in DOCX")
            
            # Combine paragraphs and tables
            all_text = paragraphs + table_texts
            full_text = "\n\n".join(all_text)
            cleaned_text = FileExtractor._clean_text(full_text)
            
            LOGGER.info("Extracted %d characters from DOCX", len(cleaned_text))
            return cleaned_text
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            LOGGER.exception("Failed to extract text from DOCX")
            raise ValueError(f"Failed to extract text from DOCX: {e}")

    @staticmethod
    def _clean_text(text):
        """Clean extracted text."""
        import re
        
        # Remove multiple consecutive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple consecutive newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
