# app/ingestion/parser.py

from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import os

def read_pdf(file_path: str) -> str:
    try:
        text = extract_pdf_text(file_path)
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def read_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Error reading DOCX: {e}"

def read_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
