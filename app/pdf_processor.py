"""
PDF Processor Module
Handles PDF text extraction using PyPDF and text chunking using LangChain Text Splitters.
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts raw text from an uploaded PDF file using PyPDF.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: Extracted text from all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid PDF or has no extractable text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF file. The file may be corrupt or encrypted. Details: {str(e)}")

    if len(reader.pages) == 0:
        raise ValueError("The uploaded PDF has 0 pages.")

    full_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text.strip())
        except Exception as e:
            # Continue reading other pages if a single page encounters an issue
            print(f"Warning: Failed to extract text from page {page_number}: {e}")

    combined_text = "\n\n".join(full_text).strip()

    if not combined_text:
        raise ValueError(
            "No extractable text found in the PDF. The document may consist solely of scanned images, "
            "which require OCR to read, or it might be completely blank."
        )

    return combined_text


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits continuous text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text (str): The raw text to split.
        chunk_size (int): Maximum size of each chunk (characters). Default is 500.
        chunk_overlap (int): Overlap between consecutive chunks (characters). Default is 50.

    Returns:
        list[str]: A list of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    # RecursiveCharacterTextSplitter attempts to split along paragraphs, then sentences, then words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    # Filter out empty or whitespace-only chunks
    cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return cleaned_chunks
