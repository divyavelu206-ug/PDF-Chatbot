"""
FastAPI Main Application
Serves the web UI and REST API endpoints for PDF upload and RAG querying.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from app.rag_pipeline import rag_pipeline
from app.vector_store import vector_store

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure uploads directory exists
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI application
app = FastAPI(
    title="PDF Chatbot – RAG with FastAPI",
    description="A beginner-friendly Retrieval-Augmented Generation chatbot using FastAPI, PyPDF, FAISS, and Groq.",
    version="1.0.0"
)

# Mount static assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# Pydantic Request & Response Models
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    message: str
    indexed_document: Optional[str] = None
    total_chunks: int = 0


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the uploaded PDF content")


class SourceItem(BaseModel):
    rank: int
    text: str
    score: float


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    document_name: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str
    total_chunks: int
    character_count: int


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_class=FileResponse, summary="Serve the Frontend Web Page")
async def get_index():
    """
    Renders and serves the HTML frontend interface.
    """
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def get_health():
    """
    Checks backend server status and reports whether a document is currently indexed.
    """
    doc_name = vector_store.document_name if not vector_store.is_empty() else None
    return HealthResponse(
        status="ok",
        message="Backend is operational.",
        indexed_document=doc_name,
        total_chunks=vector_store.get_total_chunks()
    )


@app.post("/upload", response_model=UploadResponse, summary="Upload and Process PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF document, extracts its text, splits it into chunks (500 chars, 50 overlap),
    generates embeddings with Sentence Transformers, and indexes them in FAISS.
    """
    # 1. Validate file extension
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a valid PDF (.pdf) file."
        )

    # 2. Save the uploaded file to disk
    sanitized_filename = os.path.basename(filename)
    saved_file_path = os.path.join(UPLOADS_DIR, sanitized_filename)

    try:
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    finally:
        await file.close()

    # 3. Check for empty file
    if os.path.getsize(saved_file_path) == 0:
        os.remove(saved_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty (0 bytes)."
        )

    # 4. Ingest PDF through RAG pipeline
    try:
        result = rag_pipeline.ingest_pdf(saved_file_path, original_filename=sanitized_filename)
        return UploadResponse(
            status="success",
            filename=result["filename"],
            message=f"Document '{result['filename']}' successfully processed and indexed into FAISS.",
            total_chunks=result["total_chunks"],
            character_count=result["character_count"]
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the PDF: {str(e)}"
        )


@app.post("/ask", response_model=QuestionResponse, summary="Ask a Question")
async def ask_question(payload: QuestionRequest):
    """
    Accepts a question, retrieves top 3 relevant chunks from FAISS,
    and sends the context to Groq to generate a strictly document-based answer.
    """
    try:
        response = rag_pipeline.generate_answer(payload.question, top_k=3)
        return QuestionResponse(
            question=response["question"],
            answer=response["answer"],
            sources=[
                SourceItem(rank=s["rank"], text=s["text"], score=s["score"])
                for s in response["sources"]
            ],
            document_name=response["document_name"]
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(re)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while generating the answer: {str(e)}"
        )
