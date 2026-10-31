"""
RAG Pipeline Module
Coordinates PDF ingestion, vector indexing, query retrieval, and Groq LLM answer generation.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from groq import Groq

from app.pdf_processor import extract_text_from_pdf, split_text_into_chunks
from app.embedding import get_embedding_model
from app.vector_store import vector_store

# Load environment variables from .env file
load_dotenv()


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) workflow:
    1. PDF Text Extraction (PyPDF)
    2. Text Chunking (LangChain RecursiveCharacterTextSplitter)
    3. Vector Embeddings (Sentence Transformers)
    4. Indexing & Similarity Search (FAISS)
    5. Contextual Generation (Groq LLM)
    """

    def __init__(self, groq_model: str = "qwen/qwen3.8-27b"):
        self.groq_model = groq_model
        self.embedding_model = get_embedding_model()
        self.vector_store = vector_store

    def _get_groq_client(self) -> Groq:
        """
        Retrieves the Groq client instance using the GROQ_API_KEY environment variable.
        Raises ValueError if the key is unset or set to placeholder value.
        """
        # Dynamically reload .env so user edits take effect immediately
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.strip() in ("", "your_groq_api_key_here"):
            raise ValueError(
                "GROQ_API_KEY is missing or invalid in your .env file. "
                "Please get a free API key from https://console.groq.com/keys and configure .env."
            )
        return Groq(api_key=api_key.strip())

    def ingest_pdf(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Executes the ingestion pipeline for an uploaded PDF:
        - Extracts text
        - Chunks into 500-char pieces with 50-char overlap
        - Generates sentence transformer embeddings
        - Stores in FAISS
        """
        # 1. Extract text from PDF
        text = extract_text_from_pdf(file_path)

        # 2. Chunk text
        chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        if not chunks:
            raise ValueError("Document did not produce any valid text chunks.")

        # 3. Generate embeddings
        embeddings = self.embedding_model.generate_embeddings(chunks)

        # 4. Store in FAISS
        self.vector_store.build_index(
            chunks=chunks,
            embeddings=embeddings,
            document_name=original_filename
        )

        return {
            "filename": original_filename,
            "total_chunks": len(chunks),
            "character_count": len(text)
        }

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answers a user question based strictly on the uploaded document:
        - Embeds the question
        - Retrieves top 3 relevant chunks from FAISS
        - Prompts Groq LLM with the retrieved context
        - Returns answer and source references
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        if self.vector_store.is_empty():
            raise ValueError("No document has been uploaded yet. Please upload a PDF first.")

        # 1. Embed query and search FAISS
        query_embedding = self.embedding_model.generate_query_embedding(cleaned_question)
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)

        if not retrieved_chunks:
            return {
                "question": cleaned_question,
                "answer": "Sorry, the answer is not available in the uploaded document.",
                "sources": [],
                "document_name": self.vector_store.document_name
            }

        # 2. Format context from retrieved chunks
        formatted_context_list = []
        for i, item in enumerate(retrieved_chunks, start=1):
            formatted_context_list.append(f"[Excerpt {i}]:\n{item['chunk']}")
        context_str = "\n\n".join(formatted_context_list)

        # 3. System prompt enforcing RAG guardrails
        system_prompt = (
            "You are an AI assistant designed to answer questions about an uploaded PDF document.\n"
            "Strict Instructions:\n"
            "1. Base your answer ONLY on the provided context excerpts.\n"
            "2. If the context does not contain enough information to answer the question, "
            "respond exactly with: \"Sorry, the answer is not available in the uploaded document.\"\n"
            "3. Do not extrapolate, guess, or use any prior knowledge outside the provided context.\n"
            "4. Keep your answer concise, helpful, and truthful to the context."
        )

        user_prompt = (
            f"Document Context:\n{context_str}\n\n"
            f"Question: {cleaned_question}\n\n"
            f"Answer:"
        )

        # 4. Invoke Groq API with robust model fallback
        client = self._get_groq_client()
        candidate_models = [self.groq_model, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        # Remove duplicates preserving priority
        candidate_models = list(dict.fromkeys(candidate_models))

        raw_answer = None
        last_error = None

        for model_name in candidate_models:
            try:
                chat_completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=600
                )
                answer_candidate = chat_completion.choices[0].message.content
                if answer_candidate and answer_candidate.strip():
                    raw_answer = answer_candidate.strip()
                    self.groq_model = model_name
                    break
            except Exception as e:
                last_error = e
                continue

        if not raw_answer:
            if last_error:
                raise RuntimeError(f"Groq API error: {str(last_error)}")
            raw_answer = "Sorry, the answer is not available in the uploaded document."

        # 5. Extract sources
        sources = [
            {
                "rank": item["rank"],
                "text": item["chunk"],
                "score": round(item["score"], 4)
            }
            for item in retrieved_chunks
        ]

        return {
            "question": cleaned_question,
            "answer": raw_answer,
            "sources": sources,
            "document_name": self.vector_store.document_name
        }


# Global pipeline instance
rag_pipeline = RAGPipeline()
