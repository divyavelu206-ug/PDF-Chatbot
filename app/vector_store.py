"""
Vector Store Module
Manages the FAISS (Facebook AI Similarity Search) index for indexing and retrieving chunks.
"""

from typing import List, Dict, Any
import numpy as np
import faiss


class FAISSVectorStore:
    """
    In-memory vector store using FAISS IndexFlatL2 for exact L2 nearest neighbor search.
    Stores text chunks mapped to the vector indices.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.chunks: List[str] = []
        self.document_name: str = ""
        self._initialize_index()

    def _initialize_index(self):
        """Initializes a blank FAISS IndexFlatL2."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.document_name = ""

    def is_empty(self) -> bool:
        """Checks if the vector store currently has any indexed vectors."""
        return self.index is None or self.index.ntotal == 0

    def get_total_chunks(self) -> int:
        """Returns the number of chunks currently indexed."""
        return self.index.ntotal if self.index is not None else 0

    def build_index(self, chunks: List[str], embeddings: np.ndarray, document_name: str = ""):
        """
        Populates the FAISS index with new embeddings and saves the text chunks.
        Replaces any previously stored document index.

        Args:
            chunks (List[str]): List of text chunks.
            embeddings (np.ndarray): 2D array of embeddings corresponding to the chunks.
            document_name (str): The filename of the uploaded document.
        """
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            raise ValueError("Cannot build vector index with empty chunks or embeddings.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: Got {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        # Reset existing index
        self._initialize_index()
        self.document_name = document_name

        # Ensure embeddings are contiguous float32 array
        contiguous_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Add vectors to FAISS index
        self.index.add(contiguous_embeddings)
        self.chunks = list(chunks)
        print(f"FAISS index built successfully: {self.index.ntotal} vectors added for '{document_name}'.")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the top_k most similar chunks.

        Args:
            query_embedding (np.ndarray): 2D array with shape (1, dimension).
            top_k (int): Number of nearest neighbors to retrieve (default is 3).

        Returns:
            List[Dict[str, Any]]: List of retrieved items containing chunk text, rank, and distance.
        """
        if self.is_empty():
            return []

        # Make sure top_k does not exceed total indexed items
        k = min(top_k, self.index.ntotal)
        contiguous_query = np.ascontiguousarray(query_embedding, dtype=np.float32)

        distances, indices = self.index.search(contiguous_query, k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "rank": rank,
                    "chunk": self.chunks[idx],
                    "score": float(dist),
                    "document_name": self.document_name
                })

        return results

    def clear(self):
        """Clears the FAISS index and stored chunks."""
        self._initialize_index()


# Global vector store instance for simple single-document RAG
vector_store = FAISSVectorStore(dimension=384)
