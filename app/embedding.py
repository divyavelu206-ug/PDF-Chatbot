"""
Embedding Module
Generates dense vector embeddings using Sentence Transformers (all-MiniLM-L6-v2).
Embeddings run completely locally on CPU without requiring paid external APIs.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Singleton wrapper around SentenceTransformer to prevent redundant reloading of the weights.
    """
    _instance = None
    _model = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            print(f"Loading embedding model: {model_name}...")
            # Use locally cached weights first to load instantly without network lag
            try:
                cls._model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                cls._model = SentenceTransformer(model_name)
            print("Embedding model loaded successfully.")
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates vector embeddings for a list of text strings.

        Args:
            texts (List[str]): List of text chunks.

        Returns:
            np.ndarray: 2D array of embeddings with shape (N, dimension) and dtype float32.
        """
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        # encode returns numpy array or torch tensor; ensure float32 for FAISS
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates vector embedding for a single question/query.

        Args:
            query (str): The search query or question.

        Returns:
            np.ndarray: 2D array of embedding with shape (1, dimension) and dtype float32.
        """
        embedding = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def get_dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model (384 for all-MiniLM-L6-v2).
        """
        return self._model.get_sentence_embedding_dimension()


# Convenient singleton helper function
_embedding_instance = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Returns the singleton instance of the EmbeddingModel."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel(model_name=model_name)
    return _embedding_instance
