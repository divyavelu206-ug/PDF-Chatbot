"""
Memory Optimized Embedding Module
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    _instance = None
    _model = None

    def __new__(cls, model_name="all-MiniLM-L6-v2"):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._model = SentenceTransformer(
                model_name,
                device="cpu"
            )

        return cls._instance

    @property
    def model(self):
        return self._model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(texts), 8):

            batch = texts[i:i + 8]

            embeddings = self._model.encode(
                batch,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            all_embeddings.append(
                embeddings.astype(np.float32)
            )

        return np.vstack(all_embeddings)

    def generate_query_embedding(self, query: str) -> np.ndarray:

        embedding = self._model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)

    def get_dimension(self):
        return 384


_embedding_instance = None


def get_embedding_model(model_name="all-MiniLM-L6-v2"):

    global _embedding_instance

    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel(model_name)

    return _embedding_instance
