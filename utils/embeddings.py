"""
Multilingual Embedding Manager Module.
Uses SentenceTransformers with paraphrase-multilingual-mpnet-base-v2
to provide cross-lingual text embeddings (English, Tamil, Hindi, etc.).
"""

from typing import List
from sentence_transformers import SentenceTransformer
from utils.config import config
from utils.logging_utils import get_logger

logger = get_logger("Embeddings")


class MultilingualEmbeddings:
    """
    Multilingual Embedding Model Manager.
    Encapsulates SentenceTransformer model initialization and embedding generation.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading Multilingual Embedding Model: '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Multilingual Embedding Model loaded successfully.")
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embedding vectors for a list of text documents.

        Args:
            texts: List of document strings.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generates embedding vector for a single search query.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        if not text:
            return []
        embedding = self.model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embedding.tolist()


# Global Singleton Instance
embedding_manager = MultilingualEmbeddings()
