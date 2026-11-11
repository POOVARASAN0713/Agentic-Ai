"""
ChromaDB Vector Database Management Module.
Manages persistent vector collections for FAQs and Policy Documents
using multilingual embeddings and similarity search routines.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings

from utils.config import config
from utils.embeddings import embedding_manager
from utils.logging_utils import get_logger

logger = get_logger("VectorStore")


class VectorStoreManager:
    """
    Manages ChromaDB persistent vector collections for FAQs and Policy Documents.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIR
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing persistent ChromaDB client at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # 1. Initialize isolated collections
        self.faq_collection = self.client.get_or_create_collection(
            name="faq_collection",
            metadata={"hnsw:space": "cosine"}
        )

        self.document_collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def index_faqs(self, faqs: List[Dict[str, Any]], force_reindex: bool = False) -> int:
        """
        Indexes FAQ items into faq_collection.
        Skips re-indexing if collection is already populated (unless force_reindex=True).

        Args:
            faqs: List of FAQ dictionaries (id, question, answer, category).
            force_reindex: If True, clears collection and re-indexes.

        Returns:
            Number of indexed FAQs.
        """
        existing_count = self.faq_collection.count()
        if existing_count > 0 and not force_reindex:
            logger.info(f"FAQ collection already contains {existing_count} records. Skipping re-indexing.")
            return existing_count

        if force_reindex and existing_count > 0:
            logger.info("Force re-indexing FAQ collection...")
            # Recreate collection
            self.client.delete_collection("faq_collection")
            self.faq_collection = self.client.create_collection(
                name="faq_collection", metadata={"hnsw:space": "cosine"}
            )

        if not faqs:
            logger.warning("No FAQs provided to index.")
            return 0

        logger.info(f"Generating embeddings for {len(faqs)} FAQs...")
        questions = [item["question"] for item in faqs]
        embeddings = embedding_manager.embed_documents(questions)

        ids = [str(item["id"]) for item in faqs]
        documents = [item["answer"] for item in faqs]
        metadatas = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "faq_id": str(item["id"]),
            }
            for item in faqs
        ]

        self.faq_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Successfully indexed {len(faqs)} FAQs in ChromaDB 'faq_collection'.")
        return len(faqs)

    def index_documents(self, chunks: List[Any], force_reindex: bool = False) -> int:
        """
        Indexes document chunks into document_collection.
        Skips re-indexing if collection is already populated (unless force_reindex=True).

        Args:
            chunks: List of Document chunk objects.
            force_reindex: If True, clears collection and re-indexes.

        Returns:
            Number of indexed text chunks.
        """
        existing_count = self.document_collection.count()
        if existing_count > 0 and not force_reindex:
            logger.info(f"Document collection already contains {existing_count} records. Skipping re-indexing.")
            return existing_count

        if force_reindex and existing_count > 0:
            logger.info("Force re-indexing Document collection...")
            self.client.delete_collection("document_collection")
            self.document_collection = self.client.create_collection(
                name="document_collection", metadata={"hnsw:space": "cosine"}
            )

        if not chunks:
            logger.warning("No document chunks provided to index.")
            return 0

        logger.info(f"Generating embeddings for {len(chunks)} document chunks...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.embed_documents(texts)

        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": getattr(chunk, "metadata", {}).get("source", "unknown"),
                "file_type": getattr(chunk, "metadata", {}).get("file_type", "txt"),
            }
            for chunk in chunks
        ]

        self.document_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Successfully indexed {len(chunks)} document chunks in ChromaDB 'document_collection'.")
        return len(chunks)

    def _refresh_document_collection(self) -> None:
        """Reopen the named collection after another process recreates it."""
        self.document_collection = self.client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"},
        )

    def search_faqs(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search on the FAQ collection.

        Args:
            query: User search query.
            top_k: Number of top matches to return.

        Returns:
            List of match dictionaries containing question, answer, category, and similarity_score (0.0 to 1.0).
        """
        if not query or self.faq_collection.count() == 0:
            return []

        query_embedding = embedding_manager.embed_query(query)
        results = self.faq_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.faq_collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # Convert Cosine Distance to Cosine Similarity score
                similarity = max(0.0, min(1.0, 1.0 - distance))
                metadata = results["metadatas"][0][i]

                matches.append({
                    "id": results["ids"][0][i],
                    "question": metadata.get("question", ""),
                    "answer": metadata.get("answer", results["documents"][0][i]),
                    "category": metadata.get("category", "general"),
                    "similarity_score": round(float(similarity), 4),
                    "distance": round(float(distance), 4),
                })

        return matches

    def search_documents(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search on the Document collection for RAG.

        Args:
            query: User search query.
            top_k: Number of top document chunks to retrieve.

        Returns:
            List of match dictionaries containing page_content, metadata, and similarity_score.
        """
        if not query:
            return []

        try:
            collection_count = self.document_collection.count()
        except Exception as error:
            if "does not exist" not in str(error).lower():
                raise
            logger.warning("Document collection handle is stale; reopening it.")
            self._refresh_document_collection()
            collection_count = self.document_collection.count()

        if collection_count == 0:
            return []

        query_embedding = embedding_manager.embed_query(query)
        results = self.document_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )

        matches = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                similarity = max(0.0, min(1.0, 1.0 - distance))

                matches.append({
                    "id": results["ids"][0][i],
                    "page_content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": round(float(similarity), 4),
                    "distance": round(float(distance), 4),
                })

        return matches


# Global VectorStore Singleton Instance
vector_store = VectorStoreManager()
