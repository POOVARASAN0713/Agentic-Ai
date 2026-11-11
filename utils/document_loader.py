"""
Document Loader and Chunker Utility Module.
Handles parsing CSV FAQ datasets and dynamic loading/chunking of PDF and TXT policy documents.
"""

import csv
import io
import os
from pathlib import Path
from typing import List, Dict, Any

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self):
            return f"Document(page_content='{self.page_content[:30]}...', metadata={self.metadata})"

from utils.config import config
from utils.logging_utils import get_logger

logger = get_logger("DocumentLoader")


def get_data_dir() -> Path:
    """Returns absolute path to the data directory."""
    return Path(config.CHROMA_PERSIST_DIR).parent.parent / "data"


def load_faqs(csv_path: str = None) -> List[Dict[str, Any]]:
    """
    Loads FAQ items from a CSV dataset file.

    Args:
        csv_path: Optional explicit file path to faqs.csv.

    Returns:
        List of dictionaries containing FAQ records (id, question, answer, category).
    """
    if not csv_path:
        csv_path = get_data_dir() / "faqs.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error(f"FAQ file not found at: {csv_path}")
        return []

    faqs = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("question") and row.get("answer"):
                    faqs.append({
                        "id": row.get("id", str(len(faqs) + 1)),
                        "question": row.get("question", "").strip(),
                        "answer": row.get("answer", "").strip(),
                        "category": row.get("category", "general").strip(),
                    })
        logger.info(f"Successfully loaded {len(faqs)} FAQs from {csv_path.name}.")
        return faqs
    except Exception as e:
        logger.error(f"Failed to read FAQ CSV file: {str(e)}")
        return []


def load_faqs_from_bytes(csv_bytes: bytes) -> List[Dict[str, Any]]:
    """Parse FAQ records from a CSV uploaded through the frontend."""
    if not csv_bytes:
        return []

    faqs = []
    try:
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        required_columns = {"question", "answer"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain question and answer columns.")

        for row in reader:
            if row.get("question") and row.get("answer"):
                faqs.append(
                    {
                        "id": row.get("id", str(len(faqs) + 1)),
                        "question": row["question"].strip(),
                        "answer": row["answer"].strip(),
                        "category": (row.get("category") or "general").strip(),
                    }
                )
        logger.info("Successfully parsed %s uploaded FAQs.", len(faqs))
        return faqs
    except (UnicodeDecodeError, ValueError, csv.Error) as error:
        logger.error("Failed to parse uploaded FAQ CSV: %s", error)
        raise ValueError(str(error)) from error


def load_policy_documents(documents_dir: str = None) -> List[Document]:
    """
    Dynamically loads all .txt and .pdf policy reference documents from directory.

    Args:
        documents_dir: Path to directory containing document files.

    Returns:
        List of Document objects.
    """
    if not documents_dir:
        documents_dir = get_data_dir() / "documents"

    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        logger.error(f"Documents directory not found at: {documents_dir}")
        return []

    loaded_docs: List[Document] = []

    for file_path in sorted(documents_dir.glob("*")):
        if file_path.name.startswith("."):
            continue

        if file_path.suffix.lower() == ".txt":
            try:
                with open(file_path, mode="r", encoding="utf-8") as f:
                    text = f.read()
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path.name, "file_type": "txt"}
                )
                loaded_docs.append(doc)
                logger.info(f"Loaded TXT document: {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading TXT document {file_path.name}: {str(e)}")

        elif file_path.suffix.lower() == ".pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                pdf_pages = loader.load()
                for page in pdf_pages:
                    page.metadata["file_type"] = "pdf"
                    page.metadata["source"] = file_path.name
                loaded_docs.extend(pdf_pages)
                logger.info(f"Loaded PDF document: {file_path.name} ({len(pdf_pages)} pages)")
            except Exception as e:
                logger.error(f"Error loading PDF document {file_path.name}: {str(e)}")

    logger.info(f"Loaded total {len(loaded_docs)} document files/pages from {documents_dir}.")
    return loaded_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Splits documents into smaller text chunks suitable for high-precision RAG retrieval.

    Args:
        documents: List of raw loaded documents.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.

    Returns:
        List of chunked Document objects with metadata attached.
    """
    if not documents:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\nSECTION ", "\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
    except ImportError:
        # Graceful fallback chunker if langchain_text_splitters is still installing
        chunks = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata=dict(doc.metadata) if hasattr(doc, "metadata") else {},
                    )
                )
                start += (chunk_size - chunk_overlap)

    logger.info(f"Split {len(documents)} raw document(s) into {len(chunks)} text chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return chunks
