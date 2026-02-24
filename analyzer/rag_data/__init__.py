from pathlib import Path


def get_path() -> Path:
    """Return the path to the bundled ChromaDB RAG data directory."""
    return Path(__file__).parent
