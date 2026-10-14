import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO, log_file: str = "sec_rag.log"):
    """
    Configures the global logging settings for the SEC RAG system.
    """
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path / log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific levels for noisy libraries
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    return logging.getLogger("sec_rag")

# Create a default logger instance
logger = setup_logging()
