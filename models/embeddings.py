from langchain_huggingface import HuggingFaceEmbeddings
from core.logger import logger
from core.config import EMBED_MODEL

def create_embeddings(model_name: str = EMBED_MODEL):
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)
