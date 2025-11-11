from langchain_community.vectorstores import FAISS
from core.logger import logger

def create_vector_store(docs, embeddings):
    logger.info("Creating FAISS vector store...")
    return FAISS.from_documents(docs, embeddings)
