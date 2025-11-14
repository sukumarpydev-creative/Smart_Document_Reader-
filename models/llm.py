from langchain_openai import ChatOpenAI
from core.logger import logger
from core.config import LLM_MODEL, TEMPERATURE

def create_llm(model: str = LLM_MODEL, temperature: float = TEMPERATURE):
    logger.info(f"Initializing LLM: {model}")
    model = ChatOpenAI(model=model, temperature=temperature)
    return model
