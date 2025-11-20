from functools import lru_cache

from chains.conversational_rag import ConversationalRAG
from core.logger import logger
from core.utils import load_sample_docs
from models.embeddings import create_embeddings
from models.llm import create_llm
from models.prompt import create_prompt
from models.vector_store import create_vector_store



logger.info("Initializing the Agent factory...")

@lru_cache(maxsize=1)
def get_agent():
    llm = create_llm()
    embeddings = create_embeddings()
    docs = load_sample_docs()
    vec_store = create_vector_store(docs, embeddings)
    # retriever = vec_store.as_retriever(search_kwargs={"k":4})
    # memory = ConversationSummaryMemory(path="data/long_term_summary.txt")
    prompt_template = create_prompt()
    agent = ConversationalRAG(
        vector_store=vec_store,
        llm=llm,
        prompt_template=prompt_template,
        top_k=3,
        summary_store="file"
    )

    return agent

