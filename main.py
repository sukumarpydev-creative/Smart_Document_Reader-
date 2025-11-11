from core.utils import load_sample_docs
from models.embeddings import create_embeddings
from models.vector_store import create_vector_store
from models.llm import create_llm
from models.prompt import create_prompt
from chains.conversational_rag import ConversationalRAG
from core.logger import logger
from dotenv import load_dotenv

load_dotenv()

def main():
    logger.info("Starting RAG Assistant...")

    docs = load_sample_docs()
    embeddings = create_embeddings()
    vec_store = create_vector_store(docs, embeddings)
    llm = create_llm()
    prompt_template = create_prompt()

    rag = ConversationalRAG(vec_store, llm, prompt_template)

    print("\n--- Conversation Start ---")
    print("User: What is the leave policy?")
    print("Assistant:", rag.run("What is the leave policy?"))

    print("\nUser: What about maternity leave?")
    print("Assistant:", rag.run("What about maternity leave?"))

    print("\nUser: Is paternity leave included?")
    print("Assistant:", rag.run("Is paternity leave included?"))

if __name__ == "__main__":
    main()
