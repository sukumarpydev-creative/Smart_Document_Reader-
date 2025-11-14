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

    rag = ConversationalRAG(
        vector_store=vec_store,
        llm=llm,
        prompt_template=prompt_template,
        top_k=3,
        summary_store="file"
    )

    print("\n--- Conversation Start (type 'exit' to quit) ---")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        print("Assistant:", rag.run(user_query))

if __name__ == "__main__":
    main()

