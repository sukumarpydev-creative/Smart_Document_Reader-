from langchain_core.prompts import PromptTemplate

def create_prompt():
    return PromptTemplate(
        input_variables=["input", "chat_history", "agent_scratchpad"],
        template=(
            "You are a helpful conversational RAG agent.\n"
            "Use the RAGRetriever tool whenever the user asks about the documents.\n\n"
            "Chat History:\n{chat_history}\n\n"
            "User Question:\n{input}\n\n"
            "Thoughts and reasoning (internal) should go into:\n{agent_scratchpad}\n"
        )
    )
