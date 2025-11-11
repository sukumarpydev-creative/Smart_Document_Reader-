from langchain_core.prompts import ChatPromptTemplate

def create_prompt():
    template = """
    You are a helpful assistant that answers questions using the provided context.

    Conversation History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    If the answer is not found in the context, respond with "Not enough information."
    """
    return ChatPromptTemplate.from_template(template)
