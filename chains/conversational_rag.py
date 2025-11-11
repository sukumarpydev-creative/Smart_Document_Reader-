from langchain_classic.memory import ConversationBufferMemory
from core.logger import logger
from core.utils import format_docs

class ConversationalRAG:
    """Encapsulates the conversational RAG logic."""

    def __init__(self, vector_store, llm, prompt_template, top_k=3):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        self.llm = llm
        self.prompt_template = prompt_template
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        logger.info("Conversational RAG initialized successfully.")

    def run(self, query: str) -> str:
        try:
            logger.info(f"User Query: {query}")

            # Load conversation history
            history = self.memory.load_memory_variables({})
            chat_history = history.get("chat_history", "")

            # Retrieve relevant chunks
            docs = self.retriever.invoke(query)
            context = format_docs(docs)

            # Build prompt
            formatted_prompt = self.prompt_template.format(
                chat_history=chat_history,
                context=context,
                question=query
            )

            # Get response
            response = self.llm.invoke(formatted_prompt)
            answer = response.content

            # Save conversation
            self.memory.save_context({"input": query}, {"output": answer})

            logger.info("Response generated successfully.")
            return answer

        except Exception as e:
            logger.error(f"Error in ConversationalRAG: {e}", exc_info=True)
            return "⚠️ Sorry, something went wrong while generating a response."
