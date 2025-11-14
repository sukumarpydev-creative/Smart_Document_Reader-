from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.tools import Tool
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

        # ---- RAG TOOL ----
        def rag_tool_fn(query: str) -> str:
            docs = self.retriever.invoke(query)
            return format_docs(docs)

        self.rag_tool = Tool(
            name="RAGRetriever",
            func=rag_tool_fn,
            description="Use this to search relevant document chunks. Input: natural language question.",
        )

        # ---- AGENT ----
        self.agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[self.rag_tool],
            prompt=self.prompt_template,  # MUST include {agent_scratchpad}
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=[self.rag_tool],
            memory=self.memory,  # correct place for memory
            verbose=False,
        )

    def run(self, query: str) -> str:
        try:
            logger.info(f"User Query: {query}")

            response = self.executor.invoke({"input": query})

            logger.info("Response generated successfully.")
            return response["output"]

        except Exception as e:
            logger.error(f"Error in ConversationalRAG: {e}", exc_info=True)
            return "⚠️ Sorry, something went wrong while generating a response."
