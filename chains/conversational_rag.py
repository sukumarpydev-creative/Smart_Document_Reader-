from langchain_classic.agents import create_openai_functions_agent, AgentExecutor
from langchain_classic.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from core.logger import logger
from core.utils import format_docs
from models.summary_memory import ConversationSummaryMemory


class ConversationalRAG:
    """Encapsulates the conversational RAG logic."""

    def __init__(self, vector_store, llm, prompt_template, top_k=3, summary_store="file"):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        self.llm = llm
        self.prompt_template = prompt_template
        #Adding short memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #Adding Summary Memory
        self.summary_mem = ConversationSummaryMemory(
            llm=self.llm,
            path="data/long_term_summary.txt",
            store=summary_store,
            merge_threshold=6,
            max_context_chars=2000,
        )
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
            max_iterations=3
        )

    def run(self, query: str) -> str:
        try:
            logger.info(f"User Query: {query}")
            response = self.executor.invoke({"input": query})
            answer = response.get("output") or response.get("response") or str(response)
            logger.info("Response generated successfully.")

            try:
                updated = self.summary_mem.maybe_update_from_buffer(self.memory)
                if updated:
                    logger.info("Long-term summary updated (size=%d chars).", len(updated))
            except Exception:
                logger.exception("Failed to update long-term summary (non-fatal).")

            logger.info("Response generated successfully.")

            return answer

        except Exception as e:
            logger.error(f"Error in ConversationalRAG: {e}", exc_info=True)
            return "⚠️ Sorry, something went wrong while generating a response."
