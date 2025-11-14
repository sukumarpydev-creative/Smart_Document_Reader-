# models/summary_memory.py
import os
from typing import List, Optional
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from core.logger import logger

class ConversationSummaryMemory:
    """
    Lightweight long-term summary memory.

    - Keeps an in-file summary string (persisted).
    - When buffer grows beyond `merge_threshold`, summarizes the recent buffer + existing summary.
    - Exposes `get_summary()` and `maybe_update_from_buffer(buffer_memory)`.

    Usage:
      summary_mem = ConversationSummaryMemory(llm=my_llm, path="data/summary.txt", merge_threshold=6)
      summary_mem.maybe_update_from_buffer(buffer_memory)
    """
    def __init__(
        self,
        llm: OpenAI,
        path: str = "data/long_term_summary.txt",
        store = "file",
        merge_threshold: int = 6,
        max_context_chars: int = 2000,
    ):
        self.llm = llm
        self.path = path
        self.merge_threshold = merge_threshold
        self.max_context_chars = max_context_chars
        self._summary = ""
        self._ensure_file()
        self._load()
        self.store = store

    def _ensure_file(self):
        dirpath = os.path.dirname(self.path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._summary = f.read().strip()
            except Exception:
                logger.exception("Failed to load summary file; starting with empty summary.")
                self._summary = ""
        else:
            self._summary = ""

    def _persist(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(self._summary or "")
        except Exception:
            logger.exception("Failed to persist summary to disk.")

    def get_summary(self) -> str:
        return self._summary or ""

    def _build_summarization_prompt(self, existing_summary: str, recent_messages: List[str]) -> str:
        # Keep prompt short and directive.
        recent_text = "\n\n".join(recent_messages)
        # Truncate to avoid huge prompt
        if len(recent_text) > self.max_context_chars:
            recent_text = recent_text[-self.max_context_chars:]

        prompt = (
            "You are a concise summarizer that maintains a long-term conversation summary.\n\n"
            "Existing long-term summary:\n"
            f"{existing_summary}\n\n"
            "Recent conversation snippets (newest last):\n"
            f"{recent_text}\n\n"
            "Update the long-term summary by merging new important facts from the recent conversation. "
            "Keep the summary concise (max 150 words). Keep only facts that would help the assistant answer future queries.\n\n"
            "Return ONLY the updated summary (no commentary)."
        )
        return prompt

    def maybe_update_from_buffer(self, buffer_memory: ConversationBufferMemory) -> Optional[str]:
        """
        If the buffer_memory length exceeds merge_threshold, summarize and merge into long-term summary.
        Returns the new summary if updated, otherwise None.
        """
        # buffer_memory stores messages in memory_key "chat_history" (return_messages=True)
        try:
            mem_vars = buffer_memory.load_memory_variables({})
            messages = mem_vars.get("chat_history", [])
            # If it's the new style messages (list of dict/Message), convert to simple strings
            recent_messages = []
            for m in messages:
                # support both message strings and message objects
                if isinstance(m, str):
                    recent_messages.append(m)
                else:
                    # try to get .content or dict style
                    content = None
                    if hasattr(m, "content"):
                        content = getattr(m, "content")
                    elif isinstance(m, dict):
                        content = m.get("content") or m.get("text")
                    if content:
                        recent_messages.append(content)
            # nothing to do
            if not recent_messages:
                return None

            # If not enough messages, skip summarization
            if len(recent_messages) < self.merge_threshold:
                return None

            # Build prompt and call LLM
            prompt = self._build_summarization_prompt(self.get_summary(), recent_messages[-self.merge_threshold:])

            # LLM call — use invoke to be compatible with LCEL LLM runnables
            try:
                resp = self.llm.invoke(prompt)
                # resp may be string or object with .content — handle both
                new_summary = getattr(resp, "content", resp)
                if not isinstance(new_summary, str):
                    new_summary = str(new_summary)
                new_summary = new_summary.strip()
                if new_summary:
                    # replace stored summary with the new one (could be more advanced: merge)
                    self._summary = new_summary
                    self._persist()
                    logger.info("Long-term summary updated.")
                    return self._summary
            except Exception as e:
                logger.exception("Error while calling LLM for summarization: %s", e)
                return None

        except Exception as e:
            logger.exception("Error in maybe_update_from_buffer: %s", e)
            return None

        return None
