# api/server.py

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from api.agent_factory import get_agent
from api.schemas import ChangeRequest, ChangeResponse
from dotenv import load_dotenv
from core.logger import logger

load_dotenv()

app = FastAPI(
    title="Smart Document Reader API",
    description="Conversational RAG Agent with Long-Term Memory",
    version="1.0.0"
)

# ------------------------------------------------------------
# CORS (allow frontend apps like React, Vue, etc.)
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


# ------------------------------------------------------------
# Helper: Persist long-term summary memory (async)
# ------------------------------------------------------------
def persist_memory(agent):
    """
    After every request, persist the summary memory safely.
    This operation is non-blocking.
    """
    try:
        if hasattr(agent, "summary_mem"):
            agent.summary_mem._persist()
            logger.info("Summary memory persisted in background.")
    except Exception:
        logger.exception("Failed to persist long-term summary memory.")


# ------------------------------------------------------------
# Main Chat Endpoint
# ------------------------------------------------------------
@app.post("/chat", response_model=ChangeResponse)
def chat(req: ChangeRequest, background_tasks: BackgroundTasks):

    agent = get_agent()      # loads once, cached forever
    user_msg = req.message

    logger.info(f"📩 New request | session={req.session_id} | message={user_msg}")

    # Generate response from the RAG agent
    try:
        output_text = agent.run(user_msg)
    except Exception as e:
        logger.exception("RAG agent failed.")
        output_text = "⚠️ Error: Something went wrong while generating a response."

    # Persist long-term summary memory asynchronously
    background_tasks.add_task(persist_memory, agent)

    logger.info(f"📤 Response sent | {output_text[:120]}...")
    return ChangeResponse(
        session_id=req.session_id,
        response=output_text
    )
