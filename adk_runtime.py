import asyncio
import time
import uuid
from dotenv import load_dotenv

from google.genai import types
from google.adk import Agent
from google.adk.apps.app import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

load_dotenv()

APP_NAME = "validity_cache_app"
USER_ID = "demo_user"

root_agent = Agent(
    name="validity_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a grounded QA assistant. "
        "Use only the provided input context. "
        "Never use outside knowledge or assumptions. "
        "If the answer is not supported by the provided input, return exactly 'None'. "
        "Return only the answer itself, as briefly as possible."
    ),
)

app = App(
    name=APP_NAME,
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(
        min_tokens=500,
        ttl_seconds=600,
        cache_intervals=5,
    ),
)

session_service = InMemorySessionService()

runner = Runner(
    app=app,
    session_service=session_service,
)


def _build_model_input(
    prompt: str,
    long_context: str,
    history: list[str] | None = None,
    mode: str = "document",
) -> str:
    if mode == "document":
        instruction = (
            "You are a strictly document-grounded question answering system.\n"
            "Answer the question using only the provided document text.\n"
            "Do not use prior knowledge, world knowledge, or outside assumptions.\n"
            "Do not infer facts not explicitly supported by the document.\n"
            "Return only the shortest direct answer.\n"
            "Do not explain or restate the question.\n"
            "If the answer is not explicitly stated in the document, return exactly: None"
        )

        return "\n\n".join(
            [
                instruction,
                "DOCUMENT START",
                long_context,
                "DOCUMENT END",
                f"Question: {prompt}",
                "Answer:",
            ]
        )

    if mode == "dialogue":
        history_text = "\n".join(history or [])
        instruction = (
            "You are a dialogue assistant.\n"
            "Answer the question using only the provided dialogue history and context.\n"
            "Do not use outside knowledge or assumptions.\n"
            "Respect the user's constraints and preferences from the dialogue.\n"
            "If the answer cannot be determined from the dialogue and context, return exactly: None\n"
            "Return only the answer.\n"
            "Do not explain."
        )

        return "\n\n".join(
            [
                instruction,
                f"Dialogue History:\n{history_text}" if history_text else "Dialogue History:\n",
                f"Context:\n{long_context}",
                f"Question: {prompt}",
                "Answer:",
            ]
        )

    raise ValueError(f"Unsupported mode: {mode}")


async def generate_fresh_with_agent_async(
    prompt: str,
    long_context: str,
    history: list[str] | None = None,
    mode: str = "document",
) -> str:
    session_id = f"session_{uuid.uuid4().hex}"

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    full_input = _build_model_input(
        prompt=prompt,
        long_context=long_context,
        history=history,
        mode=mode,
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text=full_input)],
    )

    final_text_parts = []

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text and text.strip():
                    final_text_parts.append(text.strip())

    return "\n".join(final_text_parts).strip()


_GLOBAL_LOOP: asyncio.AbstractEventLoop | None = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _GLOBAL_LOOP
    if _GLOBAL_LOOP is None or _GLOBAL_LOOP.is_closed():
        _GLOBAL_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_GLOBAL_LOOP)
    return _GLOBAL_LOOP


def generate_fresh_with_agent(
    prompt: str,
    long_context: str,
    history: list[str] | None = None,
    mode: str = "document",
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            loop = _get_loop()
            return loop.run_until_complete(
                generate_fresh_with_agent_async(
                    prompt=prompt,
                    long_context=long_context,
                    history=history,
                    mode=mode,
                )
            )
        except Exception as e:
            last_err = e
            msg = f"{type(e).__name__}: {e}"

            if ("503" in msg or "UNAVAILABLE" in msg) and attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue

            return f"GEMINI_CALL_FAILED: {msg}"

    return f"GEMINI_CALL_FAILED: {type(last_err).__name__}: {last_err}"