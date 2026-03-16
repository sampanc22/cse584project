import asyncio
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
        "Answer the user's question using only the provided information. "
        "Respond as directly and concisely as possible. "
        "Return only the answer itself, not an explanation or a restatement of the question. "
        "Prefer the shortest direct answer possible, usually a name or short phrase. "
        "If the information is insufficient, say 'None'."
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
    base_instruction = (
        "Answer using only the information below. "
        "Return only the shortest direct answer. "
        "Do not explain. "
        "Do not restate the question. "
        "If the answer is not supported by the information, return 'None'."
    )

    if mode == "document":
        parts = [
            base_instruction,
            f"Context:\n{long_context}",
            f"Question:\n{prompt}",
        ]
        return "\n\n".join(parts)

    if mode == "dialogue":
        history_text = "\n".join(history or [])
        parts = [
            base_instruction,
            f"Dialogue History:\n{history_text}" if history_text else "Dialogue History:\n",
            f"Context:\n{long_context}",
            f"Question:\n{prompt}",
        ]
        return "\n\n".join(parts)

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
                if text:
                    final_text_parts.append(text)

    return "\n".join(final_text_parts).strip()


def generate_fresh_with_agent(
    prompt: str,
    long_context: str,
    history: list[str] | None = None,
    mode: str = "document",
) -> str:
    try:
        return asyncio.run(
            generate_fresh_with_agent_async(
                prompt=prompt,
                long_context=long_context,
                history=history,
                mode=mode,
            )
        )
    except Exception as e:
        return f"GEMINI_CALL_FAILED: {type(e).__name__}: {e}"