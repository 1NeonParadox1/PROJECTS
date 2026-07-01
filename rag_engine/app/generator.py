"""
Generation step: builds a grounded prompt from retrieved chunks and calls
Gemini. The prompt instructs the model to answer only from the provided
context, cite sources inline as [n], and explicitly say when the context
doesn't contain the answer -- this is the main lever against hallucination.
"""
from __future__ import annotations

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge-base assistant. Answer the \
user's question using ONLY the numbered sources provided below. Rules:
- Cite every factual claim with the matching source number in square brackets, e.g. [2].
- If multiple sources support a claim, cite all of them, e.g. [1][3].
- If the sources do not contain enough information to answer, say so explicitly \
instead of guessing.
- Do not use outside knowledge beyond what is in the sources.
- Be concise and directly answer the question first, then add detail if useful."""


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("metadata", {}).get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=15))
def generate_answer(query: str, chunks: list[dict]) -> dict:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    # Initialize Google GenAI Client
    client = genai.Client(api_key=settings.gemini_api_key)
    context = _format_context(chunks)

    user_message = f"Sources:\n\n{context}\n\nQuestion: {query}"

    # Build generation configuration with System Prompt instructions
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
    )

    response = client.models.generate_content(
        model=settings.generation_model,
        contents=user_message,
        config=config,
    )

    return {
        "answer": response.text,
        "sources": [
            {
                "index": i + 1,
                "source": c.get("metadata", {}).get("source", "unknown"),
                "chunk_id": c["id"],
                "text_preview": c["text"][:220],
            }
            for i, c in enumerate(chunks)
        ],
    }