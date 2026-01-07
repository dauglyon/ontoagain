"""LLM wrapper using LiteLLM and Instructor.

Supports any LiteLLM-compatible provider. Configure via environment variables:

Direct Anthropic:
    ANTHROPIC_API_KEY=sk-ant-...

OpenAI-compatible (e.g., CBORG):
    OPENAI_API_BASE=https://api.cborg.lbl.gov/v1
    OPENAI_API_KEY=your-key
"""

import asyncio
import os

import instructor
from litellm import acompletion, completion

# Reusable event loop to avoid LiteLLM async worker conflicts
_event_loop = None


def run_async(coro):
    """Run an async coroutine, reusing the same event loop.

    This avoids LiteLLM's logging worker conflicts that occur when
    asyncio.run() is called multiple times (each creates a new loop).
    """
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop.run_until_complete(coro)


def get_client(model: str = "anthropic/claude-sonnet"):
    """Get an Instructor-wrapped LiteLLM client.

    Args:
        model: The model identifier as exposed by your provider.

    Returns:
        An Instructor client configured for the specified model.
    """
    # Instructor wraps the completion function to enable structured outputs
    client = instructor.from_litellm(completion)
    return client, model


def _get_api_base():
    """Get custom API base if configured."""
    return os.environ.get("OPENAI_API_BASE")


def call_llm(
    client,
    model: str,
    messages: list[dict],
    response_model=None,
    max_retries: int = 3,
):
    """Call the LLM with optional structured output.

    Args:
        client: The Instructor client
        model: Model identifier
        messages: List of message dicts with 'role' and 'content'
        response_model: Optional Pydantic model for structured output
        max_retries: Number of retries on failure

    Returns:
        The LLM response, parsed into response_model if provided
    """
    api_base = _get_api_base()

    # For custom OpenAI-compatible endpoints, prefix model with openai/
    # to ensure LiteLLM uses OpenAI format, not provider-specific format
    effective_model = model
    if api_base and not model.startswith("openai/"):
        effective_model = f"openai/{model}"

    # Build kwargs
    kwargs = {
        "model": effective_model,
        "messages": messages,
    }
    if api_base:
        kwargs["api_base"] = api_base

    if response_model:
        return client.chat.completions.create(
            **kwargs,
            response_model=response_model,
            max_retries=max_retries,
        )
    else:
        # Raw completion without structured output
        from litellm import completion as raw_completion

        response = raw_completion(**kwargs)
        return response.choices[0].message.content


async def call_llm_async(
    model: str,
    messages: list[dict],
) -> str:
    """Async version of call_llm for parallel processing.

    Args:
        model: Model identifier
        messages: List of message dicts with 'role' and 'content'

    Returns:
        The LLM response content
    """
    api_base = _get_api_base()

    # For custom OpenAI-compatible endpoints, prefix model with openai/
    effective_model = model
    if api_base and not model.startswith("openai/"):
        effective_model = f"openai/{model}"

    # Build kwargs
    kwargs = {
        "model": effective_model,
        "messages": messages,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = await acompletion(**kwargs)
    return response.choices[0].message.content
