import os
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=4)
def _client(base_url):
    return OpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY") or "not-required",
    )


def complete(prompt, model, *, temperature, max_tokens=0, local_port=8001):
    """Send a single prompt to an OpenAI-compatible endpoint."""
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or f"http://localhost:{local_port}/v1"
    )
    options = {"max_tokens": max_tokens} if max_tokens else {}
    response = _client(base_url).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **options,
    )
    return response.choices[0].message.content or ""
