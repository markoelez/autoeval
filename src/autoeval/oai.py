import json
import os
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from .util import get_json


async def sample(
  messages: list[dict[str, str]],
  temperature: float = 1.0,
  max_tokens: int = 64,
  stream: bool = False,
  response_format: Optional[BaseModel] = None,
  model="gpt-4o-mini",
) -> str:
  client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  res = await client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    stream=stream,
  )
  raw = res.choices[0].message.content
  if response_format is None:
    return raw
  raw_json = get_json(raw)
  return response_format(**json.loads(raw_json))
