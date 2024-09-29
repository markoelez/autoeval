import json
import os
import re
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Any, Optional

import jinja2
from openai import AsyncOpenAI
from pydantic import BaseModel


def get_path(relative_path: str) -> Path:
  module = "autoeval"
  try:
    return resources.files(module).joinpath(relative_path).resolve()
  except KeyError:
    fallback_path = Path(__file__).parent / relative_path
    if not fallback_path.exists():
      raise FileNotFoundError(f"Binary file not found at {fallback_path}")
    return fallback_path.resolve()


def get_json(s: str) -> str:
  s = re.sub(r"^[^{]*", "", s)
  s = re.sub(r"[^}]*$", "", s)
  return s


async def sample(
  messages: list[dict[str, str]],
  temperature: float = 1.0,
  max_tokens: int = 64,
  stream: bool = False,
  response_format: Optional[BaseModel] = None,
) -> str:
  client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  res = await client.chat.completions.create(
    model="gpt-4o",
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


class EvalResult(BaseModel):
  score: int
  reasoning: str

  @classmethod
  def to_tpl(cls) -> str:
    return '{"score": int (0 to 100), "reasoning": str}'


class Eval(ABC):
  @abstractmethod
  async def __call__(self, *args: Any, **kwds: Any) -> EvalResult:
    raise NotImplementedError


class SemanticEval(Eval):
  # result = await semantic_eval(input, output, expect, judge="gpt-4o")
  async def __call__(self, input: str, output: str, expect: list[str], judge: str = "gpt-4o") -> EvalResult:
    tpl_path = get_path("tpl/judge.txt")
    with open(tpl_path, "r") as f:
      tpl = jinja2.Template(f.read())
    params = {
      "input": input,
      "output": output,
      "expectations": expect,
      "response_format": EvalResult.to_tpl(),
    }
    messages = [{"role": "user", "content": tpl.render(**params)}]
    return await sample(messages=messages, temperature=0.3, max_tokens=512, response_format=EvalResult)
