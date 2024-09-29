from abc import ABC, abstractmethod
from typing import Any

import jinja2
from pydantic import BaseModel

from .oai import sample
from .util import get_path


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
