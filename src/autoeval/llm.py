import asyncio
from abc import ABC, abstractmethod

import jinja2
from pydantic import BaseModel

from .oai import sample
from .util import get_path


def load_tpl(path: str) -> jinja2.Template:
  with open(get_path(path), "r") as f:
    return jinja2.Template(f.read())


class EvalResult(BaseModel):
  score: float
  reasoning: str


class EvalCriteria(ABC):
  @abstractmethod
  async def __call__(self, input: str, output: str, expected: str, judge: str) -> EvalResult:
    raise NotImplementedError


class Factuality(EvalCriteria):
  tpl = load_tpl("tpl/factuality.txt")

  async def __call__(self, input: str, output: str, expected: str, judge: str) -> EvalResult:
    params = {
      "input": input,
      "output": output,
      "expected": expected,
    }
    messages = [{"role": "user", "content": self.tpl.render(**params)}]
    return await sample(messages=messages, temperature=0.3, max_tokens=512, response_format=EvalResult, model=judge)


class Evaluator:
  def __init__(self, criteria: list[EvalCriteria], threshold: float = 0.9, judge: str = "gpt-4o-mini"):
    self.criteria = criteria
    self.threshold = threshold
    self.judge = judge
    self._concurrency = 10
    self._semaphore = asyncio.Semaphore(self._concurrency)

  async def _run(self, input: str, output: str, expected: str, criteria: EvalCriteria):
    async with self._semaphore:
      return await criteria(input, output, expected, self.judge)

  async def run(self, dataset: list[dict[str, str]]) -> list[EvalResult]:
    tasks = [self._run(row["input"], row["output"], row["expected"], criteria) for row in dataset for criteria in self.criteria]
    return await asyncio.gather(*tasks)
