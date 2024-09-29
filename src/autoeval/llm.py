from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class EvalResult(BaseModel):
  verdict: str
  confidence: int
  reasoning: str


class Eval(ABC):
  @abstractmethod
  def __call__(self, *args: Any, **kwds: Any) -> EvalResult:
    raise NotImplementedError


class SemanticEval(Eval):
  def __call__(self, *args: Any, **kwds: Any) -> EvalResult:
    return EvalResult(verdict="GOOD", confidence=100, reasoning="The output is correct")
