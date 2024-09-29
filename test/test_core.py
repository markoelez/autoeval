import pytest

from autoeval.llm import SemanticEval


@pytest.mark.asyncio
async def test_semantic_eval():
  semantic_eval = SemanticEval()

  expect = [
    "The output should mention China",
    "The output should not exceed 5 words",
  ]

  input = "Which country has the highest population?"
  output = "People's Republic of China"

  result = await semantic_eval(input, output, expect, judge="gpt-4o")

  assert result.score > 95
