import pytest

from autoeval.llm import Evaluator, Factuality


@pytest.mark.asyncio
async def test_semantic_eval():
  dataset = [
    {"input": "Which country has the highest population?", "output": "People's Republic of China", "expected": "China"},
  ]

  evaluator = Evaluator(criteria=[Factuality()], threshold=0.9, judge="gpt-4o-mini")

  results = await evaluator.run(dataset)

  assert all([x.score >= 0.9 for x in results])
