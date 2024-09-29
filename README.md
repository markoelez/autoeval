# autoeval

autoeval framework for AI models

# goal

target interface:

```python
from autoeval.llm import SemanticEval


semantic_eval = SemanticEval()

expect = [
  'The output should mention China',
  'The output should not exceed 5 words',
]

input = "Which country has the highest population?"
output = "People's Republic of China"

result = semantic_eval(input, output, expect, judge='gpt-4o')

print(result.verdict)
print(result.confidence)
print(result.reasoning)
```

# development

run tests
```
uv run pytest test
```
