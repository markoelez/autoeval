autoeval framework for AI models


target interface:

```python
from autoeval.llm import SemanticEvaluator

eval = SemanticEval()

expect = [
  'The output should mention China',
  'The output should not exceed 5 words',
]

input = "Which country has the highest population?"
output = "People's Republic of China"

result = eval(input, output, expect)

print(result.verdict)
print(result.confidence)
print(result.reasoning)
```
