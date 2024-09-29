import re
from importlib import resources
from pathlib import Path


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
