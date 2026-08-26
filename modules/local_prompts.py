from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_PATH = Path(__file__).parent / "prompts_local.yaml"


def load_local_prompts(path: Path = _DEFAULT_PATH) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
