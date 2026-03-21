from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config = deepcopy(config)
    base_dir = config_path.parent.parent
    config.setdefault("paths", {})
    for key in ("dynamic_data", "static_data", "output_dir"):
        if key in config["paths"]:
            config["paths"][key] = _resolve_path(base_dir, config["paths"][key])
    return config
