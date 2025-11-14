"""I/O helper functions for DeepD3 to Zarr conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())
