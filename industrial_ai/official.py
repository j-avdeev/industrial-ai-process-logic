from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .paths import DEFAULT_DATA_DIR


def load_generator(data_dir: Path | None = None) -> ModuleType:
    """Load the official generator/validator module from the copied starter pack."""
    root = Path(data_dir or DEFAULT_DATA_DIR)
    path = root / "generate_sequences.py"
    if not path.exists():
        raise FileNotFoundError(f"Official generator not found: {path}")

    spec = importlib.util.spec_from_file_location("official_generate_sequences", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official generator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
