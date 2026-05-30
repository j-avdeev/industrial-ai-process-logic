from __future__ import annotations

from pathlib import Path

from .official import load_generator


def validate_steps(steps: list[str], data_dir: Path | None = None) -> tuple[bool, float, str]:
    generator = load_generator(data_dir)
    violations = generator.validate_sequence(steps)
    if not violations:
        return True, 0.99, ""
    return False, 0.01, violations[0].rule

