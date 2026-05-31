from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .hashing import file_sha256
from .paths import PROJECT_ROOT


LEONARDO_SHELL_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "leonardo_common.sh",
    PROJECT_ROOT / "scripts" / "leonardo_probe.sh",
    PROJECT_ROOT / "scripts" / "leonardo_generate.sh",
    PROJECT_ROOT / "scripts" / "leonardo_train.sh",
    PROJECT_ROOT / "scripts" / "leonardo_train_scaling.sh",
    PROJECT_ROOT / "scripts" / "leonardo_infer.sh",
    PROJECT_ROOT / "scripts" / "leonardo_finalize.sh",
    PROJECT_ROOT / "scripts" / "leonardo_full_pipeline.sh",
]


def _script_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _script_file_row(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "path": _script_label(path),
        "exists": path.exists(),
        "bytes": None,
        "sha256": "",
    }
    if path.exists() and path.is_file():
        row["bytes"] = path.stat().st_size
        row["sha256"] = file_sha256(path)
    return row


def _scan_balanced_shell_text(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    single_quote_line = 0
    double_quote_line = 0
    paren_stack: list[tuple[str, int]] = []
    in_single = False
    in_double = False
    escaped = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        index = 0
        while index < len(line):
            char = line[index]
            next_char = line[index + 1] if index + 1 < len(line) else ""
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\" and not in_single:
                escaped = True
                index += 1
                continue
            if (
                not in_single
                and not in_double
                and char == "#"
                and (index == 0 or line[index - 1].isspace())
            ):
                break
            if char == "'" and not in_double:
                if in_single:
                    in_single = False
                    single_quote_line = 0
                else:
                    in_single = True
                    single_quote_line = lineno
                index += 1
                continue
            if char == '"' and not in_single:
                if in_double:
                    in_double = False
                    double_quote_line = 0
                else:
                    in_double = True
                    double_quote_line = lineno
                index += 1
                continue
            if in_single or in_double:
                index += 1
                continue
            if char == "[" and next_char == "[":
                paren_stack.append(("[[", lineno))
                index += 2
                continue
            if char == "]" and next_char == "]":
                if not paren_stack or paren_stack[-1][0] != "[[":
                    failures.append(f"{_script_label(path)}:{lineno}: unmatched ]]")
                else:
                    paren_stack.pop()
                index += 2
                continue
            if char == "(" and next_char == "(":
                paren_stack.append(("(( ", lineno))
                index += 2
                continue
            if char == ")" and next_char == ")":
                if not paren_stack or paren_stack[-1][0] != "(( ":
                    failures.append(f"{_script_label(path)}:{lineno}: unmatched ))")
                else:
                    paren_stack.pop()
                index += 2
                continue
            index += 1
        escaped = False
    if in_single:
        failures.append(f"{_script_label(path)}:{single_quote_line}: unterminated single quote")
    if in_double:
        failures.append(f"{_script_label(path)}:{double_quote_line}: unterminated double quote")
    for opener, lineno in paren_stack:
        closer = "]]" if opener == "[[" else "))"
        failures.append(f"{_script_label(path)}:{lineno}: missing {closer}")
    return failures


def audit_script(path: Path) -> list[str]:
    failures: list[str] = []
    label = _script_label(path)
    if not path.exists():
        return [f"Missing Leonardo shell script: {label}"]
    raw = path.read_bytes()
    if not raw:
        return [f"Leonardo shell script is empty: {label}"]
    if b"\r\n" in raw:
        failures.append(f"{label} contains CRLF line endings; Slurm Bash scripts must use LF")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"{label} is not UTF-8 readable: {exc}"]
    lines = text.splitlines()
    if not lines or lines[0] != "#!/usr/bin/env bash":
        failures.append(f"{label} must start with #!/usr/bin/env bash")
    if path.name != "leonardo_common.sh" and "set -euo pipefail" not in text:
        failures.append(f"{label} does not enable set -euo pipefail")
    if path.name != "leonardo_common.sh":
        if 'source "${SCRIPT_DIR}/leonardo_common.sh"' not in text:
            failures.append(f"{label} does not source scripts/leonardo_common.sh")
        if "#SBATCH --output=artifacts/slurm/" not in text:
            failures.append(f"{label} does not write Slurm output under artifacts/slurm")
        if "mkdir -p artifacts/slurm" not in text:
            failures.append(f"{label} does not create artifacts/slurm in the script body")
    else:
        for function_name in ("require_positive_int", "require_min_int", "require_max_int", "require_choice"):
            if f"{function_name}()" not in text:
                failures.append(f"{label} is missing {function_name}()")
    failures.extend(_scan_balanced_shell_text(path, text))
    return failures


def audit_scripts(paths: list[Path] | None = None) -> list[str]:
    return [
        failure
        for path in (paths or LEONARDO_SHELL_SCRIPTS)
        for failure in audit_script(path)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Leonardo Bash scripts without requiring Bash locally.")
    parser.add_argument(
        "--script",
        type=Path,
        action="append",
        default=[],
        help="Audit one script path. Repeat to audit multiple paths. Defaults to the Leonardo launch scripts.",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "leonardo_shell_audit.json")
    args = parser.parse_args()

    scripts = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in (args.script or LEONARDO_SHELL_SCRIPTS)
    ]
    failures = audit_scripts(scripts)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "scripts": [_script_label(path) for path in scripts],
        "script_files": [_script_file_row(path) for path in scripts],
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    if failures:
        print("Leonardo shell audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Leonardo shell audit passed")


if __name__ == "__main__":
    main()
