from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import FAMILY_FILES
from .hashing import file_sha256
from .official import load_generator
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


EXPECTED_EVAL_COLUMNS = {
    "valid": ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
    "anomaly": ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
}


def _check_data_files(data_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for family, filename in FAMILY_FILES.items():
        path = data_dir / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append({"family": family, "path": str(path), "exists": exists, "bytes": size})
        if not exists:
            failures.append(f"Missing training data file: {path}")
        elif size == 0:
            failures.append(f"Training data file is empty: {path}")
    generator_path = data_dir / "generate_sequences.py"
    if not generator_path.exists():
        failures.append(f"Missing official generator: {generator_path}")
    return rows, failures


def _check_torch(require_torch: bool, require_cuda: bool) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    info: dict[str, object] = {"required": require_torch, "cuda_required": require_cuda}
    try:
        import torch
    except ImportError:
        info["available"] = False
        if require_torch or require_cuda:
            failures.append("PyTorch is not importable")
        return info, failures

    cuda_available = bool(torch.cuda.is_available())
    info.update({
        "available": True,
        "version": getattr(torch, "__version__", ""),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", "") or "",
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "devices": [],
    })
    if cuda_available:
        devices = []
        for index in range(int(torch.cuda.device_count())):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": int(props.total_memory),
            })
        info["current_device"] = int(torch.cuda.current_device())
        info["devices"] = devices
    if require_cuda:
        if not cuda_available:
            failures.append("CUDA was required but torch.cuda.is_available() is false")
        elif int(info.get("cuda_device_count", 0)) <= 0:
            failures.append("CUDA was required but no CUDA devices were reported")
        elif not info.get("devices"):
            failures.append("CUDA was required but device details were not reported")
    return info, failures


def _eval_input_status(label: str, path: Path | None, require_eval: bool) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    expected_columns = EXPECTED_EVAL_COLUMNS[label]
    if path is None:
        status = {
            "label": label,
            "path": "",
            "exists": False,
            "bytes": 0,
            "rows": 0,
            "fieldnames": [],
            "required_columns": expected_columns,
            "missing_columns": expected_columns,
        }
        if require_eval:
            failures.append(f"Missing required {label} eval input path")
        return status, failures
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    row_count = 0
    fieldnames: list[str] = []
    if exists and size > 0:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                row_count = sum(1 for _ in reader)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            failures.append(f"Eval input is not readable CSV for {label}: {path} ({exc})")
    status = {
        "label": label,
        "path": str(path),
        "exists": exists,
        "bytes": size,
        "rows": row_count,
        "sha256": file_sha256(path) if exists and size > 0 else "",
        "fieldnames": fieldnames,
        "required_columns": expected_columns,
        "missing_columns": [column for column in expected_columns if column not in fieldnames],
    }
    if require_eval:
        if not exists:
            failures.append(f"Missing required {label} eval input: {path}")
        elif size == 0:
            failures.append(f"Required {label} eval input is empty: {path}")
        elif not fieldnames:
            failures.append(f"Required {label} eval input has no CSV header: {path}")
        elif status["missing_columns"]:
            failures.append(
                f"Required {label} eval input is missing columns: "
                + ", ".join(str(column) for column in status["missing_columns"])
            )
        elif row_count <= 0:
            failures.append(f"Required {label} eval input has no rows: {path}")
    return status, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight checks for local and Leonardo runs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "preflight.json")
    parser.add_argument("--valid-input", type=Path)
    parser.add_argument("--anomaly-input", type=Path)
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    data_files, data_failures = _check_data_files(args.data_dir)
    failures.extend(data_failures)

    generator_ok = False
    try:
        load_generator(args.data_dir)
        generator_ok = True
    except Exception as exc:
        failures.append(f"Official generator failed to load: {exc}")

    eval_inputs = []
    for label, path in (("valid", args.valid_input), ("anomaly", args.anomaly_input)):
        status, eval_failures = _eval_input_status(label, path, args.require_eval)
        if path is not None or args.require_eval:
            eval_inputs.append(status)
        failures.extend(eval_failures)

    torch_info, torch_failures = _check_torch(args.require_torch, args.require_cuda)
    failures.extend(torch_failures)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "data_dir": str(args.data_dir),
        "data_files": data_files,
        "official_generator_loads": generator_ok,
        "require_eval": args.require_eval,
        "eval_inputs": eval_inputs,
        "torch": torch_info,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    if failures:
        print("Preflight failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Preflight passed")


if __name__ == "__main__":
    main()
