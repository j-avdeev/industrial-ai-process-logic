from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT
from .run_profiles import profile_for_count


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _parse_kv(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            parsed[value] = ""
            continue
        key, item = value.split("=", 1)
        parsed[key] = item
    return parsed


def _run_profile(parameters: dict[str, str]) -> str:
    try:
        count = int(parameters.get("COUNT_PER_FAMILY", ""))
    except (TypeError, ValueError):
        return ""
    return profile_for_count(count)


def _source_bundle_evidence(artifacts_dir: Path) -> dict[str, object]:
    readiness_path = artifacts_dir / "leonardo_readiness.json"
    if not readiness_path.exists():
        return {}
    try:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "readiness_path": str(readiness_path),
            "readiness_readable": False,
            "readiness_error": str(exc),
        }
    source_bundle = readiness.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    return {
        "readiness_path": str(readiness_path),
        "readiness_readable": True,
        "readiness_passed": readiness.get("passed"),
        "require_source_bundle": readiness.get("require_source_bundle"),
        "bundle_path": source_bundle.get("bundle_path", ""),
        "bundle_sha256": source_bundle.get("bundle_sha256", ""),
        "verified": source_bundle.get("verified"),
        "manifest_path": source_bundle.get("manifest_path", ""),
        "manifest_source": source_bundle.get("manifest_source", ""),
        "manifest_file_count": source_bundle.get("manifest_file_count", 0),
        "failures": source_bundle.get("failures", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a manifest for a training/inference run.")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "run_manifest.json")
    parser.add_argument("--events-out", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--stage", default="snapshot")
    parser.add_argument("--set", action="append", default=[], help="Add a key=value run parameter.")
    args = parser.parse_args()

    slurm_env_keys = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
        "SLURM_GPUS",
    ]
    artifacts_dir = args.artifacts_dir or args.out.parent
    checkpoint_dir = args.checkpoint_dir or PROJECT_ROOT / "checkpoints"
    submission_dir = args.submission_dir or PROJECT_ROOT / "submissions"
    parameters = _parse_kv(args.set)
    manifest = {
        "stage": args.stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "parameters": parameters,
        "run_profile": _run_profile(parameters),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "status_short": _git_value(["status", "--short"]),
        },
        "slurm": {
            key: os.environ[key]
            for key in slurm_env_keys
            if key in os.environ
        },
        "source_bundle": _source_bundle_evidence(artifacts_dir),
        "artifacts": {
            "leonardo_shell_audit": str(artifacts_dir / "leonardo_shell_audit.json"),
            "leonardo_readiness": str(artifacts_dir / "leonardo_readiness.json"),
            "leonardo_launch_commands": str(artifacts_dir / "leonardo_launch_commands.sh"),
            "preflight": str(artifacts_dir / "preflight_full_pipeline.json"),
            "corpus_audit": str(artifacts_dir / "corpus_audit" / "summary.json"),
            "checkpoint_audit": str(artifacts_dir / "checkpoint_audit.json"),
            "reranker_metrics": str(artifacts_dir / "reranker_compare" / "metrics.json"),
            "reranker_report": str(artifacts_dir / "reranker_compare" / "REPORT.md"),
            "validation_summary": str(artifacts_dir / "validation_summary.json"),
            "submission_package": str(artifacts_dir / "submission_package" / "track1_submission.zip"),
            "submission_package_sha256": str(artifacts_dir / "submission_package" / "track1_submission.zip.sha256"),
            "package_manifest": str(artifacts_dir / "submission_package" / "package_manifest.json"),
            "submissions": str(submission_dir),
            "checkpoints": str(checkpoint_dir),
        },
    }
    events_out = args.events_out or (args.out.parent / "run_manifest_events.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    events_out.parent.mkdir(parents=True, exist_ok=True)
    with events_out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    print(f"Appended {events_out}")


if __name__ == "__main__":
    main()
