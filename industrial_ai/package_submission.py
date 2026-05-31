from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .hashing import file_sha256
from .paths import PROJECT_ROOT
from .run_profiles import profile_for_count
from .train import MODEL_CONFIGS
from .validate_run import REQUIRED_SUBMISSIONS

LEONARDO_SCRIPT_NAMES = [
    "leonardo_common.sh",
    "leonardo_probe.sh",
    "leonardo_generate.sh",
    "leonardo_train.sh",
    "leonardo_train_scaling.sh",
    "leonardo_infer.sh",
    "leonardo_finalize.sh",
    "leonardo_full_pipeline.sh",
]
SOURCE_SNAPSHOT_PREFIX = Path("source_snapshot")


def _source_bundle_source_paths(readiness_payload: dict[str, object]) -> list[str]:
    rows = _source_bundle_manifest_rows(readiness_payload)
    return sorted(
        rel_path
        for rel_path in rows
        if rel_path.startswith("industrial_ai/")
    )


def _source_bundle_snapshot_files(readiness_payload: dict[str, object]) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for rel_path in _source_bundle_source_paths(readiness_payload):
        path = Path(rel_path)
        if path.is_absolute() or ".." in path.parts:
            continue
        files.append((PROJECT_ROOT / path, SOURCE_SNAPSHOT_PREFIX / path))
    return files


def _evidence_files(
    artifacts_dir: Path,
    checkpoint_dir: Path,
    generated_dir: Path,
    readiness_payload: dict[str, object] | None = None,
) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []

    for name in LEONARDO_SCRIPT_NAMES:
        files.append((PROJECT_ROOT / "scripts" / name, Path("scripts") / name))
    if readiness_payload:
        files.extend(_source_bundle_snapshot_files(readiness_payload))
    for src in sorted(artifacts_dir.glob("preflight*.json")):
        files.append((src, Path(src.name)))
    files.append((artifacts_dir / "eval_staging_manifest.json", Path("eval_staging_manifest.json")))
    for name in (
        "leonardo_shell_audit.json",
        "leonardo_readiness.json",
        "leonardo_launch_commands.sh",
        "source_bundle_proof_selftest.json",
    ):
        files.append((artifacts_dir / name, Path(name)))
    files.append((artifacts_dir / "checkpoint_audit.json", Path("checkpoint_audit.json")))
    for name in ("run_manifest.json", "run_manifest_events.jsonl"):
        files.append((artifacts_dir / name, Path(name)))
    files.append((artifacts_dir / "validation_summary.json", Path("validation_summary.json")))

    for subdir, names in {
        "corpus_audit": ["summary.json", "families.csv", "files.csv"],
        "completion_compare": ["metrics.json", "metrics.csv"],
        "reranker_compare": [
            "metrics.json",
            "metrics.csv",
            "family_metrics.csv",
            "REPORT.md",
            "best_checkpoint.txt",
        ],
    }.items():
        for name in names:
            files.append((artifacts_dir / subdir / name, Path(subdir) / name))
    for src in sorted((artifacts_dir / "reranker_compare").glob("*.png")):
        files.append((src, Path("reranker_compare") / src.name))

    if checkpoint_dir.exists():
        for src in sorted(checkpoint_dir.glob("*/train_summary.json")):
            files.append((src, Path("checkpoints") / src.parent.name / src.name))
        for src in sorted(checkpoint_dir.glob("*/train_log.json")):
            files.append((src, Path("checkpoints") / src.parent.name / src.name))
        for src in sorted(checkpoint_dir.glob("*/loss_curve.png")):
            files.append((src, Path("checkpoints") / src.parent.name / src.name))

    if generated_dir.exists():
        for src in sorted(generated_dir.glob("*.metadata.json")):
            files.append((src, Path("generated_metadata") / src.name))

    return files


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _copy_with_manifest(src: Path, dst: Path) -> dict[str, object]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
        "rows": _csv_count(dst) if dst.suffix.lower() == ".csv" else None,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _event_log_has_stage(path: Path, stage: str) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("stage") == stage:
                    return True
    except OSError:
        return False
    return False


def _event_log_last_stage_payload(path: Path, stage: str) -> dict[str, object]:
    found: dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("stage") == stage and isinstance(payload, dict):
                    found = payload
    except OSError:
        return {}
    return found


def _event_log_stage_positions(path: Path) -> dict[str, int]:
    positions: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = str(payload.get("stage", "") or "")
                if stage:
                    positions[stage] = index
    except OSError:
        return {}
    return positions


def _expected_run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    count = max_generated_per_family or min_generated_per_family
    if count <= 0:
        return ""
    return profile_for_count(count)


def _expected_completion_checkpoint(required_completion_checkpoint_size: str) -> str:
    size = str(required_completion_checkpoint_size or "").strip()
    if not size:
        return ""
    return f"checkpoints/{size}/model.pt"


def _manifest_completion_checkpoint_failures(
    payload: dict[str, object],
    expected_checkpoint: str,
    label: str,
) -> list[str]:
    if not expected_checkpoint:
        return []
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        return [f"{label} has no parameters object"]
    actual = str(parameters.get("COMPLETION_CHECKPOINT", "") or "").strip().replace("\\", "/")
    if not actual:
        return [f"{label} did not record COMPLETION_CHECKPOINT"]
    if actual != expected_checkpoint:
        return [f"{label} COMPLETION_CHECKPOINT is {actual!r}; expected {expected_checkpoint!r}"]
    return []


def _manifest_parameter_failures(
    payload: dict[str, object],
    label: str,
    expected_run_profile: str,
    expected_count_per_family: int,
    expected_checkpoint: str,
    require_eval: bool,
    require_source_bundle: bool,
) -> list[str]:
    failures: list[str] = []
    if expected_run_profile:
        actual_profile = str(payload.get("run_profile", "") or "")
        if actual_profile != expected_run_profile:
            failures.append(f"{label} run_profile is {actual_profile!r}; expected {expected_run_profile!r}")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        return [*failures, f"{label} has no parameters object"]
    if expected_count_per_family:
        actual_count = str(parameters.get("COUNT_PER_FAMILY", "") or "")
        if actual_count != str(expected_count_per_family):
            failures.append(
                f"{label} COUNT_PER_FAMILY is {actual_count!r}; expected {expected_count_per_family}"
            )
    if require_eval:
        actual_require_eval = str(parameters.get("REQUIRE_EVAL", "") or "")
        if actual_require_eval != "1":
            failures.append(f"{label} REQUIRE_EVAL is {actual_require_eval!r}; expected '1'")
    if require_source_bundle:
        actual_require_source = str(parameters.get("REQUIRE_SOURCE_BUNDLE", "") or "")
        if actual_require_source != "1":
            failures.append(f"{label} REQUIRE_SOURCE_BUNDLE is {actual_require_source!r}; expected '1'")
    if expected_checkpoint:
        failures.extend(_manifest_completion_checkpoint_failures(payload, expected_checkpoint, label))
    return failures


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _rows_by_label(payload: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("label", "") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("label", "") or "")
    }


def _source_bundle_summary(readiness_payload: dict[str, object]) -> dict[str, object]:
    source_bundle = readiness_payload.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    return {
        "required": readiness_payload.get("require_source_bundle"),
        "readiness_passed": readiness_payload.get("passed"),
        "verified": source_bundle.get("verified"),
        "bundle_sha256": source_bundle.get("bundle_sha256", ""),
        "bundle_path": source_bundle.get("bundle_path", ""),
        "manifest_source": source_bundle.get("manifest_source", ""),
        "manifest_file_count": source_bundle.get("manifest_file_count", 0),
        "manifest_files": source_bundle.get("manifest_files", []),
    }


def _manifest_source_bundle_failures(
    payload: dict[str, object],
    readiness_payload: dict[str, object],
    label: str,
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    source_bundle = readiness_payload.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        return ["Leonardo readiness requires source bundle but has no source_bundle object"]
    expected_hash = str(source_bundle.get("bundle_sha256", "") or "")
    if not _valid_sha256(expected_hash):
        return ["Leonardo readiness source_bundle has no valid bundle_sha256"]

    manifest_source_bundle = payload.get("source_bundle", {})
    if not isinstance(manifest_source_bundle, dict) or not manifest_source_bundle:
        return [f"{label} did not record source_bundle evidence from Leonardo readiness"]
    failures: list[str] = []
    actual_hash = str(manifest_source_bundle.get("bundle_sha256", "") or "")
    if actual_hash != expected_hash:
        failures.append(
            f"{label} source_bundle.bundle_sha256 is {actual_hash!r}; "
            f"expected readiness hash {expected_hash!r}"
        )
    if manifest_source_bundle.get("require_source_bundle") is not True:
        failures.append(f"{label} source_bundle did not record require_source_bundle=true")
    if manifest_source_bundle.get("verified") is not True:
        failures.append(f"{label} source_bundle did not record verified=true")
    if manifest_source_bundle.get("readiness_passed") is not True:
        failures.append(f"{label} source_bundle did not record readiness_passed=true")
    return failures


def _source_bundle_manifest_rows(readiness_payload: dict[str, object]) -> dict[str, dict[str, object]]:
    source_bundle = readiness_payload.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        return {}
    rows = source_bundle.get("manifest_files", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("path", "") or "").replace("\\", "/"): row
        for row in rows
        if isinstance(row, dict)
    }


def _source_bundle_current_file_failures(
    readiness_payload: dict[str, object],
    rel_paths: list[str],
    label: str,
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    rows = _source_bundle_manifest_rows(readiness_payload)
    if not rows:
        return [f"{label} cannot compare files to source bundle; readiness source_bundle has no manifest_files list"]
    failures: list[str] = []
    for rel_path in rel_paths:
        normalized = rel_path.replace("\\", "/")
        row = rows.get(normalized)
        if not row:
            failures.append(f"{label} source bundle manifest missing file: {normalized}")
            continue
        expected_hash = str(row.get("sha256", "") or "")
        if not _valid_sha256(expected_hash):
            failures.append(f"{label} source bundle manifest has invalid hash for {normalized}")
            continue
        path = PROJECT_ROOT / normalized
        if not path.exists():
            failures.append(f"{label} current file missing for source-bundle comparison: {normalized}")
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            failures.append(
                f"{label} current file hash for {normalized} does not match source bundle: "
                f"{actual_hash} != {expected_hash}"
            )
    return failures


def _resolve_optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _check_file_hash(path: Path, expected_sha256: object, label: str) -> list[str]:
    expected = str(expected_sha256 or "").strip()
    if not expected:
        return [f"Missing expected hash for {label}: {path}"]
    if not path.exists():
        return [f"Missing file for hash check of {label}: {path}"]
    actual = file_sha256(path)
    if actual != expected:
        return [f"Hash mismatch for {label}: {path} expected {expected}, got {actual}"]
    return []


def _check_preflight_eval_evidence(path: Path) -> list[str]:
    if not path.exists():
        return [f"Missing preflight evidence for eval check: {path}"]
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Preflight evidence is not readable JSON: {path} ({exc})"]
    failures: list[str] = []
    if payload.get("require_eval") is not True:
        failures.append("Preflight evidence did not require eval inputs")
    eval_inputs = payload.get("eval_inputs", [])
    if not isinstance(eval_inputs, list) or not eval_inputs:
        failures.append("Preflight evidence did not record eval input checks")
        return failures
    for row in eval_inputs:
        if not isinstance(row, dict):
            failures.append(f"Preflight eval input row is not an object: {row!r}")
            continue
        if row.get("exists") is not True:
            failures.append(f"Preflight evidence missing eval input: {row.get('path')}")
        elif int(row.get("bytes", 0) or 0) <= 0:
            failures.append(f"Preflight evidence saw empty eval input: {row.get('path')}")
        elif row.get("missing_columns"):
            failures.append(
                "Preflight evidence saw eval input with missing columns: "
                f"{row.get('path')} ({', '.join(map(str, row.get('missing_columns', [])))})"
            )
        elif int(row.get("rows", 0) or 0) <= 0:
            failures.append(f"Preflight evidence saw eval input with no rows: {row.get('path')}")
    return failures


def _check_eval_staging_evidence(
    artifacts_dir: Path,
    readiness_payload: dict[str, object],
) -> list[str]:
    staging_path = artifacts_dir / "eval_staging_manifest.json"
    preflight_path = artifacts_dir / "preflight_full_pipeline.json"
    failures: list[str] = []
    try:
        staging_payload = _read_json(staging_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Eval staging manifest is not readable JSON: {staging_path} ({exc})"]
    try:
        preflight_payload = _read_json(preflight_path)
    except (OSError, json.JSONDecodeError) as exc:
        preflight_payload = {}
        failures.append(f"Preflight evidence is not readable JSON for eval staging check: {preflight_path} ({exc})")

    if readiness_payload.get("require_eval") is not True:
        failures.append("Leonardo readiness did not require eval inputs")
    if readiness_payload.get("defer_eval_staging") is True:
        failures.append("Leonardo readiness still has defer_eval_staging=true")
    if staging_payload.get("passed") is not True:
        failures.append("Eval staging manifest did not pass")
    staging_failures = staging_payload.get("failures", [])
    if isinstance(staging_failures, list) and staging_failures:
        failures.append("Eval staging manifest recorded failures: " + "; ".join(map(str, staging_failures)))

    destinations = _rows_by_label(staging_payload, "destinations")
    readiness_rows = _rows_by_label(readiness_payload, "eval_inputs")
    preflight_rows = _rows_by_label(preflight_payload, "eval_inputs") if preflight_payload else {}
    for label in ("valid", "anomaly"):
        row = destinations.get(label)
        if row is None:
            failures.append(f"Eval staging manifest missing {label} destination row")
            continue
        staged_hash = str(row.get("sha256", "") or "")
        if not _valid_sha256(staged_hash):
            failures.append(f"Eval staging manifest has invalid {label} SHA-256")
        if row.get("exists") is not True:
            failures.append(f"Eval staging manifest says {label} destination is missing")
        if _as_int(row.get("rows", 0)) <= 0:
            failures.append(f"Eval staging manifest says {label} destination has no rows")
        if row.get("missing_columns"):
            failures.append(
                f"Eval staging manifest says {label} destination is missing columns: "
                + ", ".join(map(str, row.get("missing_columns", [])))
            )

        readiness_row = readiness_rows.get(label)
        if readiness_row is None:
            failures.append(f"Leonardo readiness missing {label} eval input row")
        else:
            readiness_hash = str(readiness_row.get("sha256", "") or "")
            if not _valid_sha256(readiness_hash):
                failures.append(f"Leonardo readiness has invalid {label} eval SHA-256")
            if readiness_hash != staged_hash:
                failures.append(
                    f"Leonardo readiness {label} eval SHA-256 does not match staging manifest: "
                    f"{readiness_hash or 'missing'} != {staged_hash or 'missing'}"
                )

        preflight_row = preflight_rows.get(label)
        if preflight_row is None:
            failures.append(f"Preflight evidence missing {label} eval input row")
        else:
            preflight_hash = str(preflight_row.get("sha256", "") or "")
            if not _valid_sha256(preflight_hash):
                failures.append(f"Preflight evidence has invalid {label} eval SHA-256")
            if preflight_hash != staged_hash:
                failures.append(
                    f"Preflight evidence {label} eval SHA-256 does not match staging manifest: "
                    f"{preflight_hash or 'missing'} != {staged_hash or 'missing'}"
                )
    return failures


def _find_checkpoint_run(reranker_payload: dict[str, object], checkpoint: Path) -> dict[str, object] | None:
    for run in reranker_payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_checkpoint = _resolve_optional_path(run.get("checkpoint"))
        if run_checkpoint == checkpoint:
            return run
    return None


def _find_reranker_run(reranker_payload: dict[str, object]) -> dict[str, object] | None:
    best_reranker = str(reranker_payload.get("best_reranker", "") or "")
    for run in reranker_payload.get("runs", []):
        if isinstance(run, dict) and str(run.get("reranker", "") or "") == best_reranker:
            return run
    return None


def _as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _best_reranker_score_failures(
    reranker_payload: dict[str, object],
    best_run: dict[str, object] | None,
) -> list[str]:
    runs = reranker_payload.get("runs", [])
    if not isinstance(runs, list):
        return ["Reranker metrics runs is not a list"]
    eligible_runs = [run for run in runs if isinstance(run, dict) and _truthy(run.get("selection_eligible", False))]
    if not eligible_runs:
        return ["Reranker metrics has no selection-eligible runs"]
    if best_run is None:
        return []
    best_score = _as_float(best_run.get("selection_score", 0.0))
    max_score = max(_as_float(run.get("selection_score", 0.0)) for run in eligible_runs)
    if best_score + 1e-12 < max_score:
        return [
            "Selected reranker is not the highest-scoring eligible run: "
            f"{best_score:.6g} < {max_score:.6g}"
        ]
    return []


def _required_checkpoint_reranker_failures(
    reranker_payload: dict[str, object],
    required_sizes: list[str],
) -> list[str]:
    runs_by_label = {
        str(run.get("reranker", "") or ""): run
        for run in reranker_payload.get("runs", [])
        if isinstance(run, dict)
    }
    failures: list[str] = []
    for size in required_sizes:
        run = runs_by_label.get(size)
        if run is None:
            failures.append(f"Reranker metrics missing required checkpoint run: {size}")
            continue
        checkpoint = _resolve_optional_path(run.get("checkpoint"))
        if checkpoint is None:
            failures.append(f"Reranker metrics required run {size} has no checkpoint")
        elif checkpoint.parent.name != size:
            failures.append(
                f"Reranker metrics required run {size} uses checkpoint size {checkpoint.parent.name!r}"
            )
        elif not checkpoint.exists():
            failures.append(f"Reranker metrics required run {size} checkpoint does not exist: {checkpoint}")
        if not _truthy(run.get("available", False)):
            failures.append(f"Reranker metrics required run {size} was not available")
        if not _truthy(run.get("selection_eligible", False)):
            failures.append(f"Reranker metrics required run {size} was not selection eligible")
        checkpoint_sha256 = str(run.get("checkpoint_sha256", "") or "").strip()
        if not checkpoint_sha256:
            failures.append(f"Reranker metrics required run {size} has no checkpoint_sha256")
        elif checkpoint is not None:
            failures.extend(_check_file_hash(
                checkpoint,
                checkpoint_sha256,
                f"required reranker {size} checkpoint against reranker metrics",
            ))
    return failures


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _clear_previous_package(out_dir: Path) -> None:
    filenames = [
        "track1_submission.zip",
        "track1_submission.zip.sha256",
        ".track1_submission.zip.tmp",
        "package_manifest.json",
        *REQUIRED_SUBMISSIONS,
    ]
    for filename in filenames:
        try:
            (out_dir / filename).unlink()
        except FileNotFoundError:
            pass
    evidence_dir = out_dir / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)


def _required_evidence_paths(
    checkpoint_sizes: list[str],
    require_generated_metadata: bool,
    require_readiness: bool,
    require_eval_staging: bool,
) -> set[Path]:
    required = {
        Path("preflight_full_pipeline.json"),
        Path("leonardo_shell_audit.json"),
        Path("inference_summary.json"),
        Path("run_manifest.json"),
        Path("run_manifest_events.jsonl"),
        Path("validation_summary.json"),
        Path("corpus_audit") / "summary.json",
        Path("corpus_audit") / "families.csv",
        Path("corpus_audit") / "files.csv",
        Path("completion_compare") / "metrics.json",
        Path("completion_compare") / "metrics.csv",
        Path("reranker_compare") / "metrics.json",
        Path("reranker_compare") / "metrics.csv",
        Path("reranker_compare") / "family_metrics.csv",
        Path("reranker_compare") / "REPORT.md",
        Path("reranker_compare") / "best_checkpoint.txt",
    }
    for size in checkpoint_sizes:
        required.add(Path("checkpoints") / size / "train_summary.json")
        required.add(Path("checkpoints") / size / "train_log.json")
    if require_generated_metadata:
        for family in ("IC", "IGBT", "MOSFET"):
            required.add(Path("generated_metadata") / f"{family}_extra.csv.metadata.json")
    if require_readiness:
        required.add(Path("leonardo_readiness.json"))
        required.add(Path("leonardo_launch_commands.sh"))
        required.add(Path("source_bundle_proof_selftest.json"))
        required.add(Path("checkpoint_audit.json"))
    if require_eval_staging:
        required.add(Path("eval_staging_manifest.json"))
    for name in LEONARDO_SCRIPT_NAMES:
        required.add(Path("scripts") / name)
    return required


def _check_readiness_launch_command_script(artifacts_dir: Path) -> list[str]:
    readiness_path = artifacts_dir / "leonardo_readiness.json"
    script_path = artifacts_dir / "leonardo_launch_commands.sh"
    failures: list[str] = []
    try:
        readiness_payload = _read_json(readiness_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Leonardo readiness is not readable JSON: {readiness_path} ({exc})"]

    commands_out = str(readiness_payload.get("commands_out", "") or "").replace("\\", "/")
    if Path(commands_out).name != "leonardo_launch_commands.sh":
        failures.append(
            "Leonardo readiness commands_out is "
            f"{commands_out!r}; expected leonardo_launch_commands.sh"
        )
    if readiness_payload.get("defer_eval_staging") is True:
        failures.append("Leonardo readiness still has defer_eval_staging=true; rerun readiness after staging official eval CSVs")
    try:
        launch_text = script_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return [*failures, f"Leonardo launch command script is not readable: {script_path} ({exc})"]

    for needle, label in {
        "sbatch scripts/leonardo_probe.sh": "probe launch command",
        "scripts/leonardo_full_pipeline.sh": "full-pipeline launch command",
        "--dependency=afterok:${PROBE_JOB}": "generation dependency on probe",
        "--dependency=afterok:${GEN_JOB}": "training dependency on generation",
        "--dependency=afterok:${TRAIN_JOB}": "finalization dependency on training",
        "verify_returned_package": "returned-package verification command",
        "run_evidence_report": "objective evidence-report command",
    }.items():
        if needle not in launch_text:
            failures.append(f"Leonardo launch command script missing {label}")

    readiness_commands = readiness_payload.get("commands", {})
    command_text = ""
    if isinstance(readiness_commands, dict):
        command_text = "\n".join(
            str(command)
            for command_list in readiness_commands.values()
            if isinstance(command_list, list)
            for command in command_list
        )
        for key, label in (
            ("full_pipeline", "recorded full-pipeline launch command"),
            ("split_jobs_with_dependencies", "recorded dependency-safe split launch command"),
        ):
            command_list = readiness_commands.get(key, [])
            if isinstance(command_list, list):
                for command in command_list:
                    if str(command) and str(command) not in launch_text:
                        failures.append(
                            f"Leonardo launch command script is missing {label}: {command}"
                        )
    if readiness_payload.get("require_source_bundle") is True:
        for needle, label in {
            "REQUIRE_SOURCE_BUNDLE=1": "required source-bundle proof export",
            "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh": "split generation source-bundle export",
            "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh": "split training source-bundle export",
        }.items():
            if needle not in launch_text:
                failures.append(f"Leonardo launch command script missing {label}")
            if command_text and needle not in command_text:
                failures.append(f"Leonardo readiness launch commands missing {label}")
        if "source_bundle_proof_selftest" not in launch_text:
            failures.append("Leonardo launch command script missing source-bundle proof self-test command")
    if readiness_payload.get("require_eval") is True and "REQUIRE_EVAL=1" not in launch_text:
        failures.append("Leonardo launch command script missing required eval export")
    if readiness_payload.get("require_eval") is True and isinstance(readiness_commands, dict):
        if "REQUIRE_EVAL=1" not in command_text:
            failures.append("Leonardo readiness launch commands missing required eval export")
    readiness_batch_size = int(readiness_payload.get("batch_size", 0) or 0)
    if readiness_batch_size > 0:
        batch_export = f"BATCH_SIZE={readiness_batch_size}"
        if batch_export not in launch_text:
            failures.append(f"Leonardo launch command script missing batch-size export: {batch_export}")
        if command_text and batch_export not in command_text:
            failures.append(f"Leonardo readiness launch commands missing batch-size export: {batch_export}")

    verification_commands = readiness_payload.get("verification_commands", [])
    if not isinstance(verification_commands, list) or not verification_commands:
        failures.append("Leonardo readiness has no verification_commands list")
    else:
        verify_commands = [
            str(command)
            for command in verification_commands
            if "verify_returned_package" in str(command)
        ]
        evidence_report_commands = [
            str(command)
            for command in verification_commands
            if "run_evidence_report" in str(command)
        ]
        return_packet_commands = [
            str(command)
            for command in verification_commands
            if "leonardo_return_packet" in str(command)
        ]
        if readiness_batch_size > 0:
            batch_flag = f"--required-batch-size {readiness_batch_size}"
            if not any(batch_flag in command for command in verify_commands):
                failures.append(f"Leonardo readiness verify_returned_package command missing batch-size proof: {batch_flag}")
            if not any(batch_flag in command for command in evidence_report_commands):
                failures.append(f"Leonardo readiness run_evidence_report command missing batch-size proof: {batch_flag}")
        if not any("--require-final-leonardo-objective" in command for command in verify_commands):
            failures.append("Leonardo readiness verify_returned_package command missing final Leonardo objective gate")
        if not return_packet_commands:
            failures.append("Leonardo readiness verification commands missing leonardo_return_packet command")
        elif not any("--require-final-leonardo-objective" in command for command in return_packet_commands):
            failures.append("Leonardo readiness leonardo_return_packet command missing final Leonardo objective gate")
        if readiness_payload.get("require_source_bundle") is True:
            if not any("source_bundle_proof_selftest" in str(command) for command in verification_commands):
                failures.append("Leonardo readiness verification commands missing source-bundle proof self-test")
            if not any("--require-source-bundle-proof" in command for command in verify_commands):
                failures.append("Leonardo readiness verify_returned_package command missing source-bundle proof flag")
            report_needles = {
                "--require-readiness": "readiness evidence flag",
                "--require-source-bundle-proof": "source-bundle proof flag",
                "--prefer-package-evidence": "package evidence precedence flag",
            }
            for needle, label in report_needles.items():
                if not any(needle in command for command in evidence_report_commands):
                    failures.append(f"Leonardo readiness run_evidence_report command missing {label}")
        else:
            if verify_commands and not any("--no-require-source-bundle-proof" in command for command in verify_commands):
                failures.append(
                    "Leonardo readiness verify_returned_package command missing explicit no-source-bundle proof flag"
                )
            if evidence_report_commands and not any(
                "--no-require-source-bundle-proof" in command for command in evidence_report_commands
            ):
                failures.append(
                    "Leonardo readiness run_evidence_report command missing explicit no-source-bundle proof flag"
                )
        for command in verification_commands:
            if str(command) not in launch_text:
                failures.append(
                    "Leonardo launch command script is missing recorded verification command: "
                    f"{command}"
                )
    return failures


def _check_source_bundle_selftest_evidence(artifacts_dir: Path) -> list[str]:
    path = artifacts_dir / "source_bundle_proof_selftest.json"
    if not path.exists():
        return [f"Missing source-bundle proof self-test evidence: {path}"]
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Source-bundle proof self-test evidence is not readable JSON: {path} ({exc})"]
    failures: list[str] = []
    if payload.get("passed") is not True:
        failures.append(f"Source-bundle proof self-test did not pass: {path}")
    selftest_failures = payload.get("failures", [])
    if isinstance(selftest_failures, list) and selftest_failures:
        failures.append(
            "Source-bundle proof self-test recorded failures: "
            + "; ".join(map(str, selftest_failures))
        )
    return failures


def _zip_arcname(path: Path, out_dir: Path) -> str:
    try:
        return str(path.relative_to(out_dir)).replace("\\", "/")
    except ValueError:
        return path.name.replace("\\", "/")


def _verify_zip(
    zip_path: Path,
    out_dir: Path,
    manifest_path: Path,
    files: list[dict[str, object]],
    evidence: list[dict[str, object]],
) -> list[str]:
    expected: dict[str, str] = {}
    for item in files:
        path = Path(str(item["path"]))
        expected[path.name] = str(item["sha256"])
    expected[manifest_path.name] = _sha256(manifest_path)
    for item in evidence:
        path = Path(str(item["path"]))
        expected[_zip_arcname(path, out_dir)] = str(item["sha256"])

    failures: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        duplicates = sorted({name for name in names if names.count(name) > 1})
        for name in duplicates:
            failures.append(f"ZIP has duplicate entry: {name}")
        actual = set(names)
        expected_names = set(expected)
        for name in sorted(expected_names - actual):
            failures.append(f"ZIP missing expected entry: {name}")
        for name in sorted(actual - expected_names):
            failures.append(f"ZIP has unexpected entry: {name}")
        for name, digest in sorted(expected.items()):
            if name not in actual:
                continue
            if _sha256_bytes(zf.read(name)) != digest:
                failures.append(f"ZIP entry hash mismatch: {name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Track 1 submission CSVs with run evidence.")
    parser.add_argument("--submission-dir", type=Path, default=PROJECT_ROOT / "submissions")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--generated-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "submission_package")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument(
        "--require-evidence",
        action="store_true",
        help="Fail if required evidence files are missing. Implies --include-evidence.",
    )
    parser.add_argument(
        "--required-checkpoint-sizes",
        nargs="*",
        default=[],
        help="Checkpoint sizes whose train_summary.json and train_log.json must be packaged when --require-evidence is set.",
    )
    parser.add_argument(
        "--required-completion-checkpoint-size",
        default="",
        help="Require completion comparison evidence to use this model-size checkpoint. Defaults to the last required checkpoint size in strict evidence mode.",
    )
    parser.add_argument(
        "--required-min-generated-per-family",
        type=int,
        default=0,
        help="Minimum min_generated_per_family that validation_summary.json must report in strict evidence mode.",
    )
    parser.add_argument(
        "--required-max-generated-per-family",
        type=int,
        default=0,
        help="Exact max_generated_per_family that validation_summary.json must report in strict evidence mode. Zero disables the check.",
    )
    parser.add_argument(
        "--required-min-reranker-count",
        type=int,
        default=0,
        help="Minimum min_reranker_count that validation_summary.json must report in strict evidence mode.",
    )
    parser.add_argument(
        "--required-min-completion-compare-count",
        type=int,
        default=0,
        help="Minimum min_completion_compare_count that validation_summary.json must report in strict evidence mode.",
    )
    parser.add_argument(
        "--required-min-train-epochs",
        type=int,
        default=0,
        help="Minimum min_train_epochs that validation_summary.json must report in strict evidence mode.",
    )
    parser.add_argument(
        "--required-batch-size",
        type=int,
        default=0,
        help="Exact training batch_size that validation/checkpoint/train summaries must report in strict evidence mode. Zero disables the check.",
    )
    parser.add_argument(
        "--required-manifest-stage",
        default="",
        help="Require artifacts/run_manifest.json and the event log to contain this latest stage.",
    )
    parser.add_argument(
        "--require-preflight-cuda",
        action="store_true",
        help="In strict evidence mode, require validation_summary.json to prove CUDA preflight validation.",
    )
    parser.add_argument(
        "--require-preflight-eval",
        action="store_true",
        help="In strict evidence mode, require validation_summary.json to prove eval preflight validation.",
    )
    parser.add_argument(
        "--required-transformer-device",
        default="",
        help="In strict evidence mode, require validation_summary.json to prove this comparison/inference transformer device.",
    )
    parser.add_argument(
        "--require-selected-checkpoint",
        action="store_true",
        help="In strict evidence mode, require validation and reranker metrics to prove a checkpoint reranker was selected.",
    )
    parser.add_argument(
        "--require-generated-metadata",
        action="store_true",
        help="In strict evidence mode, require generated CSV metadata sidecars in package evidence.",
    )
    parser.add_argument(
        "--require-readiness",
        action="store_true",
        help="In strict evidence mode, require leonardo_readiness.json in package evidence.",
    )
    args = parser.parse_args()
    required_completion_checkpoint_size = args.required_completion_checkpoint_size
    if not required_completion_checkpoint_size and args.required_transformer_device and args.required_checkpoint_sizes:
        required_completion_checkpoint_size = str(args.required_checkpoint_sizes[-1])
    expected_completion_checkpoint = _expected_completion_checkpoint(required_completion_checkpoint_size)
    if args.require_evidence:
        args.include_evidence = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_package(args.out_dir)
    files: list[dict[str, object]] = []
    failures: list[str] = []
    readiness_payload: dict[str, object] = {}
    source_bundle_summary: dict[str, object] = {}
    if args.require_readiness:
        readiness_path = args.artifacts_dir / "leonardo_readiness.json"
        try:
            readiness_payload = _read_json(readiness_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Leonardo readiness is not readable JSON: {readiness_path} ({exc})")
        else:
            source_bundle_summary = _source_bundle_summary(readiness_payload)
    for filename in REQUIRED_SUBMISSIONS:
        src = args.submission_dir / filename
        if not src.exists():
            failures.append(f"Missing submission file: {src}")
            continue
        if _csv_count(src) == 0:
            failures.append(f"Submission file has no rows: {src}")
            continue
        files.append(_copy_with_manifest(src, args.out_dir / filename))

    evidence: list[dict[str, object]] = []
    if args.include_evidence:
        evidence_dir = args.out_dir / "evidence"
        copied_evidence_paths: set[Path] = set()
        inference_summary = args.submission_dir / "inference_summary.json"
        if inference_summary.exists():
            evidence.append(_copy_with_manifest(inference_summary, evidence_dir / inference_summary.name))
            copied_evidence_paths.add(Path(inference_summary.name))
        for src, rel_path in _evidence_files(
            args.artifacts_dir,
            args.checkpoint_dir,
            args.generated_dir,
            readiness_payload,
        ):
            if src.exists():
                evidence.append(_copy_with_manifest(src, evidence_dir / rel_path))
                copied_evidence_paths.add(rel_path)
        if args.require_evidence:
            expected_run_profile = _expected_run_profile(
                args.required_min_generated_per_family,
                args.required_max_generated_per_family,
            )
            validation_payload: dict[str, object] = {}
            for rel_path in sorted(_required_evidence_paths(
                args.required_checkpoint_sizes,
                args.require_generated_metadata,
                args.require_readiness,
                args.require_readiness and args.require_preflight_eval,
            )):
                if rel_path not in copied_evidence_paths:
                    failures.append(f"Missing required evidence file: {rel_path}")
            validation_summary = args.artifacts_dir / "validation_summary.json"
            if validation_summary.exists():
                try:
                    validation_payload = _read_json(validation_summary)
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"Validation summary is not readable JSON: {validation_summary} ({exc})")
                else:
                    if validation_payload.get("passed") is not True:
                        failures.append(f"Validation summary did not pass: {validation_summary}")
                    validated_sizes = {str(size) for size in validation_payload.get("model_sizes", [])}
                    missing_validated_sizes = [
                        size for size in args.required_checkpoint_sizes
                        if size not in validated_sizes
                    ]
                    if missing_validated_sizes:
                        failures.append(
                            "Validation summary did not validate required checkpoint sizes: "
                            + ", ".join(missing_validated_sizes)
                        )
                    if validation_payload.get("require_submissions") is not True:
                        failures.append("Validation summary did not require submissions")
                    if args.require_preflight_cuda:
                        if validation_payload.get("require_preflight") is not True:
                            failures.append("Validation summary did not require preflight checks")
                        if validation_payload.get("require_preflight_torch") is not True:
                            failures.append("Validation summary did not require PyTorch preflight checks")
                        if validation_payload.get("require_preflight_cuda") is not True:
                            failures.append("Validation summary did not require CUDA preflight checks")
                        if validation_payload.get("required_checkpoint_device") != "cuda":
                            failures.append("Validation summary did not require CUDA-trained checkpoints")
                    if args.require_preflight_eval:
                        if validation_payload.get("require_preflight") is not True:
                            failures.append("Validation summary did not require preflight checks")
                        if validation_payload.get("require_preflight_eval") is not True:
                            failures.append("Validation summary did not require eval preflight checks")
                        failures.extend(_check_preflight_eval_evidence(
                            args.artifacts_dir / "preflight_full_pipeline.json"
                        ))
                    if args.required_transformer_device:
                        actual_device = str(validation_payload.get("required_transformer_device", "") or "")
                        if actual_device != args.required_transformer_device:
                            failures.append(
                                f"Validation summary required_transformer_device is {actual_device!r}; "
                                f"expected {args.required_transformer_device!r}"
                            )
                        validation_completion_size = str(
                            validation_payload.get("required_completion_checkpoint_size", "") or ""
                        )
                        if required_completion_checkpoint_size:
                            if args.require_readiness and not validation_completion_size:
                                failures.append(
                                    "Validation summary did not record required_completion_checkpoint_size"
                                )
                            elif (
                                validation_completion_size
                                and validation_completion_size != required_completion_checkpoint_size
                            ):
                                failures.append(
                                    "Validation summary required_completion_checkpoint_size is "
                                    f"{validation_completion_size!r}; expected {required_completion_checkpoint_size!r}"
                                )
                    if args.require_selected_checkpoint and validation_payload.get("require_selected_checkpoint") is not True:
                        failures.append("Validation summary did not require selected checkpoint reranker")
                    min_generated = int(validation_payload.get("min_generated_per_family", 0))
                    if min_generated < args.required_min_generated_per_family:
                        failures.append(
                            f"Validation min_generated_per_family is {min_generated}; "
                            f"expected at least {args.required_min_generated_per_family}"
                        )
                    if args.required_max_generated_per_family:
                        max_generated = int(validation_payload.get("max_generated_per_family", 0))
                        if max_generated != args.required_max_generated_per_family:
                            failures.append(
                                f"Validation max_generated_per_family is {max_generated}; "
                                f"expected {args.required_max_generated_per_family}"
                            )
                    if expected_run_profile:
                        actual_validation_profile = str(validation_payload.get("run_profile", "") or "")
                        if actual_validation_profile != expected_run_profile:
                            failures.append(
                                "Validation run_profile is "
                                f"{actual_validation_profile!r}; expected {expected_run_profile!r}"
                            )
                    min_reranker_count = int(validation_payload.get("min_reranker_count", 0))
                    if min_reranker_count < args.required_min_reranker_count:
                        failures.append(
                            f"Validation min_reranker_count is {min_reranker_count}; "
                            f"expected at least {args.required_min_reranker_count}"
                        )
                    min_completion_compare_count = int(validation_payload.get("min_completion_compare_count", 0))
                    if min_completion_compare_count < args.required_min_completion_compare_count:
                        failures.append(
                            f"Validation min_completion_compare_count is {min_completion_compare_count}; "
                            f"expected at least {args.required_min_completion_compare_count}"
                        )
                    min_train_epochs = int(validation_payload.get("min_train_epochs", 0))
                    if min_train_epochs < args.required_min_train_epochs:
                        failures.append(
                            f"Validation min_train_epochs is {min_train_epochs}; "
                            f"expected at least {args.required_min_train_epochs}"
                        )
                    if args.required_batch_size:
                        validation_batch_size = int(validation_payload.get("required_batch_size", 0))
                        if validation_batch_size != args.required_batch_size:
                            failures.append(
                                f"Validation required_batch_size is {validation_batch_size}; "
                                f"expected {args.required_batch_size}"
                            )
            if args.require_readiness:
                failures.extend(_check_readiness_launch_command_script(args.artifacts_dir))
                if args.require_preflight_eval:
                    failures.extend(_check_eval_staging_evidence(
                        args.artifacts_dir,
                        readiness_payload,
                    ))
                failures.extend(_check_source_bundle_selftest_evidence(args.artifacts_dir))
                if readiness_payload.get("require_source_bundle") is True:
                    source_bundle = readiness_payload.get("source_bundle", {})
                    if not isinstance(source_bundle, dict):
                        failures.append("Leonardo readiness required source bundle but has no source_bundle object")
                    else:
                        bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
                        if not _valid_sha256(bundle_hash):
                            failures.append("Leonardo readiness source_bundle has no valid bundle_sha256")
                        if source_bundle.get("verified") is not True:
                            failures.append("Leonardo readiness source_bundle.verified is not true")
                    failures.extend(_source_bundle_current_file_failures(
                        readiness_payload,
                        [
                            *[f"scripts/{name}" for name in LEONARDO_SCRIPT_NAMES],
                            *_source_bundle_source_paths(readiness_payload),
                        ],
                        "Leonardo readiness",
                    ))
                checkpoint_audit = args.artifacts_dir / "checkpoint_audit.json"
                if checkpoint_audit.exists():
                    try:
                        checkpoint_payload = _read_json(checkpoint_audit)
                    except (OSError, json.JSONDecodeError) as exc:
                        failures.append(f"Checkpoint audit is not readable JSON: {checkpoint_audit} ({exc})")
                    else:
                        if checkpoint_payload.get("passed") is not True:
                            failures.append(f"Checkpoint audit did not pass: {checkpoint_audit}")
                        if expected_run_profile:
                            actual_checkpoint_profile = str(checkpoint_payload.get("run_profile", "") or "")
                            if actual_checkpoint_profile != expected_run_profile:
                                failures.append(
                                    "Checkpoint audit run_profile is "
                                    f"{actual_checkpoint_profile!r}; expected {expected_run_profile!r}"
                                )
                            actual_validation_profile = str(validation_payload.get("run_profile", "") or "")
                            if validation_payload and actual_checkpoint_profile != actual_validation_profile:
                                failures.append(
                                    "Checkpoint audit run_profile does not match validation run_profile: "
                                    f"checkpoint={actual_checkpoint_profile!r}, "
                                    f"validation={actual_validation_profile!r}"
                                )
                        if args.required_batch_size:
                            checkpoint_batch_size = int(checkpoint_payload.get("required_batch_size", 0))
                            if checkpoint_batch_size != args.required_batch_size:
                                failures.append(
                                    f"Checkpoint audit required_batch_size is {checkpoint_batch_size}; "
                                    f"expected {args.required_batch_size}"
                                )
                            if validation_payload:
                                validation_batch_size = int(validation_payload.get("required_batch_size", 0))
                                if checkpoint_batch_size != validation_batch_size:
                                    failures.append(
                                        "Checkpoint audit required_batch_size does not match validation "
                                        f"required_batch_size: checkpoint={checkpoint_batch_size}, "
                                        f"validation={validation_batch_size}"
                                    )
                        if readiness_payload.get("require_source_bundle") is True:
                            source_bundle = readiness_payload.get("source_bundle", {})
                            if not isinstance(source_bundle, dict):
                                source_bundle = {}
                            expected_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
                            checkpoint_bundle_hash = str(checkpoint_payload.get("source_bundle_sha256", "") or "")
                            validation_bundle_hash = str(validation_payload.get("source_bundle_sha256", "") or "")
                            if checkpoint_bundle_hash != expected_bundle_hash:
                                failures.append(
                                    "Checkpoint audit source_bundle_sha256 does not match readiness source bundle: "
                                    f"checkpoint={checkpoint_bundle_hash!r}, readiness={expected_bundle_hash!r}"
                                )
                            if validation_payload and validation_bundle_hash != expected_bundle_hash:
                                failures.append(
                                    "Validation source_bundle_sha256 does not match readiness source bundle: "
                                    f"validation={validation_bundle_hash!r}, readiness={expected_bundle_hash!r}"
                                )
            for size in args.required_checkpoint_sizes:
                train_summary = args.checkpoint_dir / size / "train_summary.json"
                train_log = args.checkpoint_dir / size / "train_log.json"
                model_path = args.checkpoint_dir / size / "model.pt"
                if not train_summary.exists():
                    failures.append(f"Missing checkpoint train summary for hash check: {train_summary}")
                    continue
                try:
                    train_payload = _read_json(train_summary)
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"Checkpoint train summary is not readable JSON: {train_summary} ({exc})")
                    continue
                failures.extend(_check_file_hash(
                    model_path,
                    train_payload.get("model_sha256"),
                    f"{size} checkpoint model against train_summary.json",
                ))
                failures.extend(_check_file_hash(
                    train_log,
                    train_payload.get("train_log_sha256"),
                    f"{size} checkpoint train_log against train_summary.json",
                ))
                if train_payload.get("model_size") != size:
                    failures.append(
                        f"{size} checkpoint train_summary model_size is {train_payload.get('model_size')!r}"
                    )
                expected_config = MODEL_CONFIGS.get(size)
                if expected_config is not None and train_payload.get("config") != expected_config:
                    failures.append(
                        f"{size} checkpoint train_summary config does not match expected {size} architecture"
                    )
                if args.required_batch_size:
                    actual_batch_size = int(train_payload.get("batch_size", 0))
                    if actual_batch_size != args.required_batch_size:
                        failures.append(
                            f"{size} checkpoint train_summary batch_size is {actual_batch_size}; "
                            f"expected {args.required_batch_size}"
                        )
                if readiness_payload.get("require_source_bundle") is True:
                    source_bundle = readiness_payload.get("source_bundle", {})
                    if not isinstance(source_bundle, dict):
                        source_bundle = {}
                    expected_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
                    actual_bundle_hash = str(train_payload.get("source_bundle_sha256", "") or "")
                    if actual_bundle_hash != expected_bundle_hash:
                        failures.append(
                            f"{size} checkpoint train_summary source_bundle_sha256 is "
                            f"{actual_bundle_hash!r}; expected readiness source bundle {expected_bundle_hash!r}"
                        )
                    if train_payload.get("source_bundle_required") is not True:
                        failures.append(f"{size} checkpoint train_summary did not require source-bundle proof")
                    if train_payload.get("source_bundle_verified") is not True:
                        failures.append(f"{size} checkpoint train_summary did not verify source-bundle proof")
            inference_summary = args.submission_dir / "inference_summary.json"
            reranker_metrics = args.artifacts_dir / "reranker_compare" / "metrics.json"
            completion_metrics = args.artifacts_dir / "completion_compare" / "metrics.json"
            if inference_summary.exists() and reranker_metrics.exists():
                try:
                    inference_payload = _read_json(inference_summary)
                    reranker_payload = _read_json(reranker_metrics)
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"Inference/reranker evidence is not readable JSON: {exc}")
                else:
                    best_checkpoint = _resolve_optional_path(reranker_payload.get("best_checkpoint"))
                    best_checkpoint_text_path = args.artifacts_dir / "reranker_compare" / "best_checkpoint.txt"
                    expected_best_checkpoint_text = str(reranker_payload.get("best_checkpoint", "") or "").strip()
                    try:
                        actual_best_checkpoint_text = best_checkpoint_text_path.read_text(
                            encoding="utf-8-sig",
                        ).strip()
                    except OSError as exc:
                        actual_best_checkpoint_text = ""
                        if args.require_selected_checkpoint:
                            failures.append(
                                f"Reranker best checkpoint pointer is not readable: {best_checkpoint_text_path} ({exc})"
                            )
                    else:
                        if args.require_selected_checkpoint and not actual_best_checkpoint_text:
                            failures.append(f"Reranker best checkpoint pointer is empty: {best_checkpoint_text_path}")
                        elif actual_best_checkpoint_text != expected_best_checkpoint_text:
                            failures.append(
                                "Reranker best checkpoint pointer does not match metrics.json best_checkpoint: "
                                f"{actual_best_checkpoint_text!r} != {expected_best_checkpoint_text!r}"
                            )
                    checkpoint_used = _resolve_optional_path(inference_payload.get("checkpoint_used"))
                    best_run = _find_reranker_run(reranker_payload)
                    if args.require_selected_checkpoint:
                        selection_scope = str(reranker_payload.get("selection_scope", "") or "")
                        if selection_scope != "checkpoints":
                            failures.append(
                                f"Reranker comparison selection_scope is {selection_scope!r}; expected 'checkpoints'"
                            )
                        failures.extend(_required_checkpoint_reranker_failures(
                            reranker_payload,
                            args.required_checkpoint_sizes,
                        ))
                    if best_run is None:
                        failures.append(
                            f"Best reranker is missing from reranker runs: {reranker_payload.get('best_reranker')}"
                        )
                    else:
                        failures.extend(_best_reranker_score_failures(reranker_payload, best_run))
                        if "selection_eligible" in best_run and not _truthy(best_run.get("selection_eligible", False)):
                            failures.append(
                                f"Selected reranker {reranker_payload.get('best_reranker')} is not selection eligible"
                            )
                    if args.require_selected_checkpoint and best_checkpoint is None:
                        failures.append("Reranker comparison did not select a checkpoint")
                    if args.require_selected_checkpoint and checkpoint_used is None:
                        failures.append("Inference did not use a selected checkpoint")
                    if args.require_selected_checkpoint:
                        if best_checkpoint is None and checkpoint_used is not None:
                            failures.append(
                                f"Reranker comparison selected baseline, but inference used checkpoint {checkpoint_used}"
                            )
                        if best_checkpoint is not None and checkpoint_used != best_checkpoint:
                            failures.append(
                                f"Inference checkpoint {checkpoint_used} does not match selected checkpoint {best_checkpoint}"
                            )
                        if best_checkpoint is not None:
                            selected_size = best_checkpoint.parent.name
                            if selected_size and selected_size not in set(args.required_checkpoint_sizes):
                                failures.append(
                                    f"Selected checkpoint size {selected_size} is not required by package evidence"
                                )
                        if best_checkpoint is not None and inference_payload.get("transformer_available") is not True:
                            failures.append("Inference used a selected checkpoint but transformer_available is not true")
                    expected_mode = str(reranker_payload.get("completion_mode", "") or "")
                    if expected_mode and str(inference_payload.get("completion_mode", "")) != expected_mode:
                        failures.append(
                            f"Inference completion mode {inference_payload.get('completion_mode')} "
                            f"does not match reranker comparison mode {expected_mode}"
                        )
                    selected_reranker_sha256 = ""
                    if args.require_selected_checkpoint and best_checkpoint is not None:
                        selected_run = _find_checkpoint_run(reranker_payload, best_checkpoint)
                        if selected_run is None:
                            failures.append(f"Selected checkpoint is missing from reranker runs: {best_checkpoint}")
                        else:
                            selected_reranker_sha256 = str(selected_run.get("checkpoint_sha256") or "").strip()
                            failures.extend(_check_file_hash(
                                best_checkpoint,
                                selected_reranker_sha256,
                                "selected reranker checkpoint against reranker metrics",
                            ))
                    if checkpoint_used is not None:
                        inference_sha256 = str(inference_payload.get("checkpoint_sha256") or "").strip()
                        failures.extend(_check_file_hash(
                            checkpoint_used,
                            inference_sha256,
                            "inference checkpoint against inference_summary.json",
                        ))
                        if selected_reranker_sha256 and inference_sha256 != selected_reranker_sha256:
                            failures.append(
                                "Inference checkpoint hash does not match selected reranker checkpoint hash: "
                                f"{inference_sha256} != {selected_reranker_sha256}"
                            )
            if completion_metrics.exists():
                try:
                    completion_payload = _read_json(completion_metrics)
                except (OSError, json.JSONDecodeError) as exc:
                    failures.append(f"Completion comparison metrics are not readable JSON: {completion_metrics} ({exc})")
                else:
                    completion_checkpoint = _resolve_optional_path(completion_payload.get("checkpoint_used"))
                    if completion_checkpoint is not None:
                        if (
                            required_completion_checkpoint_size
                            and completion_checkpoint.parent.name != required_completion_checkpoint_size
                        ):
                            failures.append(
                                "Completion comparison checkpoint size is "
                                f"{completion_checkpoint.parent.name!r}; "
                                f"expected {required_completion_checkpoint_size!r}"
                            )
                        failures.extend(_check_file_hash(
                            completion_checkpoint,
                            completion_payload.get("checkpoint_sha256"),
                            "completion comparison checkpoint against completion metrics",
                        ))
                    elif required_completion_checkpoint_size:
                        failures.append(
                            "Completion comparison did not use the required checkpoint size "
                            f"{required_completion_checkpoint_size!r}"
                        )

    if args.require_readiness and expected_completion_checkpoint:
        manifest_path = args.artifacts_dir / "run_manifest.json"
        if not manifest_path.exists():
            failures.append(f"Missing run manifest for completion checkpoint check: {manifest_path}")
        else:
            try:
                manifest_payload = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"Run manifest is not readable JSON: {manifest_path} ({exc})")
            else:
                failures.extend(_manifest_completion_checkpoint_failures(
                    manifest_payload,
                    expected_completion_checkpoint,
                    "Run manifest",
                ))

    required_stage = str(args.required_manifest_stage or "").strip()
    if required_stage:
        manifest_path = args.artifacts_dir / "run_manifest.json"
        events_path = args.artifacts_dir / "run_manifest_events.jsonl"
        expected_run_profile = _expected_run_profile(
            args.required_min_generated_per_family,
            args.required_max_generated_per_family,
        )
        expected_count_per_family = (
            args.required_max_generated_per_family
            or args.required_min_generated_per_family
        )
        if not manifest_path.exists():
            failures.append(f"Missing run manifest for required stage check: {manifest_path}")
        else:
            try:
                manifest_payload = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"Run manifest is not readable JSON: {manifest_path} ({exc})")
            else:
                actual_stage = str(manifest_payload.get("stage", "") or "")
                if actual_stage != required_stage:
                    failures.append(f"Run manifest stage is {actual_stage!r}; expected {required_stage!r}")
                if expected_run_profile:
                    actual_profile = str(manifest_payload.get("run_profile", "") or "")
                    if actual_profile != expected_run_profile:
                        failures.append(
                            f"Run manifest run_profile is {actual_profile!r}; expected {expected_run_profile!r}"
                        )
                if args.require_readiness:
                    failures.extend(_manifest_parameter_failures(
                        manifest_payload,
                        "Run manifest",
                        expected_run_profile,
                        expected_count_per_family,
                        expected_completion_checkpoint,
                        args.require_preflight_eval,
                        bool(readiness_payload and readiness_payload.get("require_source_bundle") is True),
                    ))
                if args.require_readiness and readiness_payload:
                    failures.extend(_manifest_source_bundle_failures(
                        manifest_payload,
                        readiness_payload,
                        "Run manifest",
                    ))
        if not _event_log_has_stage(events_path, required_stage):
            failures.append(f"Run manifest event log does not contain stage {required_stage!r}: {events_path}")
        elif expected_run_profile or (args.require_readiness and (expected_completion_checkpoint or readiness_payload)):
            terminal_event = _event_log_last_stage_payload(events_path, required_stage)
            actual_profile = str(terminal_event.get("run_profile", "") or "")
            if expected_run_profile and actual_profile != expected_run_profile:
                failures.append(
                    "Run manifest terminal event run_profile is "
                    f"{actual_profile!r}; expected {expected_run_profile!r}"
                )
            if args.require_readiness and expected_completion_checkpoint:
                failures.extend(_manifest_completion_checkpoint_failures(
                    terminal_event,
                    expected_completion_checkpoint,
                    "Run manifest terminal event",
                ))
            if args.require_readiness and readiness_payload:
                failures.extend(_manifest_source_bundle_failures(
                    terminal_event,
                    readiness_payload,
                    "Run manifest terminal event",
                ))
            if args.require_readiness:
                failures.extend(_manifest_parameter_failures(
                    terminal_event,
                    "Run manifest terminal event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    bool(readiness_payload and readiness_payload.get("require_source_bundle") is True),
                ))
        if args.require_readiness:
            stage_positions = _event_log_stage_positions(events_path)
            for stage in ("generation_prepared", "checkpoint_audited", "comparisons_complete"):
                if stage not in stage_positions:
                    failures.append(f"Run manifest event log does not contain stage {stage!r}: {events_path}")
            ordered_pairs = (
                ("generation_prepared", "checkpoint_audited"),
                ("checkpoint_audited", "comparisons_complete"),
                ("comparisons_complete", required_stage),
            )
            for earlier_stage, later_stage in ordered_pairs:
                if (
                    earlier_stage in stage_positions
                    and later_stage in stage_positions
                    and stage_positions[earlier_stage] >= stage_positions[later_stage]
                ):
                    failures.append(
                        "Run manifest event log does not record "
                        f"{earlier_stage!r} before {later_stage!r}: {events_path}"
                    )
            if "generation_prepared" in stage_positions and readiness_payload:
                generation_event = _event_log_last_stage_payload(events_path, "generation_prepared")
                failures.extend(_manifest_parameter_failures(
                    generation_event,
                    "Run manifest generation_prepared event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    bool(readiness_payload and readiness_payload.get("require_source_bundle") is True),
                ))
                failures.extend(_manifest_source_bundle_failures(
                    generation_event,
                    readiness_payload,
                    "Run manifest generation_prepared event",
                ))
            if not _event_log_has_stage(events_path, "checkpoint_audited"):
                failures.append(f"Run manifest event log does not contain stage 'checkpoint_audited': {events_path}")
            elif readiness_payload:
                checkpoint_event = _event_log_last_stage_payload(events_path, "checkpoint_audited")
                failures.extend(_manifest_parameter_failures(
                    checkpoint_event,
                    "Run manifest checkpoint_audited event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    bool(readiness_payload and readiness_payload.get("require_source_bundle") is True),
                ))
                failures.extend(_manifest_source_bundle_failures(
                    checkpoint_event,
                    readiness_payload,
                    "Run manifest checkpoint_audited event",
                ))
            if "comparisons_complete" in stage_positions and readiness_payload:
                comparisons_event = _event_log_last_stage_payload(events_path, "comparisons_complete")
                failures.extend(_manifest_parameter_failures(
                    comparisons_event,
                    "Run manifest comparisons_complete event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    bool(readiness_payload and readiness_payload.get("require_source_bundle") is True),
                ))
                failures.extend(_manifest_source_bundle_failures(
                    comparisons_event,
                    readiness_payload,
                    "Run manifest comparisons_complete event",
                ))

    if failures:
        _clear_previous_package(args.out_dir)
        print("Submission package failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_dir": str(args.submission_dir),
        "artifacts_dir": str(args.artifacts_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "generated_dir": str(args.generated_dir),
        "include_evidence": args.include_evidence,
        "require_evidence": args.require_evidence,
        "required_checkpoint_sizes": args.required_checkpoint_sizes,
        "required_completion_checkpoint_size": required_completion_checkpoint_size,
        "required_min_generated_per_family": args.required_min_generated_per_family,
        "required_max_generated_per_family": args.required_max_generated_per_family,
        "run_profile": _expected_run_profile(
            args.required_min_generated_per_family,
            args.required_max_generated_per_family,
        ),
        "required_min_reranker_count": args.required_min_reranker_count,
        "required_min_completion_compare_count": args.required_min_completion_compare_count,
        "required_min_train_epochs": args.required_min_train_epochs,
        "required_batch_size": args.required_batch_size,
        "required_manifest_stage": args.required_manifest_stage,
        "require_preflight_cuda": args.require_preflight_cuda,
        "require_preflight_eval": args.require_preflight_eval,
        "required_transformer_device": args.required_transformer_device,
        "require_selected_checkpoint": args.require_selected_checkpoint,
        "require_generated_metadata": args.require_generated_metadata,
        "require_readiness": args.require_readiness,
        "source_bundle": source_bundle_summary,
        "files": files,
        "evidence": evidence,
    }
    manifest_path = args.out_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    zip_path = args.out_dir / "track1_submission.zip"
    tmp_zip_path = args.out_dir / ".track1_submission.zip.tmp"
    with zipfile.ZipFile(tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in files:
            path = Path(str(item["path"]))
            zf.write(path, arcname=path.name)
        zf.write(manifest_path, arcname=manifest_path.name)
        for item in evidence:
            path = Path(str(item["path"]))
            zf.write(path, arcname=_zip_arcname(path, args.out_dir))
    zip_failures = _verify_zip(tmp_zip_path, args.out_dir, manifest_path, files, evidence)
    if zip_failures:
        _clear_previous_package(args.out_dir)
        print("Submission package failed:")
        for failure in zip_failures:
            print(f"- {failure}")
        raise SystemExit(2)
    tmp_zip_path.replace(zip_path)
    zip_sha256 = file_sha256(zip_path)
    checksum_path = args.out_dir / "track1_submission.zip.sha256"
    checksum_path.write_text(f"{zip_sha256}  {zip_path.name}\n", encoding="utf-8")

    print(f"Wrote {zip_path}")
    print(f"Wrote {checksum_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
