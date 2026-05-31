from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT
from .run_profiles import COUNT_PROFILES, profile_for_count
from .train import MODEL_CONFIGS
from .verify_package import verify_package


REQUIRED_SUBMISSIONS = ["nextstep.csv", "completion.csv", "anomaly.csv"]
LEONARDO_SCRIPT_PATHS = {
    "scripts/leonardo_common.sh",
    "scripts/leonardo_probe.sh",
    "scripts/leonardo_generate.sh",
    "scripts/leonardo_train.sh",
    "scripts/leonardo_train_scaling.sh",
    "scripts/leonardo_infer.sh",
    "scripts/leonardo_finalize.sh",
    "scripts/leonardo_full_pipeline.sh",
}
SOURCE_SNAPSHOT_PREFIX = Path("source_snapshot")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_root_entry_sha256(package_dir: Path, rel_path: Path) -> str:
    path = package_dir / rel_path
    if path.exists() and path.is_file():
        return _file_sha256(path)
    zip_path = package_dir / "track1_submission.zip"
    if not zip_path.exists():
        return ""
    member = str(rel_path).replace("\\", "/")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return hashlib.sha256(zf.read(member)).hexdigest()
    except (KeyError, OSError, zipfile.BadZipFile):
        return ""


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


def _event_log_text_has_stage(text: str, stage: str) -> bool:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("stage") == stage:
            return True
    return False


def _event_log_text_stage_positions(text: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, line in enumerate(text.splitlines()):
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
    return positions


def _event_log_text_last_stage_payload(text: str, stage: str) -> dict[str, object]:
    found: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("stage") == stage and isinstance(payload, dict):
            found = payload
    return found


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _expected_run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    if max_generated_per_family and max_generated_per_family != min_generated_per_family:
        return ""
    count = max_generated_per_family or min_generated_per_family
    if count <= 0:
        return ""
    return profile_for_count(count)


def _require_threshold(payload: dict[str, object], key: str, minimum: int) -> list[str]:
    actual = _as_int(payload.get(key, 0))
    if actual < minimum:
        return [f"{key} is {actual}; expected at least {minimum}"]
    return []


def _required_sizes_missing(payload: dict[str, object], key: str, required_sizes: list[str]) -> list[str]:
    present = {str(size) for size in payload.get(key, [])}
    missing = [size for size in required_sizes if size not in present]
    if missing:
        return [f"{key} is missing required sizes: {', '.join(missing)}"]
    return []


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


def _manifest_source_bundle_failures(
    payload: dict[str, object],
    readiness_payload: dict[str, object],
    label: str,
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    readiness_source_bundle = readiness_payload.get("source_bundle", {})
    if not isinstance(readiness_source_bundle, dict):
        return ["Packaged Leonardo readiness requires source bundle but has no source_bundle object"]
    expected_hash = str(readiness_source_bundle.get("bundle_sha256", "") or "")
    if not _valid_sha256(expected_hash):
        return ["Packaged Leonardo readiness source_bundle has no valid bundle_sha256"]

    manifest_source_bundle = payload.get("source_bundle", {})
    if not isinstance(manifest_source_bundle, dict) or not manifest_source_bundle:
        return [f"{label} did not record source_bundle evidence from Leonardo readiness"]
    failures: list[str] = []
    actual_hash = str(manifest_source_bundle.get("bundle_sha256", "") or "")
    if actual_hash != expected_hash:
        failures.append(
            f"{label} source_bundle.bundle_sha256 is {actual_hash!r}; "
            f"expected packaged readiness hash {expected_hash!r}"
        )
    if manifest_source_bundle.get("require_source_bundle") is not True:
        failures.append(f"{label} source_bundle did not record require_source_bundle=true")
    if manifest_source_bundle.get("verified") is not True:
        failures.append(f"{label} source_bundle did not record verified=true")
    if manifest_source_bundle.get("readiness_passed") is not True:
        failures.append(f"{label} source_bundle did not record readiness_passed=true")
    bundle_failures = manifest_source_bundle.get("failures", [])
    if isinstance(bundle_failures, list) and bundle_failures:
        failures.append(f"{label} source_bundle recorded failures: " + "; ".join(map(str, bundle_failures)))
    manifest_source = str(manifest_source_bundle.get("manifest_source", "") or "")
    if not manifest_source:
        failures.append(f"{label} source_bundle did not record manifest_source")
    return failures


def _package_source_bundle_failures(
    package_payload: dict[str, object],
    readiness_payload: dict[str, object],
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    readiness_source_bundle = readiness_payload.get("source_bundle", {})
    if not isinstance(readiness_source_bundle, dict):
        return ["Packaged Leonardo readiness requires source bundle but has no source_bundle object"]
    expected_hash = str(readiness_source_bundle.get("bundle_sha256", "") or "")
    if not _valid_sha256(expected_hash):
        return ["Packaged Leonardo readiness source_bundle has no valid bundle_sha256"]
    package_source_bundle = package_payload.get("source_bundle", {})
    if not isinstance(package_source_bundle, dict) or not package_source_bundle:
        return ["Package manifest did not record source_bundle summary"]
    failures: list[str] = []
    actual_hash = str(package_source_bundle.get("bundle_sha256", "") or "")
    if actual_hash != expected_hash:
        failures.append(
            f"Package manifest source_bundle.bundle_sha256 is {actual_hash!r}; "
            f"expected packaged readiness hash {expected_hash!r}"
        )
    if package_source_bundle.get("required") is not True:
        failures.append("Package manifest source_bundle did not record required=true")
    if package_source_bundle.get("verified") is not True:
        failures.append("Package manifest source_bundle did not record verified=true")
    if package_source_bundle.get("readiness_passed") is not True:
        failures.append("Package manifest source_bundle did not record readiness_passed=true")
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


def _packaged_scripts_source_bundle_failures(
    package_dir: Path,
    readiness_payload: dict[str, object],
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    rows = _source_bundle_manifest_rows(readiness_payload)
    if not rows:
        return ["Packaged Leonardo readiness source_bundle has no manifest_files list for script hash proof"]
    failures: list[str] = []
    for script in sorted(LEONARDO_SCRIPT_PATHS):
        row = rows.get(script)
        if not row:
            failures.append(f"Source bundle manifest missing packaged Leonardo script: {script}")
            continue
        expected_hash = str(row.get("sha256", "") or "")
        if not _valid_sha256(expected_hash):
            failures.append(f"Source bundle manifest hash is invalid for packaged Leonardo script: {script}")
            continue
        data, data_failures = _read_package_evidence_bytes(package_dir, Path(script))
        failures.extend(data_failures)
        if data_failures:
            continue
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            failures.append(
                f"Packaged Leonardo script hash for {script} does not match source bundle: "
                f"{actual_hash} != {expected_hash}"
            )
    return failures


def _packaged_source_snapshot_source_bundle_failures(
    package_dir: Path,
    readiness_payload: dict[str, object],
) -> list[str]:
    if readiness_payload.get("require_source_bundle") is not True:
        return []
    rows = _source_bundle_manifest_rows(readiness_payload)
    if not rows:
        return ["Packaged Leonardo readiness source_bundle has no manifest_files list for source hash proof"]
    source_paths = sorted(rel_path for rel_path in rows if rel_path.startswith("industrial_ai/"))
    if not source_paths:
        return ["Source bundle manifest has no industrial_ai source files for packaged source proof"]
    failures: list[str] = []
    for rel_path in source_paths:
        row = rows.get(rel_path, {})
        expected_hash = str(row.get("sha256", "") or "")
        if not _valid_sha256(expected_hash):
            failures.append(f"Source bundle manifest hash is invalid for packaged source file: {rel_path}")
            continue
        data, data_failures = _read_package_evidence_bytes(
            package_dir,
            SOURCE_SNAPSHOT_PREFIX / Path(rel_path),
        )
        failures.extend(data_failures)
        if data_failures:
            continue
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != expected_hash:
            failures.append(
                f"Packaged source snapshot hash for {rel_path} does not match source bundle: "
                f"{actual_hash} != {expected_hash}"
            )
    return failures


def _packaged_best_reranker_score_failures(
    reranker_payload: dict[str, object],
    best_row: dict[str, object] | None,
) -> list[str]:
    runs = reranker_payload.get("runs", [])
    if not isinstance(runs, list):
        return ["Packaged reranker metrics runs is not a list"]
    eligible_rows = [row for row in runs if isinstance(row, dict) and _truthy(row.get("selection_eligible", False))]
    if not eligible_rows:
        return ["Packaged reranker metrics has no selection-eligible runs"]
    if best_row is None:
        return []
    best_score = _as_float(best_row.get("selection_score", 0.0))
    max_score = max(_as_float(row.get("selection_score", 0.0)) for row in eligible_rows)
    if best_score + 1e-12 < max_score:
        return [
            "Packaged selected reranker is not the highest-scoring eligible run: "
            f"{best_score:.6g} < {max_score:.6g}"
        ]
    return []


def _packaged_required_checkpoint_reranker_failures(
    reranker_payload: dict[str, object],
    required_sizes: list[str],
) -> list[str]:
    runs_by_label = {
        str(row.get("reranker", "") or ""): row
        for row in reranker_payload.get("runs", [])
        if isinstance(row, dict)
    }
    failures: list[str] = []
    for size in required_sizes:
        row = runs_by_label.get(size)
        if row is None:
            failures.append(f"Packaged reranker metrics missing required checkpoint run: {size}")
            continue
        checkpoint = _resolve_optional_path(row.get("checkpoint"))
        if checkpoint is None:
            failures.append(f"Packaged reranker required run {size} has no checkpoint")
        elif checkpoint.parent.name != size:
            failures.append(
                f"Packaged reranker required run {size} uses checkpoint size {checkpoint.parent.name!r}"
            )
        if not _truthy(row.get("available", False)):
            failures.append(f"Packaged reranker required run {size} was not available")
        if not _truthy(row.get("selection_eligible", False)):
            failures.append(f"Packaged reranker required run {size} was not selection eligible")
        if not str(row.get("checkpoint_sha256", "") or "").strip():
            failures.append(f"Packaged reranker required run {size} has no checkpoint_sha256")
    return failures


def _evidence_rel_path(path_text: str) -> Path:
    parts = [part for part in path_text.replace("\\", "/").split("/") if part]
    if "evidence" in parts:
        evidence_index = parts.index("evidence")
        return Path(*parts[evidence_index + 1:])
    return Path(parts[-1]) if parts else Path("")


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
    for script in LEONARDO_SCRIPT_PATHS:
        required.add(Path(script))
    return required


def _check_required_evidence_entries(
    package_payload: dict[str, object],
    checkpoint_sizes: list[str],
    require_generated_metadata: bool,
    require_readiness: bool,
    require_eval_staging: bool,
) -> list[str]:
    present: set[Path] = set()
    for item in package_payload.get("evidence", []):
        if not isinstance(item, dict):
            continue
        present.add(_evidence_rel_path(str(item.get("path", "") or "")))
    failures: list[str] = []
    for rel_path in sorted(_required_evidence_paths(
        checkpoint_sizes,
        require_generated_metadata,
        require_readiness,
        require_eval_staging,
    )):
        if rel_path not in present:
            failures.append(f"Package manifest missing required evidence entry: {rel_path}")
    return failures


def _check_required_submission_entries(package_payload: dict[str, object]) -> list[str]:
    rows_by_name: dict[str, int] = {}
    for item in package_payload.get("files", []):
        if not isinstance(item, dict):
            continue
        name = Path(str(item.get("path", ""))).name
        rows_by_name[name] = _as_int(item.get("rows", 0))
    failures: list[str] = []
    for filename in REQUIRED_SUBMISSIONS:
        if filename not in rows_by_name:
            failures.append(f"Package manifest missing required submission entry: {filename}")
        elif rows_by_name[filename] <= 0:
            failures.append(f"Package manifest entry {filename} has no rows")
    return failures


def _check_manifest_artifact_paths(artifacts: object, require_readiness: bool) -> list[str]:
    if not isinstance(artifacts, dict):
        return ["Run manifest has no artifacts object"]
    failures: list[str] = []
    expected_names = {
        "leonardo_shell_audit": "leonardo_shell_audit.json",
        "submission_package": "track1_submission.zip",
        "submission_package_sha256": "track1_submission.zip.sha256",
        "package_manifest": "package_manifest.json",
    }
    if require_readiness:
        expected_names["leonardo_readiness"] = "leonardo_readiness.json"
        expected_names["leonardo_launch_commands"] = "leonardo_launch_commands.sh"
        expected_names["checkpoint_audit"] = "checkpoint_audit.json"
    for key, expected_name in expected_names.items():
        value = str(artifacts.get(key, "") or "").strip()
        if not value:
            failures.append(f"Run manifest does not record {key}")
            continue
        actual_name = Path(value).name
        if actual_name != expected_name:
            failures.append(f"Run manifest {key} points to {actual_name!r}; expected {expected_name!r}")
    return failures


def _resolve_optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _read_package_evidence_json(package_dir: Path, rel_path: Path) -> tuple[dict[str, object], list[str]]:
    payload, failures = _read_package_evidence_text(package_dir, rel_path)
    if failures:
        return {}, failures
    try:
        return json.loads(payload), []
    except json.JSONDecodeError as exc:
        arcname = str((Path("evidence") / rel_path)).replace("\\", "/")
        return {}, [f"Package evidence JSON is not readable: {arcname} ({exc})"]


def _read_package_root_json(package_dir: Path, rel_path: Path) -> tuple[dict[str, object], list[str]]:
    path = package_dir / rel_path
    if path.exists():
        try:
            return _read_json(path), []
        except (OSError, json.JSONDecodeError) as exc:
            return {}, [f"Package file is not readable JSON: {path} ({exc})"]
    zip_path = package_dir / "track1_submission.zip"
    arcname = str(rel_path).replace("\\", "/")
    if not zip_path.exists():
        return {}, [f"Missing package file: {path}; package ZIP also missing: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                raw = zf.read(arcname)
            except KeyError:
                return {}, [f"Missing package file: {path} and ZIP entry {arcname}"]
    except zipfile.BadZipFile as exc:
        return {}, [f"Package ZIP is not readable while checking {arcname}: {zip_path} ({exc})"]
    try:
        return json.loads(raw.decode("utf-8-sig")), []
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, [f"Package ZIP file is not readable JSON: {arcname} ({exc})"]


def _read_package_evidence_text(package_dir: Path, rel_path: Path) -> tuple[str, list[str]]:
    path = package_dir / "evidence" / rel_path
    if path.exists():
        try:
            return path.read_text(encoding="utf-8-sig"), []
        except OSError as exc:
            return "", [f"Package evidence file is not readable: {path} ({exc})"]

    zip_path = package_dir / "track1_submission.zip"
    arcname = str((Path("evidence") / rel_path)).replace("\\", "/")
    if not zip_path.exists():
        return "", [f"Missing package evidence file: {path}; package ZIP also missing: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                raw = zf.read(arcname)
            except KeyError:
                return "", [f"Missing package evidence file: {path} and ZIP entry {arcname}"]
    except zipfile.BadZipFile as exc:
        return "", [f"Package ZIP is not readable while checking evidence {arcname}: {zip_path} ({exc})"]
    try:
        return raw.decode("utf-8-sig"), []
    except UnicodeDecodeError as exc:
        return "", [f"Package ZIP evidence file is not UTF-8 readable: {arcname} ({exc})"]


def _read_package_evidence_bytes(package_dir: Path, rel_path: Path) -> tuple[bytes, list[str]]:
    path = package_dir / "evidence" / rel_path
    if path.exists():
        try:
            return path.read_bytes(), []
        except OSError as exc:
            return b"", [f"Package evidence file is not readable: {path} ({exc})"]

    zip_path = package_dir / "track1_submission.zip"
    arcname = str((Path("evidence") / rel_path)).replace("\\", "/")
    if not zip_path.exists():
        return b"", [f"Missing package evidence file: {path}; package ZIP also missing: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            try:
                return zf.read(arcname), []
            except KeyError:
                return b"", [f"Missing package evidence file: {path} and ZIP entry {arcname}"]
    except zipfile.BadZipFile as exc:
        return b"", [f"Package ZIP is not readable while checking evidence {arcname}: {zip_path} ({exc})"]


def _read_artifact_or_package_json(
    artifact_path: Path,
    package_dir: Path,
    rel_evidence_path: Path,
    label: str,
) -> tuple[dict[str, object], list[str]]:
    if artifact_path.exists():
        try:
            artifact_payload = _read_json(artifact_path)
        except (OSError, json.JSONDecodeError) as exc:
            return {}, [f"{label} is not readable JSON: {artifact_path} ({exc})"]
        package_payload, package_failures = _read_package_evidence_json(package_dir, rel_evidence_path)
        if not package_failures and artifact_payload != package_payload:
            return artifact_payload, [
                f"{label} differs between local artifact {artifact_path} and package evidence {rel_evidence_path}"
            ]
        return artifact_payload, []
    payload, failures = _read_package_evidence_json(package_dir, rel_evidence_path)
    if failures:
        return {}, [f"Missing {label}: {artifact_path}; {failure}" for failure in failures]
    return payload, []


def _read_event_log_text_with_package_fallback(
    artifact_path: Path,
    package_dir: Path,
) -> tuple[str, list[str]]:
    package_text = ""
    package_failures: list[str] = []
    if artifact_path.exists():
        try:
            artifact_text = artifact_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            return "", [f"Run manifest event log is not readable: {artifact_path} ({exc})"]
        package_text, package_failures = _read_package_evidence_text(
            package_dir,
            Path("run_manifest_events.jsonl"),
        )
        if not package_failures and artifact_text != package_text:
            return artifact_text, [
                "Run manifest event log differs between local artifact "
                f"{artifact_path} and package evidence run_manifest_events.jsonl"
            ]
        return artifact_text, []
    package_text, package_failures = _read_package_evidence_text(package_dir, Path("run_manifest_events.jsonl"))
    if package_failures:
        return "", [f"Missing run manifest event log: {artifact_path}; {failure}" for failure in package_failures]
    return package_text, []


def _event_log_has_stage_with_package_fallback(
    artifact_path: Path,
    package_dir: Path,
    required_stage: str,
) -> tuple[bool, list[str]]:
    text, failures = _read_event_log_text_with_package_fallback(artifact_path, package_dir)
    if failures:
        return False, failures
    return _event_log_text_has_stage(text, required_stage), []


def _event_log_stage_order_with_package_fallback(
    artifact_path: Path,
    package_dir: Path,
    earlier_stage: str,
    later_stage: str,
) -> tuple[bool, list[str]]:
    text, failures = _read_event_log_text_with_package_fallback(artifact_path, package_dir)
    if failures:
        return False, failures
    positions = _event_log_text_stage_positions(text)
    if earlier_stage not in positions or later_stage not in positions:
        return False, []
    return positions[earlier_stage] < positions[later_stage], []


def _event_log_stage_payload_with_package_fallback(
    artifact_path: Path,
    package_dir: Path,
    stage: str,
) -> tuple[dict[str, object], list[str]]:
    text, failures = _read_event_log_text_with_package_fallback(artifact_path, package_dir)
    if failures:
        return {}, failures
    return _event_log_text_last_stage_payload(text, stage), []


def _check_packaged_execution_evidence(
    package_dir: Path,
    required_transformer_device: str,
    require_selected_checkpoint: bool,
    required_checkpoint_sizes: list[str],
    required_completion_checkpoint_size: str,
    required_batch_size: int,
    expected_source_bundle_sha256: str,
) -> list[str]:
    failures: list[str] = []
    reranker_payload, reranker_failures = _read_package_evidence_json(
        package_dir,
        Path("reranker_compare") / "metrics.json",
    )
    completion_payload, completion_failures = _read_package_evidence_json(
        package_dir,
        Path("completion_compare") / "metrics.json",
    )
    inference_payload, inference_failures = _read_package_evidence_json(
        package_dir,
        Path("inference_summary.json"),
    )
    failures.extend(reranker_failures)
    failures.extend(completion_failures)
    failures.extend(inference_failures)
    if failures:
        return failures

    if required_transformer_device:
        for label, payload in (
            ("Reranker comparison", reranker_payload),
            ("Completion comparison", completion_payload),
            ("Inference", inference_payload),
        ):
            actual_device = str(payload.get("transformer_device", "") or "")
            if actual_device != required_transformer_device:
                failures.append(
                    f"Packaged {label} transformer_device is {actual_device!r}; "
                    f"expected {required_transformer_device!r}"
                )

    def check_train_summary_hash(label: str, checkpoint: Path | None, checkpoint_sha256: str) -> None:
        if checkpoint is None:
            return
        size = checkpoint.parent.name
        if not size:
            failures.append(f"Packaged {label} checkpoint path has no model-size parent: {checkpoint}")
            return
        if not checkpoint_sha256:
            failures.append(f"Packaged {label} has no checkpoint_sha256")
            return
        summary_payload, summary_failures = _read_package_evidence_json(
            package_dir,
            Path("checkpoints") / size / "train_summary.json",
        )
        failures.extend(summary_failures)
        if summary_failures:
            return
        log_bytes, log_failures = _read_package_evidence_bytes(
            package_dir,
            Path("checkpoints") / size / "train_log.json",
        )
        failures.extend(log_failures)
        summary_size = str(summary_payload.get("model_size", "") or "")
        if summary_size and summary_size != size:
            failures.append(
                f"Packaged {label} checkpoint size {size!r} does not match train_summary model_size {summary_size!r}"
            )
        expected_config = MODEL_CONFIGS.get(size)
        if expected_config is not None and summary_payload.get("config") != expected_config:
            failures.append(
                f"Packaged {label} {size} train_summary config does not match expected {size} architecture"
            )
        if required_batch_size:
            actual_batch_size = _as_int(summary_payload.get("batch_size", 0))
            if actual_batch_size != required_batch_size:
                failures.append(
                    f"Packaged {label} {size} train_summary batch_size is {actual_batch_size}; "
                    f"expected {required_batch_size}"
                )
        if expected_source_bundle_sha256:
            actual_source_bundle_sha = str(summary_payload.get("source_bundle_sha256", "") or "")
            if actual_source_bundle_sha != expected_source_bundle_sha256:
                failures.append(
                    f"Packaged {label} {size} train_summary source_bundle_sha256 is "
                    f"{actual_source_bundle_sha!r}; expected {expected_source_bundle_sha256!r}"
                )
            if summary_payload.get("source_bundle_required") is not True:
                failures.append(f"Packaged {label} {size} train_summary did not require source-bundle proof")
            if summary_payload.get("source_bundle_verified") is not True:
                failures.append(f"Packaged {label} {size} train_summary did not verify source-bundle proof")
        summary_sha256 = str(summary_payload.get("model_sha256", "") or "")
        if not summary_sha256:
            failures.append(f"Packaged {label} train_summary has no model_sha256 for {size}")
        elif checkpoint_sha256 != summary_sha256:
            failures.append(
                f"Packaged {label} checkpoint_sha256 does not match {size} train_summary model_sha256: "
                f"{checkpoint_sha256} != {summary_sha256}"
            )
        log_sha256 = str(summary_payload.get("train_log_sha256", "") or "")
        if not log_sha256:
            failures.append(f"Packaged {label} train_summary has no train_log_sha256 for {size}")
        elif not log_failures:
            actual_log_sha256 = hashlib.sha256(log_bytes).hexdigest()
            if log_sha256 != actual_log_sha256:
                failures.append(
                    f"Packaged {label} train_log_sha256 does not match {size} train_log.json: "
                    f"{log_sha256} != {actual_log_sha256}"
                )

    completion_checkpoint = _resolve_optional_path(completion_payload.get("checkpoint_used"))
    completion_checkpoint_sha = str(completion_payload.get("checkpoint_sha256", "") or "")
    if required_completion_checkpoint_size:
        if completion_checkpoint is None:
            failures.append(
                "Packaged completion comparison did not use the required checkpoint size "
                f"{required_completion_checkpoint_size!r}"
            )
        elif completion_checkpoint.parent.name != required_completion_checkpoint_size:
            failures.append(
                "Packaged completion comparison checkpoint size is "
                f"{completion_checkpoint.parent.name!r}; expected {required_completion_checkpoint_size!r}"
            )
    check_train_summary_hash("completion comparison", completion_checkpoint, completion_checkpoint_sha)

    if require_selected_checkpoint:
        selection_scope = str(reranker_payload.get("selection_scope", "") or "")
        if selection_scope != "checkpoints":
            failures.append(f"Packaged reranker selection_scope is {selection_scope!r}; expected 'checkpoints'")
        failures.extend(_packaged_required_checkpoint_reranker_failures(
            reranker_payload,
            required_checkpoint_sizes,
        ))
        runs_by_label = {
            str(row.get("reranker", "") or ""): row
            for row in reranker_payload.get("runs", [])
            if isinstance(row, dict)
        }
        for size in required_checkpoint_sizes:
            row = runs_by_label.get(size)
            if row is None:
                continue
            check_train_summary_hash(
                f"required reranker {size}",
                _resolve_optional_path(row.get("checkpoint")),
                str(row.get("checkpoint_sha256", "") or ""),
            )
        best_reranker = str(reranker_payload.get("best_reranker", "") or "")
        best_checkpoint_text, best_checkpoint_text_failures = _read_package_evidence_text(
            package_dir,
            Path("reranker_compare") / "best_checkpoint.txt",
        )
        failures.extend(best_checkpoint_text_failures)
        expected_best_checkpoint_text = str(reranker_payload.get("best_checkpoint", "") or "").strip()
        actual_best_checkpoint_text = best_checkpoint_text.strip()
        best_checkpoint = _resolve_optional_path(reranker_payload.get("best_checkpoint"))
        if not best_reranker or best_reranker == "baseline":
            failures.append(f"Packaged reranker selected {best_reranker!r}; expected a checkpoint reranker")
        if best_checkpoint is None:
            failures.append("Packaged reranker did not select a checkpoint")
        if not actual_best_checkpoint_text:
            failures.append("Packaged reranker best_checkpoint.txt is empty")
        elif actual_best_checkpoint_text != expected_best_checkpoint_text:
            failures.append(
                "Packaged reranker best_checkpoint.txt does not match metrics.json best_checkpoint: "
                f"{actual_best_checkpoint_text!r} != {expected_best_checkpoint_text!r}"
            )
        best_row = next(
            (
                row for row in reranker_payload.get("runs", [])
                if isinstance(row, dict) and str(row.get("reranker", "") or "") == best_reranker
            ),
            {},
        )
        if not best_row:
            failures.append(f"Packaged reranker best row is missing: {best_reranker}")
        else:
            failures.extend(_packaged_best_reranker_score_failures(reranker_payload, best_row))
            if not _truthy(best_row.get("selection_eligible", False)):
                failures.append(f"Packaged reranker best row is not selection eligible: {best_reranker}")
            if not _truthy(best_row.get("available", False)):
                failures.append(f"Packaged reranker best row was not available: {best_reranker}")
            row_checkpoint = _resolve_optional_path(best_row.get("checkpoint"))
            if best_checkpoint is not None and row_checkpoint != best_checkpoint:
                failures.append(
                    f"Packaged reranker best row checkpoint {row_checkpoint} "
                    f"does not match best_checkpoint {best_checkpoint}"
                )
            check_train_summary_hash(
                "selected reranker",
                row_checkpoint,
                str(best_row.get("checkpoint_sha256", "") or ""),
            )
        inference_checkpoint = _resolve_optional_path(inference_payload.get("checkpoint_used"))
        inference_checkpoint_sha = str(inference_payload.get("checkpoint_sha256", "") or "")
        if best_checkpoint is not None and inference_checkpoint != best_checkpoint:
            failures.append(
                f"Packaged inference checkpoint {inference_checkpoint} "
                f"does not match selected checkpoint {best_checkpoint}"
            )
        check_train_summary_hash("inference", inference_checkpoint, inference_checkpoint_sha)
        if inference_payload.get("transformer_available") is not True:
            failures.append("Packaged inference did not report transformer_available=true")
    return failures


def _check_packaged_preflight_evidence(package_dir: Path, require_cuda: bool, require_eval: bool) -> list[str]:
    payload, failures = _read_package_evidence_json(package_dir, Path("preflight_full_pipeline.json"))
    if failures:
        return failures
    preflight_failures = payload.get("failures", [])
    if isinstance(preflight_failures, list) and preflight_failures:
        failures.append("Packaged preflight recorded failures: " + "; ".join(map(str, preflight_failures)))
    if payload.get("official_generator_loads") is not True:
        failures.append("Packaged preflight did not confirm official generator loading")
    preflight_requires_eval = payload.get("require_eval") is True
    if require_eval and not preflight_requires_eval:
        failures.append("Packaged preflight did not require eval inputs")
    eval_inputs = payload.get("eval_inputs", [])
    if (preflight_requires_eval or require_eval) and isinstance(eval_inputs, list):
        if require_eval and not eval_inputs:
            failures.append("Packaged preflight did not record eval input checks")
        for row in eval_inputs:
            if not isinstance(row, dict):
                failures.append(f"Packaged preflight eval input row is not an object: {row!r}")
                continue
            if row.get("exists") is not True:
                failures.append(f"Packaged preflight missing eval input: {row.get('path')}")
            elif _as_int(row.get("bytes", 0)) <= 0:
                failures.append(f"Packaged preflight saw empty eval input: {row.get('path')}")
            elif row.get("missing_columns"):
                failures.append(
                    "Packaged preflight saw eval input with missing columns: "
                    f"{row.get('path')} ({', '.join(map(str, row.get('missing_columns', [])))})"
                )
            elif _as_int(row.get("rows", 0)) <= 0:
                failures.append(f"Packaged preflight saw eval input with no rows: {row.get('path')}")
    torch_info = payload.get("torch", {})
    if not isinstance(torch_info, dict):
        return failures + ["Packaged preflight has no torch object"]
    if torch_info.get("available") is not True:
        failures.append("Packaged preflight did not confirm PyTorch availability")
    if require_cuda:
        if torch_info.get("cuda_required") is not True:
            failures.append("Packaged preflight did not require CUDA")
        if torch_info.get("cuda_available") is not True:
            failures.append("Packaged preflight did not confirm CUDA availability")
        if _as_int(torch_info.get("cuda_device_count", 0)) <= 0:
            failures.append("Packaged preflight did not report a CUDA device")
        devices = torch_info.get("devices", [])
        if not isinstance(devices, list) or not devices:
            failures.append("Packaged preflight did not report CUDA device details")
        else:
            for row in devices:
                if not isinstance(row, dict):
                    failures.append(f"Packaged preflight CUDA device row is not an object: {row!r}")
                    continue
                if not str(row.get("name", "") or ""):
                    failures.append("Packaged preflight CUDA device row has no name")
                if _as_int(row.get("total_memory_bytes", 0)) <= 0:
                    failures.append("Packaged preflight CUDA device row has no total_memory_bytes")
        if not str(torch_info.get("cuda_version", "") or ""):
            failures.append("Packaged preflight has no torch CUDA runtime version")
    return failures


def _rows_by_label(payload: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("label", "") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("label", "") or "")
    }


def _check_packaged_eval_staging_evidence(
    package_dir: Path,
    readiness_payload: dict[str, object],
) -> list[str]:
    staging_payload, failures = _read_package_evidence_json(
        package_dir,
        Path("eval_staging_manifest.json"),
    )
    if failures:
        return failures
    preflight_payload, preflight_failures = _read_package_evidence_json(
        package_dir,
        Path("preflight_full_pipeline.json"),
    )
    failures.extend(preflight_failures)
    if staging_payload.get("passed") is not True:
        failures.append("Packaged eval staging manifest did not pass")
    staging_failures = staging_payload.get("failures", [])
    if isinstance(staging_failures, list) and staging_failures:
        failures.append("Packaged eval staging manifest recorded failures: " + "; ".join(map(str, staging_failures)))
    destinations = _rows_by_label(staging_payload, "destinations")
    readiness_rows = _rows_by_label(readiness_payload, "eval_inputs")
    preflight_rows = _rows_by_label(preflight_payload, "eval_inputs") if preflight_payload else {}
    for label in ("valid", "anomaly"):
        row = destinations.get(label)
        if row is None:
            failures.append(f"Packaged eval staging manifest missing {label} destination row")
            continue
        staged_hash = str(row.get("sha256", "") or "")
        if not _valid_sha256(staged_hash):
            failures.append(f"Packaged eval staging manifest has invalid {label} SHA-256")
        if row.get("exists") is not True:
            failures.append(f"Packaged eval staging manifest says {label} destination is missing")
        if _as_int(row.get("rows", 0)) <= 0:
            failures.append(f"Packaged eval staging manifest says {label} destination has no rows")
        if row.get("missing_columns"):
            failures.append(
                f"Packaged eval staging manifest says {label} destination is missing columns: "
                + ", ".join(map(str, row.get("missing_columns", [])))
            )
        readiness_row = readiness_rows.get(label)
        if readiness_row is None:
            failures.append(f"Packaged Leonardo readiness missing {label} eval input row")
        else:
            readiness_hash = str(readiness_row.get("sha256", "") or "")
            if readiness_hash != staged_hash:
                failures.append(
                    f"Packaged Leonardo readiness {label} eval SHA-256 does not match staging manifest: "
                    f"{readiness_hash or 'missing'} != {staged_hash or 'missing'}"
                )
        preflight_row = preflight_rows.get(label)
        if preflight_row is None:
            failures.append(f"Packaged preflight missing {label} eval input row")
        else:
            preflight_hash = str(preflight_row.get("sha256", "") or "")
            if preflight_hash != staged_hash:
                failures.append(
                    f"Packaged preflight {label} eval SHA-256 does not match staging manifest: "
                    f"{preflight_hash or 'missing'} != {staged_hash or 'missing'}"
                )
    return failures


def _metadata_has_successful_origin(
    metadata: object,
    family: str,
    min_count: int,
    max_count: int,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    status = str(metadata.get("status", "") or "")
    if status == "generated_chunked":
        pass
    elif status == "reused_existing_output":
        if not _metadata_has_successful_origin(
            metadata.get("existing_metadata"),
            family,
            min_count,
            max_count,
        ):
            return False
    else:
        return False
    if str(metadata.get("family", "") or "") != family:
        return False
    actual_count = _as_int(metadata.get("actual_count", 0))
    if actual_count < min_count:
        return False
    return actual_count == max_count if max_count else True


def _generated_file_hashes_by_family(payload: dict[str, object]) -> tuple[dict[str, str], list[str]]:
    hashes: dict[str, str] = {}
    failures: list[str] = []
    files = payload.get("files", [])
    if not isinstance(files, list):
        return hashes, ["Packaged corpus audit files is not a list"]
    for row in files:
        if not isinstance(row, dict) or str(row.get("source", "") or "") != "generated":
            continue
        family = str(row.get("family", "") or "")
        file_hash = str(row.get("file_sha256", "") or "")
        metadata_hash = str(row.get("metadata_output_sha256", "") or "")
        path = str(row.get("path", "") or "")
        if not family:
            failures.append(f"Packaged corpus audit generated file row has no family: {path or row!r}")
            continue
        if len(file_hash) != 64:
            failures.append(f"Packaged corpus audit generated file has invalid file_sha256 for {path or family}")
        if metadata_hash != file_hash:
            failures.append(
                f"Packaged corpus audit generated metadata hash does not match file hash for {path or family}: "
                f"{metadata_hash or 'missing'} != {file_hash or 'missing'}"
            )
        hashes[family] = file_hash
    return hashes, failures


def _check_packaged_corpus_audit_evidence(
    package_dir: Path,
    min_generated_per_family: int,
    max_generated_per_family: int,
    require_generated_metadata: bool,
) -> list[str]:
    payload, failures = _read_package_evidence_json(package_dir, Path("corpus_audit") / "summary.json")
    if failures:
        return failures
    for failure in payload.get("failures", []):
        failures.append(f"Packaged corpus audit failure: {failure}")
    audit_min = _as_int(payload.get("min_generated_per_family", 0))
    if audit_min < min_generated_per_family:
        failures.append(
            f"Packaged corpus audit min_generated_per_family is {audit_min}; "
            f"expected at least {min_generated_per_family}"
        )
    audit_max = _as_int(payload.get("max_generated_per_family", 0))
    if max_generated_per_family and audit_max != max_generated_per_family:
        failures.append(
            f"Packaged corpus audit max_generated_per_family is {audit_max}; "
            f"expected {max_generated_per_family}"
        )

    family_rows = payload.get("families", [])
    if not isinstance(family_rows, list) or not family_rows:
        failures.append("Packaged corpus audit has no family rows")
        family_rows = []
    expected_families = {"IC", "IGBT", "MOSFET"}
    seen_families: set[str] = set()
    for row in family_rows:
        if not isinstance(row, dict):
            failures.append(f"Packaged corpus audit family row is not an object: {row!r}")
            continue
        family = str(row.get("family", "") or "")
        seen_families.add(family)
        generated = _as_int(row.get("generated_sequences", 0))
        if generated < min_generated_per_family:
            failures.append(
                f"Packaged corpus audit {family} generated_sequences is {generated}; "
                f"expected at least {min_generated_per_family}"
            )
        if max_generated_per_family and generated > max_generated_per_family:
            failures.append(
                f"Packaged corpus audit {family} generated_sequences is {generated}; "
                f"expected at most {max_generated_per_family}"
            )
        if row.get("meets_generated_minimum") is not True:
            failures.append(f"Packaged corpus audit {family} does not meet generated minimum")
        if max_generated_per_family and row.get("meets_generated_maximum") is not True:
            failures.append(f"Packaged corpus audit {family} does not meet generated maximum")
    missing_families = sorted(expected_families - seen_families)
    if missing_families:
        failures.append("Packaged corpus audit missing families: " + ", ".join(missing_families))

    generated_hashes, hash_failures = _generated_file_hashes_by_family(payload)
    failures.extend(hash_failures)
    if require_generated_metadata and min_generated_per_family > 0:
        for family in sorted(expected_families):
            metadata_payload, metadata_failures = _read_package_evidence_json(
                package_dir,
                Path("generated_metadata") / f"{family}_extra.csv.metadata.json",
            )
            failures.extend(metadata_failures)
            if metadata_failures:
                continue
            if not _metadata_has_successful_origin(
                metadata_payload,
                family,
                min_generated_per_family,
                max_generated_per_family,
            ):
                expected = (
                    f"exact {max_generated_per_family}"
                    if max_generated_per_family
                    else f"at least {min_generated_per_family}"
                )
                failures.append(
                    f"Packaged generated metadata for {family} does not prove a successful "
                    f"{expected}-sequence generation"
                )
            audit_hash = generated_hashes.get(family, "")
            metadata_hash = str(metadata_payload.get("output_sha256", "") or "")
            if len(metadata_hash) != 64:
                failures.append(f"Packaged generated metadata for {family} has no output_sha256")
            elif audit_hash and metadata_hash != audit_hash:
                failures.append(
                    f"Packaged generated metadata for {family} output_sha256 does not match corpus audit: "
                    f"{metadata_hash} != {audit_hash}"
                )
    return failures


def _check_packaged_shell_audit_evidence(package_dir: Path) -> list[str]:
    payload, failures = _read_package_evidence_json(package_dir, Path("leonardo_shell_audit.json"))
    if failures:
        return failures
    if payload.get("passed") is not True:
        failures.append("Packaged Leonardo shell audit did not pass")
    audit_failures = payload.get("failures", [])
    if isinstance(audit_failures, list) and audit_failures:
        failures.append("Packaged Leonardo shell audit recorded failures: " + "; ".join(map(str, audit_failures)))
    scripts = payload.get("scripts", [])
    if not isinstance(scripts, list) or not scripts:
        failures.append("Packaged Leonardo shell audit has no scripts list")
    else:
        expected_scripts = LEONARDO_SCRIPT_PATHS
        present = {str(script).replace("\\", "/") for script in scripts}
        missing = sorted(expected_scripts - present)
        if missing:
            failures.append("Packaged Leonardo shell audit missing scripts: " + ", ".join(missing))
    script_files = payload.get("script_files", [])
    if not isinstance(script_files, list) or not script_files:
        failures.append("Packaged Leonardo shell audit has no script_files list")
    else:
        rows_by_path = {
            str(row.get("path", "")).replace("\\", "/"): row
            for row in script_files
            if isinstance(row, dict)
        }
        expected_scripts = LEONARDO_SCRIPT_PATHS
        for script in sorted(expected_scripts):
            row = rows_by_path.get(script)
            if not row:
                failures.append(f"Packaged Leonardo shell audit missing script hash row: {script}")
                continue
            if row.get("exists") is not True:
                failures.append(f"Packaged Leonardo shell audit script did not exist: {script}")
            if _as_int(row.get("bytes", 0)) <= 0:
                failures.append(f"Packaged Leonardo shell audit script has no bytes: {script}")
            digest = str(row.get("sha256", "") or "")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                failures.append(f"Packaged Leonardo shell audit script hash is invalid for {script}")
                continue
            data, data_failures = _read_package_evidence_bytes(package_dir, Path(script))
            failures.extend(data_failures)
            if data_failures:
                continue
            actual_digest = hashlib.sha256(data).hexdigest()
            if actual_digest != digest:
                failures.append(
                    f"Packaged Leonardo script hash mismatch for {script}: "
                    f"shell audit={digest}, package evidence={actual_digest}"
                )
    return failures


def _check_packaged_readiness_evidence(
    package_dir: Path,
    min_generated_per_family: int,
    max_generated_per_family: int,
    min_reranker_count: int,
    min_train_epochs: int,
    expected_run_profile: str,
    require_eval: bool,
    require_source_bundle_proof: bool,
) -> list[str]:
    payload, failures = _read_package_evidence_json(package_dir, Path("leonardo_readiness.json"))
    if failures:
        return failures
    launch_commands_text, launch_command_failures = _read_package_evidence_text(
        package_dir,
        Path("leonardo_launch_commands.sh"),
    )
    if launch_command_failures:
        failures.extend(launch_command_failures)
    else:
        for needle, label in {
            "sbatch scripts/leonardo_probe.sh": "probe launch command",
            "scripts/leonardo_full_pipeline.sh": "full-pipeline launch command",
            "--dependency=afterok:${PROBE_JOB}": "generation dependency on probe",
            "--dependency=afterok:${GEN_JOB}": "training dependency on generation",
            "--dependency=afterok:${TRAIN_JOB}": "finalization dependency on training",
            "verify_returned_package": "returned-package verification command",
            "run_evidence_report": "objective evidence-report command",
        }.items():
            if needle not in launch_commands_text:
                failures.append(f"Packaged Leonardo launch commands missing {label}")
        commands_out = str(payload.get("commands_out", "") or "").replace("\\", "/")
        if Path(commands_out).name != "leonardo_launch_commands.sh":
            failures.append(
                "Packaged Leonardo readiness commands_out is "
                f"{commands_out!r}; expected leonardo_launch_commands.sh"
            )
    if payload.get("passed") is not True:
        failures.append("Packaged Leonardo readiness did not pass")
    if payload.get("defer_eval_staging") is True:
        failures.append("Packaged Leonardo readiness still has defer_eval_staging=true; rerun readiness after staging official eval CSVs")
    readiness_failures = payload.get("failures", [])
    if isinstance(readiness_failures, list) and readiness_failures:
        failures.append("Packaged Leonardo readiness recorded failures: " + "; ".join(map(str, readiness_failures)))
    if expected_run_profile:
        actual_profile = str(payload.get("count_profile", "") or "")
        if actual_profile != expected_run_profile:
            failures.append(
                f"Packaged Leonardo readiness count_profile is {actual_profile!r}; "
                f"expected {expected_run_profile!r}"
            )
    if require_eval and payload.get("require_eval") is not True:
        failures.append("Packaged Leonardo readiness did not require eval inputs")
    if require_source_bundle_proof and payload.get("require_source_bundle") is not True:
        failures.append("Packaged Leonardo readiness did not require source-bundle proof")
    if payload.get("require_source_bundle") is True:
        selftest_payload, selftest_failures = _read_package_evidence_json(
            package_dir,
            Path("source_bundle_proof_selftest.json"),
        )
        failures.extend(selftest_failures)
        if not selftest_failures:
            if selftest_payload.get("passed") is not True:
                failures.append("Packaged source-bundle proof self-test did not pass")
            recorded_failures = selftest_payload.get("failures", [])
            if isinstance(recorded_failures, list) and recorded_failures:
                failures.append(
                    "Packaged source-bundle proof self-test recorded failures: "
                    + "; ".join(map(str, recorded_failures))
                )
        source_bundle = payload.get("source_bundle", {})
        if not isinstance(source_bundle, dict):
            failures.append("Packaged Leonardo readiness required source bundle but has no source_bundle object")
        else:
            if source_bundle.get("verified") is not True:
                failures.append("Packaged Leonardo readiness required source bundle but source_bundle.verified is not true")
            bundle_failures = source_bundle.get("failures", [])
            if isinstance(bundle_failures, list) and bundle_failures:
                failures.append(
                    "Packaged Leonardo readiness source_bundle recorded failures: "
                    + "; ".join(map(str, bundle_failures))
                )
            bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
            if not _valid_sha256(bundle_hash):
                failures.append("Packaged Leonardo readiness source_bundle has no valid bundle_sha256")
            if _as_int(source_bundle.get("manifest_file_count", 0)) <= 0:
                failures.append("Packaged Leonardo readiness source_bundle has no manifest_file_count")
        if launch_commands_text:
            for needle, label in {
                "REQUIRE_SOURCE_BUNDLE=1": "required source-bundle proof export",
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh": (
                    "split generation source-bundle export"
                ),
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh": (
                    "split training source-bundle export"
                ),
            }.items():
                if needle not in launch_commands_text:
                    failures.append(f"Packaged Leonardo launch commands missing {label}")
        if launch_commands_text and "source_bundle_proof_selftest" not in launch_commands_text:
            failures.append("Packaged Leonardo launch commands missing source-bundle proof self-test command")
    if require_eval and payload.get("require_eval") is True and launch_commands_text and "REQUIRE_EVAL=1" not in launch_commands_text:
        failures.append("Packaged Leonardo launch commands missing required eval export")
    readiness_count = _as_int(payload.get("count_per_family", 0))
    readiness_epochs = _as_int(payload.get("epochs", 0))
    readiness_batch_size = _as_int(payload.get("batch_size", 0))
    readiness_reranker_valid_per_family = _as_int(payload.get("reranker_valid_per_family", 0))
    commands = payload.get("commands", {})
    if not isinstance(commands, dict) or not commands.get("full_pipeline"):
        failures.append("Packaged Leonardo readiness has no full_pipeline command list")
    if isinstance(commands, dict):
        command_text = "\n".join(
            str(command)
            for command_list in commands.values()
            if isinstance(command_list, list)
            for command in command_list
        )
        if launch_commands_text:
            for key, label in (
                ("full_pipeline", "recorded full-pipeline launch command"),
                ("split_jobs_with_dependencies", "recorded dependency-safe split launch command"),
            ):
                command_list = commands.get(key, [])
                if isinstance(command_list, list):
                    for command in command_list:
                        if str(command) and str(command) not in launch_commands_text:
                            failures.append(
                                f"Packaged Leonardo launch commands missing {label}: {command}"
                            )
        command_needles = {
            f"COUNT_PER_FAMILY={readiness_count}": "recorded COUNT_PER_FAMILY",
            f"EPOCHS={readiness_epochs}": "recorded EPOCHS",
            f"RERANKER_VALID_PER_FAMILY={readiness_reranker_valid_per_family}": (
                "recorded RERANKER_VALID_PER_FAMILY"
            ),
        }
        if readiness_batch_size > 0:
            command_needles[f"BATCH_SIZE={readiness_batch_size}"] = "recorded BATCH_SIZE"
        if require_eval and payload.get("require_eval") is True:
            command_needles["REQUIRE_EVAL=1"] = "required eval export"
        if require_source_bundle_proof and payload.get("require_source_bundle") is True:
            command_needles["REQUIRE_SOURCE_BUNDLE=1"] = "required source-bundle proof export"
            command_needles["REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh"] = (
                "split generation source-bundle export"
            )
            command_needles["REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh"] = (
                "split training source-bundle export"
            )
        for needle, label in command_needles.items():
            if needle not in command_text:
                failures.append(f"Packaged Leonardo readiness launch commands missing {label}: {needle}")
        dependency_commands = commands.get("split_jobs_with_dependencies", [])
        if not isinstance(dependency_commands, list) or not dependency_commands:
            failures.append("Packaged Leonardo readiness has no dependency-safe split-job command list")
        else:
            dependency_text = "\n".join(str(command) for command in dependency_commands)
            dependency_needles = {
                "--parsable": "parsable Slurm job id capture",
                "--export=ALL,": "explicit Slurm environment preservation",
                "--dependency=afterok:${PROBE_JOB}": "generation dependency on probe",
                "--dependency=afterok:${GEN_JOB}": "training dependency on generation",
                "--dependency=afterok:${TRAIN_JOB}": "finalization dependency on training",
                "leonardo_finalize.sh": "split-job finalization command",
            }
            for needle, label in dependency_needles.items():
                if needle not in dependency_text:
                    failures.append(f"Packaged Leonardo readiness split dependency commands missing {label}")
    resume_guidance = payload.get("resume_guidance", [])
    if not isinstance(resume_guidance, list) or not resume_guidance:
        failures.append("Packaged Leonardo readiness has no resume_guidance list")
    else:
        resume_text = "\n".join(str(item) for item in resume_guidance)
        resume_needles = {
            "leonardo_probe.sh": "probe-before-final-run guidance",
            "interrupted": "full-pipeline interruption guidance",
            "reuse": "exact artifact reuse guidance",
            "train_scaling_complete": "split-job training completion guidance",
            "leonardo_finalize.sh": "split-job finalization guidance",
            "verify_returned_package": "returned-package verification guidance",
            "run_evidence_report": "objective evidence-report guidance",
        }
        for needle, label in resume_needles.items():
            if needle not in resume_text:
                failures.append(f"Packaged Leonardo readiness resume_guidance missing {label}")
    verification_commands = payload.get("verification_commands", [])
    if not isinstance(verification_commands, list) or not verification_commands:
        failures.append("Packaged Leonardo readiness has no verification_commands list")
    else:
        if not any("verify_returned_package" in str(command) for command in verification_commands):
            failures.append("Packaged Leonardo readiness does not record verify_returned_package command")
        if not any("run_evidence_report" in str(command) for command in verification_commands):
            failures.append("Packaged Leonardo readiness does not record run_evidence_report command")
        if not any("leonardo_return_packet" in str(command) for command in verification_commands):
            failures.append("Packaged Leonardo readiness does not record leonardo_return_packet command")
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
        if payload.get("require_source_bundle") is True:
            if not any("source_bundle_proof_selftest" in str(command) for command in verification_commands):
                failures.append(
                    "Packaged Leonardo readiness verification commands missing source-bundle proof self-test"
                )
            if not any("--require-source-bundle-proof" in command for command in verify_commands):
                failures.append(
                    "Packaged Leonardo readiness verify_returned_package command missing source-bundle proof flag"
                )
            report_needles = {
                "--require-readiness": "readiness evidence flag",
                "--require-source-bundle-proof": "source-bundle proof flag",
                "--prefer-package-evidence": "package evidence precedence flag",
            }
            for needle, label in report_needles.items():
                if not any(needle in command for command in evidence_report_commands):
                    failures.append(f"Packaged Leonardo readiness run_evidence_report command missing {label}")
        else:
            if verify_commands and not any("--no-require-source-bundle-proof" in command for command in verify_commands):
                failures.append(
                    "Packaged Leonardo readiness verify_returned_package command missing explicit no-source-bundle proof flag"
                )
            if evidence_report_commands and not any(
                "--no-require-source-bundle-proof" in command for command in evidence_report_commands
            ):
                failures.append(
                    "Packaged Leonardo readiness run_evidence_report command missing explicit no-source-bundle proof flag"
                )
        if readiness_batch_size > 0:
            batch_flag = f"--required-batch-size {readiness_batch_size}"
            if not any(batch_flag in command for command in verify_commands):
                failures.append(
                    f"Packaged Leonardo readiness verify_returned_package command missing batch-size proof: {batch_flag}"
                )
            if not any(batch_flag in command for command in evidence_report_commands):
                failures.append(
                    f"Packaged Leonardo readiness run_evidence_report command missing batch-size proof: {batch_flag}"
                )
        if not any("--require-final-leonardo-objective" in command for command in verify_commands):
            failures.append(
                "Packaged Leonardo readiness verify_returned_package command missing final Leonardo objective gate"
            )
        if return_packet_commands and not any(
            "--require-final-leonardo-objective" in command for command in return_packet_commands
        ):
            failures.append(
                "Packaged Leonardo readiness leonardo_return_packet command missing final Leonardo objective gate"
            )
        if launch_commands_text:
            for command in verification_commands:
                if str(command) not in launch_commands_text:
                    failures.append(
                        "Packaged Leonardo launch commands script is missing recorded "
                        f"verification command: {command}"
                    )
        if max_generated_per_family:
            exact_flag = (
                f"--min-generated-per-family {max_generated_per_family} "
                f"--max-generated-per-family {max_generated_per_family}"
            )
            profile_flags = [
                f"--count-profile {profile}"
                for profile, count in COUNT_PROFILES.items()
                if count == max_generated_per_family
            ]
            if verify_commands and not any(
                exact_flag in command or any(profile_flag in command for profile_flag in profile_flags)
                for command in verify_commands
            ):
                failures.append(
                    "Packaged Leonardo readiness verify_returned_package command does not match "
                    f"generated count {max_generated_per_family}"
                )
            if evidence_report_commands and not any(
                exact_flag in command or any(profile_flag in command for profile_flag in profile_flags)
                for command in evidence_report_commands
            ):
                failures.append(
                    "Packaged Leonardo readiness run_evidence_report command does not match "
                    f"generated count {max_generated_per_family}"
                )
    if readiness_count < min_generated_per_family:
        failures.append(
            f"Packaged Leonardo readiness count_per_family is {readiness_count}; "
            f"expected at least {min_generated_per_family}"
        )
    if max_generated_per_family and readiness_count != max_generated_per_family:
        failures.append(
            f"Packaged Leonardo readiness count_per_family is {readiness_count}; "
            f"expected {max_generated_per_family}"
        )
    readiness_examples = _as_int(payload.get("reranker_examples", 0))
    if readiness_examples < min_reranker_count:
        failures.append(
            f"Packaged Leonardo readiness reranker_examples is {readiness_examples}; "
            f"expected at least {min_reranker_count}"
        )
    if readiness_epochs < min_train_epochs:
        failures.append(
            f"Packaged Leonardo readiness epochs is {readiness_epochs}; "
            f"expected at least {min_train_epochs}"
        )
    return failures


def _check_packaged_checkpoint_audit_evidence(
    package_dir: Path,
    min_generated_per_family: int,
    max_generated_per_family: int,
    min_train_epochs: int,
    required_batch_size: int,
    expected_source_bundle_sha256: str,
    required_transformer_device: str,
    required_checkpoint_sizes: list[str],
    expected_run_profile: str,
) -> list[str]:
    payload, failures = _read_package_evidence_json(package_dir, Path("checkpoint_audit.json"))
    if failures:
        return failures
    if payload.get("passed") is not True:
        failures.append("Packaged checkpoint audit did not pass")
    audit_failures = payload.get("failures", [])
    if isinstance(audit_failures, list) and audit_failures:
        failures.append("Packaged checkpoint audit recorded failures: " + "; ".join(map(str, audit_failures)))
    failures.extend(_require_threshold(payload, "min_generated_per_family", min_generated_per_family))
    if max_generated_per_family:
        actual_max = _as_int(payload.get("max_generated_per_family", 0))
        if actual_max != max_generated_per_family:
            failures.append(
                f"Packaged checkpoint audit max_generated_per_family is {actual_max}; "
                f"expected {max_generated_per_family}"
            )
    if expected_run_profile:
        actual_profile = str(payload.get("run_profile", "") or "")
        if actual_profile != expected_run_profile:
            failures.append(
                f"Packaged checkpoint audit run_profile is {actual_profile!r}; "
                f"expected {expected_run_profile!r}"
            )
    failures.extend(_require_threshold(payload, "min_train_epochs", min_train_epochs))
    if required_batch_size:
        actual_batch_size = _as_int(payload.get("required_batch_size", 0))
        if actual_batch_size != required_batch_size:
            failures.append(
                f"Packaged checkpoint audit required_batch_size is {actual_batch_size}; "
                f"expected {required_batch_size}"
            )
    if expected_source_bundle_sha256:
        actual_bundle_hash = str(payload.get("source_bundle_sha256", "") or "")
        if actual_bundle_hash != expected_source_bundle_sha256:
            failures.append(
                f"Packaged checkpoint audit source_bundle_sha256 is {actual_bundle_hash!r}; "
                f"expected {expected_source_bundle_sha256!r}"
            )
    failures.extend(_required_sizes_missing(payload, "model_sizes", required_checkpoint_sizes))
    if required_transformer_device:
        actual_device = str(payload.get("required_checkpoint_device", "") or "")
        if actual_device != required_transformer_device:
            failures.append(
                f"Packaged checkpoint audit required_checkpoint_device is {actual_device!r}; "
                f"expected {required_transformer_device!r}"
            )
    expected_counts = payload.get("expected_family_counts", {})
    if not isinstance(expected_counts, dict) or not expected_counts:
        failures.append("Packaged checkpoint audit has no expected_family_counts")
    else:
        missing_families = sorted({"IC", "IGBT", "MOSFET"} - {str(family) for family in expected_counts})
        if missing_families:
            failures.append("Packaged checkpoint audit missing families: " + ", ".join(missing_families))
    if not str(payload.get("corpus_fingerprint", "") or ""):
        failures.append("Packaged checkpoint audit has no corpus_fingerprint")
    return failures


def _check_validation_package_consistency(
    validation_payload: dict[str, object],
    package_payload: dict[str, object],
) -> list[str]:
    if not validation_payload or not package_payload:
        return []
    pairs = [
        ("min_generated_per_family", "required_min_generated_per_family"),
        ("max_generated_per_family", "required_max_generated_per_family"),
        ("min_reranker_count", "required_min_reranker_count"),
        ("min_completion_compare_count", "required_min_completion_compare_count"),
        ("min_train_epochs", "required_min_train_epochs"),
        ("required_batch_size", "required_batch_size"),
    ]
    failures: list[str] = []
    for validation_key, package_key in pairs:
        validation_value = _as_int(validation_payload.get(validation_key, 0))
        package_value = _as_int(package_payload.get(package_key, 0))
        if validation_value != package_value:
            failures.append(
                f"Validation {validation_key} is {validation_value}; "
                f"package {package_key} is {package_value}"
            )
    validation_sizes = {str(size) for size in validation_payload.get("model_sizes", [])}
    package_sizes = {str(size) for size in package_payload.get("required_checkpoint_sizes", [])}
    if validation_sizes != package_sizes:
        failures.append(
            "Validation model_sizes do not match package required_checkpoint_sizes: "
            f"validation={sorted(validation_sizes)}, package={sorted(package_sizes)}"
        )
    validation_require_submissions = validation_payload.get("require_submissions")
    if validation_require_submissions is not True:
        failures.append("Validation summary did not require submissions for packaged run")
    validation_transformer_device = str(validation_payload.get("required_transformer_device", "") or "")
    package_transformer_device = str(package_payload.get("required_transformer_device", "") or "")
    if validation_transformer_device != package_transformer_device:
        failures.append(
            "Validation required_transformer_device does not match package required_transformer_device: "
            f"validation={validation_transformer_device!r}, package={package_transformer_device!r}"
        )
    validation_profile = str(validation_payload.get("run_profile", "") or "")
    package_profile = str(package_payload.get("run_profile", "") or "")
    if validation_profile != package_profile:
        failures.append(
            "Validation run_profile does not match package run_profile: "
            f"validation={validation_profile!r}, package={package_profile!r}"
        )
    validation_requires_checkpoint = _truthy(validation_payload.get("require_selected_checkpoint", False))
    package_requires_checkpoint = _truthy(package_payload.get("require_selected_checkpoint", False))
    if validation_requires_checkpoint != package_requires_checkpoint:
        failures.append(
            "Validation require_selected_checkpoint does not match package require_selected_checkpoint: "
            f"validation={validation_requires_checkpoint}, package={package_requires_checkpoint}"
        )
    return failures


def _check_checkpoint_audit_validation_consistency(
    checkpoint_payload: dict[str, object],
    validation_payload: dict[str, object],
) -> list[str]:
    if not checkpoint_payload or not validation_payload:
        return []
    failures: list[str] = []
    pairs = [
        ("min_generated_per_family", "min_generated_per_family"),
        ("max_generated_per_family", "max_generated_per_family"),
        ("min_train_epochs", "min_train_epochs"),
        ("required_batch_size", "required_batch_size"),
    ]
    for checkpoint_key, validation_key in pairs:
        checkpoint_value = _as_int(checkpoint_payload.get(checkpoint_key, 0))
        validation_value = _as_int(validation_payload.get(validation_key, 0))
        if checkpoint_value != validation_value:
            failures.append(
                f"Checkpoint audit {checkpoint_key} is {checkpoint_value}; "
                f"validation {validation_key} is {validation_value}"
            )
    checkpoint_sizes = {str(size) for size in checkpoint_payload.get("model_sizes", [])}
    validation_sizes = {str(size) for size in validation_payload.get("model_sizes", [])}
    if checkpoint_sizes != validation_sizes:
        failures.append(
            "Checkpoint audit model_sizes do not match validation model_sizes: "
            f"checkpoint={sorted(checkpoint_sizes)}, validation={sorted(validation_sizes)}"
        )
    checkpoint_counts = checkpoint_payload.get("expected_family_counts", {})
    validation_counts = validation_payload.get("expected_family_counts", {})
    if checkpoint_counts != validation_counts:
        failures.append(
            "Checkpoint audit expected_family_counts do not match validation expected_family_counts"
        )
    checkpoint_fingerprint = str(checkpoint_payload.get("corpus_fingerprint", "") or "")
    validation_fingerprint = str(validation_payload.get("corpus_fingerprint", "") or "")
    if checkpoint_fingerprint != validation_fingerprint:
        failures.append("Checkpoint audit corpus_fingerprint does not match validation corpus_fingerprint")
    checkpoint_bundle_hash = str(checkpoint_payload.get("source_bundle_sha256", "") or "")
    validation_bundle_hash = str(validation_payload.get("source_bundle_sha256", "") or "")
    if checkpoint_bundle_hash != validation_bundle_hash:
        failures.append("Checkpoint audit source_bundle_sha256 does not match validation source_bundle_sha256")
    checkpoint_profile = str(checkpoint_payload.get("run_profile", "") or "")
    validation_profile = str(validation_payload.get("run_profile", "") or "")
    if checkpoint_profile != validation_profile:
        failures.append(
            "Checkpoint audit run_profile does not match validation run_profile: "
            f"checkpoint={checkpoint_profile!r}, validation={validation_profile!r}"
        )
    validation_checkpoint_device = str(validation_payload.get("required_checkpoint_device", "") or "")
    if validation_checkpoint_device:
        checkpoint_device = str(checkpoint_payload.get("required_checkpoint_device", "") or "")
        if checkpoint_device != validation_checkpoint_device:
            failures.append(
                "Checkpoint audit required_checkpoint_device does not match validation "
                f"required_checkpoint_device: checkpoint={checkpoint_device!r}, "
                f"validation={validation_checkpoint_device!r}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Final pre-submit audit for a returned Leonardo Track 1 run.")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--package-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "submission_package")
    parser.add_argument("--required-manifest-stage", default="packaged_with_submissions")
    parser.add_argument("--required-checkpoint-sizes", nargs="*", default=["tiny", "small", "medium"])
    parser.add_argument(
        "--required-completion-checkpoint-size",
        default="",
        help="Require completion comparison evidence to use this model-size checkpoint. Defaults to the last required checkpoint size when transformer evidence is required.",
    )
    parser.add_argument(
        "--count-profile",
        choices=sorted(COUNT_PROFILES),
        default=None,
        help="Named generation target: standard=50k/family, max=150k/family.",
    )
    parser.add_argument("--min-generated-per-family", type=int, default=None)
    parser.add_argument("--max-generated-per-family", type=int, default=None)
    parser.add_argument("--min-reranker-count", type=int, default=240)
    parser.add_argument("--min-completion-compare-count", type=int, default=240)
    parser.add_argument("--min-train-epochs", type=int, default=6)
    parser.add_argument(
        "--required-batch-size",
        type=int,
        default=0,
        help="Exact training batch_size that packaged validation/checkpoint evidence must prove. Zero disables the check.",
    )
    parser.add_argument(
        "--required-transformer-device",
        default="cuda",
        help="Require validation and packaging to prove this comparison/inference transformer device.",
    )
    parser.add_argument(
        "--require-selected-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require validation and packaging to prove a checkpoint reranker was selected.",
    )
    parser.add_argument(
        "--require-preflight-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require validation and packaging to prove CUDA preflight/checkpoint evidence.",
    )
    parser.add_argument(
        "--require-preflight-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require validation and packaging to prove eval preflight evidence.",
    )
    parser.add_argument(
        "--require-generated-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require generated CSV metadata sidecars in package evidence.",
    )
    parser.add_argument(
        "--require-readiness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require packaged Leonardo readiness launch/verification command evidence.",
    )
    parser.add_argument(
        "--require-source-bundle-proof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require packaged Leonardo readiness and manifest/package evidence to prove source-bundle verification.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    required_completion_checkpoint_size = args.required_completion_checkpoint_size
    if not required_completion_checkpoint_size and args.required_transformer_device and args.required_checkpoint_sizes:
        required_completion_checkpoint_size = str(args.required_checkpoint_sizes[-1])
    expected_completion_checkpoint = _expected_completion_checkpoint(required_completion_checkpoint_size)
    if args.count_profile is not None:
        profile_count = COUNT_PROFILES[args.count_profile]
        if args.min_generated_per_family is None:
            args.min_generated_per_family = profile_count
        elif args.min_generated_per_family != profile_count:
            parser.error(
                f"--count-profile {args.count_profile} expects --min-generated-per-family {profile_count}, "
                f"but got {args.min_generated_per_family}"
            )
        if args.max_generated_per_family is None:
            args.max_generated_per_family = profile_count
        elif args.max_generated_per_family != profile_count:
            parser.error(
                f"--count-profile {args.count_profile} expects --max-generated-per-family {profile_count}, "
                f"but got {args.max_generated_per_family}"
            )
    if args.min_generated_per_family is None:
        args.min_generated_per_family = COUNT_PROFILES["standard"]
    if args.max_generated_per_family is None:
        args.max_generated_per_family = 0
    if (
        args.max_generated_per_family
        and args.min_generated_per_family
        and args.max_generated_per_family < args.min_generated_per_family
    ):
        parser.error("--max-generated-per-family cannot be less than --min-generated-per-family")
    if args.require_source_bundle_proof and not args.require_readiness:
        parser.error("--require-source-bundle-proof requires --require-readiness")
    expected_run_profile = _expected_run_profile(
        args.min_generated_per_family,
        args.max_generated_per_family,
    )
    expected_count_per_family = args.max_generated_per_family or args.min_generated_per_family

    failures: list[str] = []
    validation_path = args.artifacts_dir / "validation_summary.json"
    package_manifest_path = args.package_dir / "package_manifest.json"
    manifest_path = args.artifacts_dir / "run_manifest.json"
    events_path = args.artifacts_dir / "run_manifest_events.jsonl"
    packaged_readiness_payload: dict[str, object] = {}
    if args.require_readiness:
        packaged_readiness_payload, _ = _read_package_evidence_json(
            args.package_dir,
            Path("leonardo_readiness.json"),
        )
    packaged_readiness_source_bundle = (
        packaged_readiness_payload.get("source_bundle", {})
        if isinstance(packaged_readiness_payload, dict)
        else {}
    )
    if not isinstance(packaged_readiness_source_bundle, dict):
        packaged_readiness_source_bundle = {}
    expected_source_bundle_sha256 = (
        str(packaged_readiness_source_bundle.get("bundle_sha256", "") or "")
        if packaged_readiness_payload.get("require_source_bundle") is True
        else ""
    )

    validation_payload: dict[str, object] = {}
    validation_payload, validation_failures = _read_artifact_or_package_json(
        validation_path,
        args.package_dir,
        Path("validation_summary.json"),
        "validation summary",
    )
    failures.extend(validation_failures)
    if validation_payload:
        if validation_payload.get("passed") is not True:
            failures.append("Validation summary did not pass")
        if validation_payload.get("require_submissions") is not True:
            failures.append("Validation summary did not require submissions")
        failures.extend(_require_threshold(
            validation_payload,
            "min_generated_per_family",
            args.min_generated_per_family,
        ))
        if args.max_generated_per_family:
            actual_max = _as_int(validation_payload.get("max_generated_per_family", 0))
            if actual_max != args.max_generated_per_family:
                failures.append(
                    f"Validation summary max_generated_per_family is {actual_max}; "
                    f"expected {args.max_generated_per_family}"
                )
        if expected_run_profile:
            actual_profile = str(validation_payload.get("run_profile", "") or "")
            if actual_profile != expected_run_profile:
                failures.append(
                    f"Validation summary run_profile is {actual_profile!r}; "
                    f"expected {expected_run_profile!r}"
                )
        failures.extend(_require_threshold(
            validation_payload,
            "min_reranker_count",
            args.min_reranker_count,
        ))
        failures.extend(_require_threshold(
            validation_payload,
            "min_completion_compare_count",
            args.min_completion_compare_count,
        ))
        failures.extend(_require_threshold(
            validation_payload,
            "min_train_epochs",
            args.min_train_epochs,
        ))
        if args.required_batch_size:
            actual_batch_size = _as_int(validation_payload.get("required_batch_size", 0))
            if actual_batch_size != args.required_batch_size:
                failures.append(
                    f"Validation summary required_batch_size is {actual_batch_size}; "
                    f"expected {args.required_batch_size}"
                )
        failures.extend(_required_sizes_missing(
            validation_payload,
            "model_sizes",
            args.required_checkpoint_sizes,
        ))
        if args.require_preflight_cuda:
            if validation_payload.get("require_preflight") is not True:
                failures.append("Validation summary did not require preflight")
            if validation_payload.get("require_preflight_torch") is not True:
                failures.append("Validation summary did not require PyTorch preflight")
            if validation_payload.get("require_preflight_cuda") is not True:
                failures.append("Validation summary did not require CUDA preflight")
            if validation_payload.get("required_checkpoint_device") != "cuda":
                failures.append("Validation summary did not require CUDA checkpoint device")
        if args.require_preflight_eval:
            if validation_payload.get("require_preflight") is not True:
                failures.append("Validation summary did not require preflight")
            if validation_payload.get("require_preflight_eval") is not True:
                failures.append("Validation summary did not require eval preflight")
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
                    failures.append("Validation summary did not record required_completion_checkpoint_size")
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

    manifest_payload, manifest_failures = _read_artifact_or_package_json(
        manifest_path,
        args.package_dir,
        Path("run_manifest.json"),
        "run manifest",
    )
    failures.extend(manifest_failures)
    if manifest_payload:
        actual_stage = str(manifest_payload.get("stage", "") or "")
        if actual_stage != args.required_manifest_stage:
            failures.append(
                f"Run manifest stage is {actual_stage!r}; expected {args.required_manifest_stage!r}"
            )
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
                args.require_source_bundle_proof,
            ))
        failures.extend(_check_manifest_artifact_paths(
            manifest_payload.get("artifacts", {}),
            args.require_readiness,
        ))
        if args.require_readiness and expected_completion_checkpoint:
            failures.extend(_manifest_completion_checkpoint_failures(
                manifest_payload,
                expected_completion_checkpoint,
                "Run manifest",
            ))
        if args.require_readiness and packaged_readiness_payload:
            failures.extend(_manifest_source_bundle_failures(
                manifest_payload,
                packaged_readiness_payload,
                "Run manifest",
            ))
    event_log_has_stage, event_log_failures = _event_log_has_stage_with_package_fallback(
        events_path,
        args.package_dir,
        args.required_manifest_stage,
    )
    failures.extend(event_log_failures)
    if not event_log_has_stage:
        failures.append(f"Run manifest event log does not contain stage {args.required_manifest_stage!r}: {events_path}")
    elif expected_run_profile or (args.require_readiness and expected_completion_checkpoint):
        terminal_event, terminal_event_failures = _event_log_stage_payload_with_package_fallback(
            events_path,
            args.package_dir,
            args.required_manifest_stage,
        )
        failures.extend(terminal_event_failures)
        terminal_profile = str(terminal_event.get("run_profile", "") or "")
        if expected_run_profile and terminal_profile != expected_run_profile:
            failures.append(
                "Run manifest terminal event run_profile is "
                f"{terminal_profile!r}; expected {expected_run_profile!r}"
            )
        if args.require_readiness and expected_completion_checkpoint:
            failures.extend(_manifest_completion_checkpoint_failures(
                terminal_event,
                expected_completion_checkpoint,
                "Run manifest terminal event",
            ))
        if args.require_readiness and packaged_readiness_payload:
            failures.extend(_manifest_source_bundle_failures(
                terminal_event,
                packaged_readiness_payload,
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
                args.require_source_bundle_proof,
            ))
    if args.require_readiness:
        generation_event_has_stage, generation_event_failures = _event_log_has_stage_with_package_fallback(
            events_path,
            args.package_dir,
            "generation_prepared",
        )
        failures.extend(generation_event_failures)
        if not generation_event_has_stage:
            failures.append(f"Run manifest event log does not contain stage 'generation_prepared': {events_path}")
        checkpoint_event_has_stage, checkpoint_event_failures = _event_log_has_stage_with_package_fallback(
            events_path,
            args.package_dir,
            "checkpoint_audited",
        )
        failures.extend(checkpoint_event_failures)
        if not checkpoint_event_has_stage:
            failures.append(f"Run manifest event log does not contain stage 'checkpoint_audited': {events_path}")
        generation_before_checkpoint, generation_order_failures = _event_log_stage_order_with_package_fallback(
            events_path,
            args.package_dir,
            "generation_prepared",
            "checkpoint_audited",
        )
        failures.extend(generation_order_failures)
        if not generation_before_checkpoint:
            failures.append(
                "Run manifest event log does not record 'generation_prepared' "
                f"before 'checkpoint_audited': {events_path}"
            )
        checkpoint_before_terminal, checkpoint_order_failures = _event_log_stage_order_with_package_fallback(
            events_path,
            args.package_dir,
            "checkpoint_audited",
            args.required_manifest_stage,
        )
        failures.extend(checkpoint_order_failures)
        if not checkpoint_before_terminal:
            failures.append(
                "Run manifest event log does not record 'checkpoint_audited' "
                f"before {args.required_manifest_stage!r}: {events_path}"
            )
        comparisons_event_has_stage, comparisons_event_failures = _event_log_has_stage_with_package_fallback(
            events_path,
            args.package_dir,
            "comparisons_complete",
        )
        failures.extend(comparisons_event_failures)
        if not comparisons_event_has_stage:
            failures.append(f"Run manifest event log does not contain stage 'comparisons_complete': {events_path}")
        checkpoint_before_comparisons, comparisons_order_failures = _event_log_stage_order_with_package_fallback(
            events_path,
            args.package_dir,
            "checkpoint_audited",
            "comparisons_complete",
        )
        failures.extend(comparisons_order_failures)
        if not checkpoint_before_comparisons:
            failures.append(
                "Run manifest event log does not record 'checkpoint_audited' "
                "before 'comparisons_complete': "
                f"{events_path}"
            )
        comparisons_before_terminal, comparisons_order_failures = _event_log_stage_order_with_package_fallback(
            events_path,
            args.package_dir,
            "comparisons_complete",
            args.required_manifest_stage,
        )
        failures.extend(comparisons_order_failures)
        if not comparisons_before_terminal:
            failures.append(
                "Run manifest event log does not record 'comparisons_complete' "
                f"before {args.required_manifest_stage!r}: {events_path}"
            )
        if packaged_readiness_payload:
            generation_event, generation_event_payload_failures = _event_log_stage_payload_with_package_fallback(
                events_path,
                args.package_dir,
                "generation_prepared",
            )
            failures.extend(generation_event_payload_failures)
            if generation_event:
                failures.extend(_manifest_parameter_failures(
                    generation_event,
                    "Run manifest generation_prepared event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    args.require_source_bundle_proof,
                ))
                failures.extend(_manifest_source_bundle_failures(
                    generation_event,
                    packaged_readiness_payload,
                    "Run manifest generation_prepared event",
                ))
            checkpoint_event, checkpoint_event_payload_failures = _event_log_stage_payload_with_package_fallback(
                events_path,
                args.package_dir,
                "checkpoint_audited",
            )
            failures.extend(checkpoint_event_payload_failures)
            if checkpoint_event:
                failures.extend(_manifest_parameter_failures(
                    checkpoint_event,
                    "Run manifest checkpoint_audited event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    args.require_source_bundle_proof,
                ))
                failures.extend(_manifest_source_bundle_failures(
                    checkpoint_event,
                    packaged_readiness_payload,
                    "Run manifest checkpoint_audited event",
                ))
            comparisons_event, comparisons_event_payload_failures = _event_log_stage_payload_with_package_fallback(
                events_path,
                args.package_dir,
                "comparisons_complete",
            )
            failures.extend(comparisons_event_payload_failures)
            if comparisons_event:
                failures.extend(_manifest_parameter_failures(
                    comparisons_event,
                    "Run manifest comparisons_complete event",
                    expected_run_profile,
                    expected_count_per_family,
                    expected_completion_checkpoint,
                    args.require_preflight_eval,
                    args.require_source_bundle_proof,
                ))
                failures.extend(_manifest_source_bundle_failures(
                    comparisons_event,
                    packaged_readiness_payload,
                    "Run manifest comparisons_complete event",
                ))

    package_payload: dict[str, object] = {}
    package_payload, package_manifest_failures = _read_package_root_json(
        args.package_dir,
        Path("package_manifest.json"),
    )
    failures.extend(package_manifest_failures)
    if package_payload:
            if package_payload.get("require_evidence") is not True:
                failures.append("Package manifest did not require evidence")
            if package_payload.get("required_manifest_stage") != args.required_manifest_stage:
                failures.append(
                    "Package manifest required_manifest_stage is "
                    f"{package_payload.get('required_manifest_stage')!r}; "
                    f"expected {args.required_manifest_stage!r}"
                )
            failures.extend(_require_threshold(
                package_payload,
                "required_min_generated_per_family",
                args.min_generated_per_family,
            ))
            if args.max_generated_per_family:
                actual_max = _as_int(package_payload.get("required_max_generated_per_family", 0))
                if actual_max != args.max_generated_per_family:
                    failures.append(
                        f"Package manifest required_max_generated_per_family is {actual_max}; "
                        f"expected {args.max_generated_per_family}"
                    )
            if expected_run_profile:
                actual_profile = str(package_payload.get("run_profile", "") or "")
                if actual_profile != expected_run_profile:
                    failures.append(
                        f"Package manifest run_profile is {actual_profile!r}; "
                        f"expected {expected_run_profile!r}"
                    )
            failures.extend(_require_threshold(
                package_payload,
                "required_min_reranker_count",
                args.min_reranker_count,
            ))
            failures.extend(_require_threshold(
                package_payload,
                "required_min_completion_compare_count",
                args.min_completion_compare_count,
            ))
            failures.extend(_require_threshold(
                package_payload,
                "required_min_train_epochs",
                args.min_train_epochs,
            ))
            if args.required_batch_size:
                actual_batch_size = _as_int(package_payload.get("required_batch_size", 0))
                if actual_batch_size != args.required_batch_size:
                    failures.append(
                        f"Package manifest required_batch_size is {actual_batch_size}; "
                        f"expected {args.required_batch_size}"
                    )
            failures.extend(_required_sizes_missing(
                package_payload,
                "required_checkpoint_sizes",
                args.required_checkpoint_sizes,
            ))
            package_completion_size = str(package_payload.get("required_completion_checkpoint_size", "") or "")
            if required_completion_checkpoint_size:
                if args.require_readiness and not package_completion_size:
                    failures.append("Package manifest did not record required_completion_checkpoint_size")
                elif (
                    package_completion_size
                    and package_completion_size != required_completion_checkpoint_size
                ):
                    failures.append(
                        "Package manifest required_completion_checkpoint_size is "
                        f"{package_completion_size!r}; expected {required_completion_checkpoint_size!r}"
                    )
            failures.extend(_check_required_submission_entries(package_payload))
            failures.extend(_check_required_evidence_entries(
                package_payload,
                args.required_checkpoint_sizes,
                args.require_generated_metadata,
                args.require_readiness,
                args.require_readiness and args.require_preflight_eval,
            ))
            if args.require_preflight_cuda and package_payload.get("require_preflight_cuda") is not True:
                failures.append("Package manifest did not require CUDA preflight")
            if args.require_preflight_eval and package_payload.get("require_preflight_eval") is not True:
                failures.append("Package manifest did not require eval preflight")
            if args.required_transformer_device:
                actual_device = str(package_payload.get("required_transformer_device", "") or "")
                if actual_device != args.required_transformer_device:
                    failures.append(
                        f"Package manifest required_transformer_device is {actual_device!r}; "
                        f"expected {args.required_transformer_device!r}"
                    )
            if args.require_selected_checkpoint and package_payload.get("require_selected_checkpoint") is not True:
                failures.append("Package manifest did not require selected checkpoint reranker")
            if args.require_generated_metadata and package_payload.get("require_generated_metadata") is not True:
                failures.append("Package manifest did not require generated metadata")
            if args.require_readiness and package_payload.get("require_readiness") is not True:
                failures.append("Package manifest did not require Leonardo readiness evidence")
            if args.require_readiness and packaged_readiness_payload:
                failures.extend(_package_source_bundle_failures(
                    package_payload,
                    packaged_readiness_payload,
                ))

    failures.extend(_check_packaged_execution_evidence(
        args.package_dir,
        args.required_transformer_device,
        args.require_selected_checkpoint,
        args.required_checkpoint_sizes,
        required_completion_checkpoint_size,
        args.required_batch_size,
        expected_source_bundle_sha256,
    ))
    if args.require_preflight_cuda or args.require_preflight_eval:
        failures.extend(_check_packaged_preflight_evidence(
            args.package_dir,
            args.require_preflight_cuda,
            args.require_preflight_eval,
        ))
    if args.require_readiness and args.require_preflight_eval and packaged_readiness_payload:
        failures.extend(_check_packaged_eval_staging_evidence(
            args.package_dir,
            packaged_readiness_payload,
        ))
    failures.extend(_check_packaged_corpus_audit_evidence(
        args.package_dir,
        args.min_generated_per_family,
        args.max_generated_per_family,
        args.require_generated_metadata,
    ))
    failures.extend(_check_packaged_shell_audit_evidence(args.package_dir))
    if args.require_readiness and args.require_source_bundle_proof and packaged_readiness_payload:
        failures.extend(_packaged_scripts_source_bundle_failures(
            args.package_dir,
            packaged_readiness_payload,
        ))
        failures.extend(_packaged_source_snapshot_source_bundle_failures(
            args.package_dir,
            packaged_readiness_payload,
        ))
    if args.require_readiness:
        failures.extend(_check_packaged_readiness_evidence(
            args.package_dir,
            args.min_generated_per_family,
            args.max_generated_per_family,
            args.min_reranker_count,
            args.min_train_epochs,
            expected_run_profile,
            args.require_preflight_eval,
            args.require_source_bundle_proof,
        ))
        failures.extend(_check_packaged_checkpoint_audit_evidence(
            args.package_dir,
            args.min_generated_per_family,
            args.max_generated_per_family,
            args.min_train_epochs,
            args.required_batch_size,
            expected_source_bundle_sha256,
            args.required_transformer_device,
            args.required_checkpoint_sizes,
            expected_run_profile,
        ))
        checkpoint_audit_payload, checkpoint_audit_failures = _read_package_evidence_json(
            args.package_dir,
            Path("checkpoint_audit.json"),
        )
        if not checkpoint_audit_failures:
            failures.extend(_check_checkpoint_audit_validation_consistency(
                checkpoint_audit_payload,
                validation_payload,
            ))
    failures.extend(_check_validation_package_consistency(validation_payload, package_payload))
    failures.extend(verify_package(args.package_dir))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "artifacts_dir": str(args.artifacts_dir),
        "package_dir": str(args.package_dir),
        "package_zip_sha256": _file_sha256(args.package_dir / "track1_submission.zip")
        if (args.package_dir / "track1_submission.zip").exists()
        else "",
        "package_sidecar_sha256": _file_sha256(args.package_dir / "track1_submission.zip.sha256")
        if (args.package_dir / "track1_submission.zip.sha256").exists()
        else "",
        "package_manifest_sha256": _package_root_entry_sha256(args.package_dir, Path("package_manifest.json")),
        "required_manifest_stage": args.required_manifest_stage,
        "required_checkpoint_sizes": args.required_checkpoint_sizes,
        "required_completion_checkpoint_size": required_completion_checkpoint_size,
        "count_profile": args.count_profile or expected_run_profile or "custom",
        "run_profile": expected_run_profile,
        "min_generated_per_family": args.min_generated_per_family,
        "max_generated_per_family": args.max_generated_per_family,
        "min_reranker_count": args.min_reranker_count,
        "min_completion_compare_count": args.min_completion_compare_count,
        "min_train_epochs": args.min_train_epochs,
        "required_batch_size": args.required_batch_size,
        "required_transformer_device": args.required_transformer_device,
        "require_selected_checkpoint": args.require_selected_checkpoint,
        "require_preflight_cuda": args.require_preflight_cuda,
        "require_preflight_eval": args.require_preflight_eval,
        "require_generated_metadata": args.require_generated_metadata,
        "require_readiness": args.require_readiness,
        "require_source_bundle_proof": args.require_source_bundle_proof,
        "failures": failures,
    }
    out_path = args.out or (args.artifacts_dir / "final_audit_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")

    if failures:
        print("Final audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)

    print("Final audit passed")
    print(f"Artifacts: {args.artifacts_dir}")
    print(f"Package: {args.package_dir / 'track1_submission.zip'}")


if __name__ == "__main__":
    main()
