from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .run_profiles import COUNT_PROFILES, profile_for_count


FAMILIES = ("IC", "IGBT", "MOSFET")
DEFAULT_CHECKPOINT_SIZES = ("tiny", "small", "medium")
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


def _read_json_if_exists(path: Path) -> tuple[dict[str, object], str]:
    if not path.exists():
        return {}, ""
    try:
        return _read_json(path), str(path)
    except (OSError, json.JSONDecodeError):
        return {}, str(path)


def _read_package_json(package_dir: Path, rel_path: Path) -> tuple[dict[str, object], str]:
    for candidate in (package_dir / rel_path, package_dir / "evidence" / rel_path):
        payload, source = _read_json_if_exists(candidate)
        if payload:
            return payload, source
    zip_path = package_dir / "track1_submission.zip"
    if zip_path.exists():
        member = str(rel_path).replace("\\", "/")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if member in zf.namelist():
                    return json.loads(zf.read(member).decode("utf-8-sig")), f"{zip_path}!{member}"
                evidence_member = f"evidence/{member}"
                if evidence_member in zf.namelist():
                    return json.loads(zf.read(evidence_member).decode("utf-8-sig")), f"{zip_path}!{evidence_member}"
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile):
            return {}, str(zip_path)
    return {}, ""


def _read_text_if_exists(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    try:
        return path.read_text(encoding="utf-8-sig"), str(path)
    except OSError:
        return "", str(path)


def _read_package_text(package_dir: Path, rel_path: Path) -> tuple[str, str]:
    for candidate in (package_dir / rel_path, package_dir / "evidence" / rel_path):
        text, source = _read_text_if_exists(candidate)
        if text:
            return text, source
    zip_path = package_dir / "track1_submission.zip"
    if zip_path.exists():
        member = str(rel_path).replace("\\", "/")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if member in zf.namelist():
                    return zf.read(member).decode("utf-8-sig"), f"{zip_path}!{member}"
                evidence_member = f"evidence/{member}"
                if evidence_member in zf.namelist():
                    return zf.read(evidence_member).decode("utf-8-sig"), f"{zip_path}!{evidence_member}"
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile):
            return "", str(zip_path)
    return "", ""


def _read_package_bytes(package_dir: Path, rel_path: Path) -> tuple[bytes, str]:
    for candidate in (package_dir / rel_path, package_dir / "evidence" / rel_path):
        if candidate.exists():
            try:
                return candidate.read_bytes(), str(candidate)
            except OSError:
                return b"", str(candidate)
    zip_path = package_dir / "track1_submission.zip"
    if zip_path.exists():
        member = str(rel_path).replace("\\", "/")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if member in zf.namelist():
                    return zf.read(member), f"{zip_path}!{member}"
                evidence_member = f"evidence/{member}"
                if evidence_member in zf.namelist():
                    return zf.read(evidence_member), f"{zip_path}!{evidence_member}"
        except (OSError, zipfile.BadZipFile):
            return b"", str(zip_path)
    return b"", ""


def _read_evidence_json(
    artifacts_dir: Path,
    package_dir: Path,
    rel_path: Path,
    prefer_package: bool,
) -> tuple[dict[str, object], str]:
    if prefer_package:
        package_payload, package_source = _read_package_json(package_dir, rel_path)
        if package_payload:
            return package_payload, package_source
    payload, source = _read_json_if_exists(artifacts_dir / rel_path)
    if payload:
        return payload, source
    return _read_package_json(package_dir, rel_path)


def _read_evidence_text(
    artifacts_dir: Path,
    package_dir: Path,
    rel_path: Path,
    prefer_package: bool,
) -> tuple[str, str]:
    if prefer_package:
        package_text, package_source = _read_package_text(package_dir, rel_path)
        if package_text:
            return package_text, package_source
    text, source = _read_text_if_exists(artifacts_dir / rel_path)
    if text:
        return text, source
    return _read_package_text(package_dir, rel_path)


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _checkpoint_size(path_text: object) -> str:
    text = str(path_text or "").replace("\\", "/").strip()
    parts = [part for part in text.split("/") if part]
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        if index + 1 < len(parts):
            return parts[index + 1]
    path = Path(text)
    return path.parent.name if text else ""


def _checkpoint_identity(path_text: object) -> str:
    text = str(path_text or "").replace("\\", "/").strip()
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    if "checkpoints" in parts:
        index = parts.index("checkpoints")
        return "/".join(parts[index:])
    return text


def _check(name: str, passed: bool, detail: str, evidence: str = "") -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "detail": detail,
        "evidence": evidence,
    }


def _expected_run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    if max_generated_per_family and max_generated_per_family != min_generated_per_family:
        return ""
    count = max_generated_per_family or min_generated_per_family
    return profile_for_count(count) if count > 0 else ""


def _corpus_checks(
    payload: dict[str, object],
    source: str,
    min_generated_per_family: int,
    max_generated_per_family: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    family_counts: dict[str, int] = {}
    for row in payload.get("families", []):
        if not isinstance(row, dict):
            continue
        family = str(row.get("family", "") or "")
        if family:
            family_counts[family] = _as_int(row.get("generated_sequences", 0))
    checks = [_check("corpus audit present", bool(payload), source or "missing", source)]
    audit_failures = payload.get("failures", [])
    checks.append(_check(
        "corpus audit passed",
        isinstance(audit_failures, list) and not audit_failures,
        "; ".join(map(str, audit_failures)) if isinstance(audit_failures, list) and audit_failures else "no failures",
        source,
    ))
    for family in FAMILIES:
        count = family_counts.get(family, 0)
        upper_ok = not max_generated_per_family or count <= max_generated_per_family
        checks.append(_check(
            f"{family} generated corpus target",
            count >= min_generated_per_family and upper_ok,
            f"generated_sequences={count}, expected {min_generated_per_family}"
            + (f"-{max_generated_per_family}" if max_generated_per_family else "+"),
            source,
        ))
    fingerprint = str(payload.get("corpus_fingerprint", "") or "")
    checks.append(_check("corpus fingerprint recorded", bool(fingerprint), fingerprint or "missing", source))
    files = payload.get("files", [])
    if isinstance(files, list):
        for row in files:
            if not isinstance(row, dict) or str(row.get("source", "") or "") != "generated":
                continue
            path = str(row.get("path", "") or "")
            file_hash = str(row.get("file_sha256", "") or "")
            metadata_hash = str(row.get("metadata_output_sha256", "") or "")
            checks.append(_check(
                f"generated metadata hash binding for {Path(path).name or 'generated file'}",
                len(file_hash) == 64 and metadata_hash == file_hash,
                f"metadata_output_sha256={metadata_hash or 'missing'}, file_sha256={file_hash or 'missing'}",
                source,
            ))
    return checks, family_counts


def _checkpoint_checks(
    validation: dict[str, object],
    validation_source: str,
    checkpoint: dict[str, object],
    checkpoint_source: str,
    required_sizes: list[str],
    min_train_epochs: int,
    required_device: str,
    required_batch_size: int,
    expected_source_bundle_sha256: str,
) -> list[dict[str, object]]:
    checks = [
        _check("validation passed", validation.get("passed") is True, str(validation.get("passed")), validation_source),
        _check("checkpoint audit passed", checkpoint.get("passed") is True, str(checkpoint.get("passed")), checkpoint_source),
    ]
    validation_sizes = {str(size) for size in validation.get("model_sizes", [])}
    checkpoint_sizes = {str(size) for size in checkpoint.get("model_sizes", [])}
    for size in required_sizes:
        checks.append(_check(
            f"{size} validation evidence",
            size in validation_sizes,
            f"validation model_sizes={sorted(validation_sizes)}",
            validation_source,
        ))
        checks.append(_check(
            f"{size} checkpoint audit evidence",
            size in checkpoint_sizes,
            f"checkpoint model_sizes={sorted(checkpoint_sizes)}",
            checkpoint_source,
        ))
    actual_epochs = _as_int(checkpoint.get("min_train_epochs", validation.get("min_train_epochs", 0)))
    checks.append(_check(
        "minimum training epochs",
        actual_epochs >= min_train_epochs,
        f"min_train_epochs={actual_epochs}, expected at least {min_train_epochs}",
        checkpoint_source or validation_source,
    ))
    actual_device = str(checkpoint.get("required_checkpoint_device", validation.get("required_checkpoint_device", "")) or "")
    checks.append(_check(
        "checkpoint device evidence",
        not required_device or actual_device == required_device,
        f"required_checkpoint_device={actual_device!r}, expected {required_device!r}",
        checkpoint_source or validation_source,
    ))
    if required_batch_size:
        checkpoint_batch_size = _as_int(checkpoint.get("required_batch_size", 0))
        validation_batch_size = _as_int(validation.get("required_batch_size", 0))
        checks.append(_check(
            "checkpoint batch-size evidence",
            checkpoint_batch_size == required_batch_size and validation_batch_size == required_batch_size,
            (
                f"checkpoint required_batch_size={checkpoint_batch_size}, "
                f"validation required_batch_size={validation_batch_size}, "
                f"expected {required_batch_size}"
            ),
            checkpoint_source or validation_source,
        ))
    if expected_source_bundle_sha256:
        checkpoint_bundle_hash = str(checkpoint.get("source_bundle_sha256", "") or "")
        validation_bundle_hash = str(validation.get("source_bundle_sha256", "") or "")
        checks.append(_check(
            "checkpoint source-bundle evidence",
            checkpoint_bundle_hash == expected_source_bundle_sha256
            and validation_bundle_hash == expected_source_bundle_sha256,
            (
                f"checkpoint={checkpoint_bundle_hash or 'missing'}, "
                f"validation={validation_bundle_hash or 'missing'}, "
                f"expected={expected_source_bundle_sha256}"
            ),
            checkpoint_source or validation_source,
        ))
    return checks


def _checkpoint_artifact_hash_checks(
    package_dir: Path,
    required_sizes: list[str],
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for size in required_sizes:
        summary, summary_source = _read_package_json(
            package_dir,
            Path("checkpoints") / size / "train_summary.json",
        )
        log_bytes, log_source = _read_package_bytes(
            package_dir,
            Path("checkpoints") / size / "train_log.json",
        )
        model_sha = str(summary.get("model_sha256", "") or "")
        log_sha = str(summary.get("train_log_sha256", "") or "")
        actual_log_sha = hashlib.sha256(log_bytes).hexdigest() if log_bytes else ""
        source = summary_source or log_source
        checks.extend([
            _check(
                f"{size} packaged train summary present",
                bool(summary),
                summary_source or "missing",
                summary_source,
            ),
            _check(
                f"{size} model hash recorded",
                _valid_sha256(model_sha),
                model_sha or "missing",
                summary_source,
            ),
            _check(
                f"{size} packaged train log present",
                bool(log_bytes),
                log_source or "missing",
                log_source,
            ),
            _check(
                f"{size} train log hash binding",
                _valid_sha256(log_sha) and actual_log_sha == log_sha,
                f"summary={log_sha or 'missing'}, train_log={actual_log_sha or 'missing'}",
                source,
            ),
        ])
    return checks


def _reranker_checks(
    payload: dict[str, object],
    source: str,
    required_sizes: list[str],
    min_reranker_count: int,
    require_selected_checkpoint: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    best = str(payload.get("best_reranker", "") or "")
    best_checkpoint = str(payload.get("best_checkpoint", "") or "")
    runs = [row for row in payload.get("runs", []) if isinstance(row, dict)]
    rows_by_name = {str(row.get("reranker", "") or ""): row for row in runs}
    checks = [
        _check("reranker metrics present", bool(payload), source or "missing", source),
    ]
    if require_selected_checkpoint:
        checks.extend([
            _check("checkpoint reranker selected", bool(best and best != "baseline"), best or "missing", source),
            _check("selected checkpoint recorded", bool(best_checkpoint), best_checkpoint or "missing", source),
            _check(
                "checkpoint-only reranker scope",
                str(payload.get("selection_scope", "") or "") == "checkpoints",
                str(payload.get("selection_scope", "") or ""),
                source,
            ),
        ])
        best_row = rows_by_name.get(best)
        eligible_scores = [
            _as_float(row.get("selection_score", 0.0))
            for row in runs
            if _truthy(row.get("selection_eligible", False))
        ]
        best_score = _as_float(best_row.get("selection_score", 0.0)) if best_row else 0.0
        max_score = max(eligible_scores) if eligible_scores else 0.0
        checks.append(_check(
            "selected reranker score ordering",
            bool(best_row) and bool(eligible_scores) and best_score + 1e-12 >= max_score,
            f"selected={best_score:.6g}, max_eligible={max_score:.6g}",
            source,
        ))
    if min_reranker_count > 0:
        for row in runs:
            label = str(row.get("reranker", "UNKNOWN") or "UNKNOWN")
            next_count = _as_int(row.get("nextstep_count", 0))
            completion_count = _as_int(row.get("completion_count", 0))
            checks.append(_check(
                f"{label} reranker next-step count",
                next_count >= min_reranker_count,
                f"nextstep_count={next_count}, expected at least {min_reranker_count}",
                source,
            ))
            checks.append(_check(
                f"{label} reranker completion count",
                completion_count >= min_reranker_count,
                f"completion_count={completion_count}, expected at least {min_reranker_count}",
                source,
            ))
    if require_selected_checkpoint:
        for size in required_sizes:
            row = rows_by_name.get(size)
            checks.append(_check(
                f"{size} reranker available",
                bool(row) and _truthy(row.get("available", False)) and _truthy(row.get("selection_eligible", False)),
                json.dumps({
                    "available": row.get("available") if row else None,
                    "selection_eligible": row.get("selection_eligible") if row else None,
                    "checkpoint_sha256": row.get("checkpoint_sha256") if row else "",
                }, sort_keys=True),
                source,
            ))
    return checks, {"best_reranker": best, "best_checkpoint": best_checkpoint}


def _completion_checks(
    payload: dict[str, object],
    source: str,
    required_size: str,
    required_device: str,
    min_completion_compare_count: int,
    require_selected_checkpoint: bool,
) -> list[dict[str, object]]:
    checkpoint_size = _checkpoint_size(payload.get("checkpoint_used"))
    modes = payload.get("modes", [])
    exact_match = ""
    mode_count_checks: list[dict[str, object]] = []
    if isinstance(modes, list):
        ensemble = next((row for row in modes if isinstance(row, dict) and row.get("mode") == "ensemble"), None)
        if isinstance(ensemble, dict):
            exact_match = str(ensemble.get("exact_match", ""))
        if min_completion_compare_count > 0:
            for row in modes:
                if not isinstance(row, dict):
                    continue
                mode = str(row.get("mode", "UNKNOWN") or "UNKNOWN")
                count = _as_int(row.get("count", 0))
                mode_count_checks.append(_check(
                    f"{mode} completion comparison count",
                    count >= min_completion_compare_count,
                    f"count={count}, expected at least {min_completion_compare_count}",
                    source,
                ))
    checks = [
        _check("completion comparison present", bool(payload), source or "missing", source),
        _check(
            "completion comparison checkpoint size",
            not require_selected_checkpoint or checkpoint_size == required_size,
            f"checkpoint_size={checkpoint_size!r}, expected {required_size!r}",
            source,
        ),
        _check(
            "completion comparison transformer device",
            not required_device or str(payload.get("transformer_device", "") or "") == required_device,
            f"transformer_device={payload.get('transformer_device')!r}, expected {required_device!r}",
            source,
        ),
        _check(
            "completion exact match recorded",
            exact_match != "",
            f"ensemble exact_match={exact_match or 'missing'}; 0.0 is allowed on tiny smoke only",
            source,
        ),
    ]
    checks.extend(mode_count_checks)
    return checks


def _inference_checks(
    payload: dict[str, object],
    source: str,
    required_device: str,
    require_selected_checkpoint: bool,
) -> list[dict[str, object]]:
    next_rows = _as_int(payload.get("nextstep_rows", 0))
    completion_rows = _as_int(payload.get("completion_rows", 0))
    anomaly_rows = _as_int(payload.get("anomaly_rows", 0))
    return [
        _check("inference summary present", bool(payload), source or "missing", source),
        _check(
            "inference transformer device",
            not required_device or str(payload.get("transformer_device", "") or "") == required_device,
            f"transformer_device={payload.get('transformer_device')!r}, expected {required_device!r}",
            source,
        ),
        _check(
            "inference selected checkpoint gate",
            not require_selected_checkpoint or payload.get("require_selected_checkpoint") is True,
            str(payload.get("require_selected_checkpoint")),
            source,
        ),
        _check(
            "submission rows present",
            next_rows > 0 and completion_rows > 0 and anomaly_rows > 0,
            f"nextstep={next_rows}, completion={completion_rows}, anomaly={anomaly_rows}",
            source,
        ),
    ]


def _selected_inference_checkpoint_checks(
    reranker: dict[str, object],
    reranker_source: str,
    inference: dict[str, object],
    inference_source: str,
    require_selected_checkpoint: bool,
) -> list[dict[str, object]]:
    if not require_selected_checkpoint:
        return []
    best_reranker = str(reranker.get("best_reranker", "") or "")
    best_checkpoint = reranker.get("best_checkpoint")
    best_identity = _checkpoint_identity(best_checkpoint)
    best_row = next(
        (
            row for row in reranker.get("runs", [])
            if isinstance(row, dict) and str(row.get("reranker", "") or "") == best_reranker
        ),
        {},
    )
    best_row_checkpoint_identity = _checkpoint_identity(best_row.get("checkpoint") if isinstance(best_row, dict) else "")
    inference_identity = _checkpoint_identity(inference.get("checkpoint_used"))
    summary_selected_identity = _checkpoint_identity(inference.get("selected_checkpoint"))
    best_sha = str(best_row.get("checkpoint_sha256", "") or "") if isinstance(best_row, dict) else ""
    inference_sha = str(inference.get("checkpoint_sha256", "") or "")
    evidence = inference_source or reranker_source
    checks = [
        _check(
            "selected checkpoint row path",
            bool(best_identity) and bool(best_row_checkpoint_identity) and best_row_checkpoint_identity == best_identity,
            f"best_checkpoint={best_identity or 'missing'}, row_checkpoint={best_row_checkpoint_identity or 'missing'}",
            reranker_source,
        ),
        _check(
            "inference selected checkpoint path",
            bool(best_identity) and bool(inference_identity) and inference_identity == best_identity,
            f"best_checkpoint={best_identity or 'missing'}, checkpoint_used={inference_identity or 'missing'}",
            evidence,
        ),
        _check(
            "inference selected checkpoint summary path",
            bool(best_identity) and bool(summary_selected_identity) and summary_selected_identity == best_identity,
            f"best_checkpoint={best_identity or 'missing'}, selected_checkpoint={summary_selected_identity or 'missing'}",
            inference_source,
        ),
        _check(
            "inference selected checkpoint hash",
            bool(best_sha) and bool(inference_sha) and inference_sha == best_sha,
            f"reranker={best_sha or 'missing'}, inference={inference_sha or 'missing'}",
            evidence,
        ),
    ]
    return checks


def _source_bundle_hash(readiness: dict[str, object]) -> str:
    source_bundle = readiness.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        return ""
    return str(source_bundle.get("bundle_sha256", "") or "")


def _manifest_checks(
    manifest: dict[str, object],
    manifest_source: str,
    readiness: dict[str, object],
    final_audit: dict[str, object],
    final_audit_source: str,
    required_stage: str,
    expected_profile: str,
    expected_completion_checkpoint: str,
    require_readiness: bool,
    require_source_bundle_proof: bool,
) -> list[dict[str, object]]:
    parameters = manifest.get("parameters", {})
    completion_checkpoint = ""
    if isinstance(parameters, dict):
        completion_checkpoint = str(parameters.get("COMPLETION_CHECKPOINT", "") or "").replace("\\", "/")
    source_bundle = manifest.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    expected_bundle_hash = _source_bundle_hash(readiness)
    manifest_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
    checks = [
        _check("run manifest terminal stage", manifest.get("stage") == required_stage, str(manifest.get("stage")), manifest_source),
        _check(
            "run manifest run profile",
            not expected_profile or manifest.get("run_profile") == expected_profile,
            f"{manifest.get('run_profile')!r}" + (f", expected {expected_profile!r}" if expected_profile else ""),
            manifest_source,
        ),
        _check(
            "run manifest completion checkpoint",
            not require_readiness or completion_checkpoint == expected_completion_checkpoint,
            f"{completion_checkpoint!r}, expected {expected_completion_checkpoint!r}",
            manifest_source,
        ),
    ]
    if require_readiness and require_source_bundle_proof:
        checks.extend([
            _check(
                "run manifest source-bundle proof recorded",
                bool(source_bundle) and source_bundle.get("require_source_bundle") is True,
                json.dumps(source_bundle, sort_keys=True),
                manifest_source,
            ),
            _check(
                "run manifest source-bundle verified",
                source_bundle.get("verified") is True and source_bundle.get("readiness_passed") is True,
                json.dumps({
                    "verified": source_bundle.get("verified"),
                    "readiness_passed": source_bundle.get("readiness_passed"),
                }, sort_keys=True),
                manifest_source,
            ),
            _check(
                "run manifest source-bundle hash matches readiness",
                bool(expected_bundle_hash) and manifest_bundle_hash == expected_bundle_hash,
                f"manifest={manifest_bundle_hash or 'missing'}, readiness={expected_bundle_hash or 'missing'}",
                manifest_source,
            ),
        ])
    return checks


def _final_audit_checks(
    final_audit: dict[str, object],
    final_audit_source: str,
    artifacts_dir: Path,
    package_dir: Path,
    expected_profile: str,
    required_stage: str,
    required_checkpoint_sizes: list[str],
    required_completion_size: str,
    min_generated_per_family: int,
    max_generated_per_family: int,
    min_reranker_count: int,
    min_completion_compare_count: int,
    min_train_epochs: int,
    required_batch_size: int,
    required_transformer_device: str,
    require_selected_checkpoint: bool,
    require_readiness: bool,
    require_source_bundle_proof: bool,
) -> list[dict[str, object]]:
    expected_artifacts_dir = str(artifacts_dir).replace("\\", "/")
    actual_artifacts_dir = str(final_audit.get("artifacts_dir", "") or "").replace("\\", "/")
    expected_package_dir = str(package_dir).replace("\\", "/")
    actual_package_dir = str(final_audit.get("package_dir", "") or "").replace("\\", "/")
    expected_package_zip_sha256 = _file_sha256(package_dir / "track1_submission.zip") if (package_dir / "track1_submission.zip").exists() else ""
    actual_package_zip_sha256 = str(final_audit.get("package_zip_sha256", "") or "")
    expected_package_sidecar_sha256 = _file_sha256(package_dir / "track1_submission.zip.sha256") if (package_dir / "track1_submission.zip.sha256").exists() else ""
    actual_package_sidecar_sha256 = str(final_audit.get("package_sidecar_sha256", "") or "")
    expected_package_manifest_sha256 = _package_root_entry_sha256(package_dir, Path("package_manifest.json"))
    actual_package_manifest_sha256 = str(final_audit.get("package_manifest_sha256", "") or "")
    actual_checkpoint_sizes = [
        str(item)
        for item in final_audit.get("required_checkpoint_sizes", [])
        if str(item)
    ]
    return [
        _check(
            "returned-package final audit passed",
            final_audit.get("passed") is True,
            str(final_audit.get("passed", "missing")),
            final_audit_source,
        ),
        _check(
            "final audit artifacts directory matches report",
            actual_artifacts_dir == expected_artifacts_dir,
            f"{final_audit.get('artifacts_dir')!r}, expected {expected_artifacts_dir!r}",
            final_audit_source,
        ),
        _check(
            "final audit package directory matches report",
            actual_package_dir == expected_package_dir,
            f"{final_audit.get('package_dir')!r}, expected {expected_package_dir!r}",
            final_audit_source,
        ),
        _check(
            "final audit package ZIP hash matches current package",
            _valid_sha256(actual_package_zip_sha256)
            and actual_package_zip_sha256 == expected_package_zip_sha256,
            (
                f"{actual_package_zip_sha256 or 'missing'}, "
                f"expected {expected_package_zip_sha256 or 'missing'}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit package manifest hash matches current package",
            _valid_sha256(actual_package_manifest_sha256)
            and actual_package_manifest_sha256 == expected_package_manifest_sha256,
            (
                f"{actual_package_manifest_sha256 or 'missing'}, "
                f"expected {expected_package_manifest_sha256 or 'missing'}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit package sidecar hash matches current package",
            _valid_sha256(actual_package_sidecar_sha256)
            and actual_package_sidecar_sha256 == expected_package_sidecar_sha256,
            (
                f"{actual_package_sidecar_sha256 or 'missing'}, "
                f"expected {expected_package_sidecar_sha256 or 'missing'}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit manifest stage matches report",
            str(final_audit.get("required_manifest_stage", "") or "") == required_stage,
            f"{final_audit.get('required_manifest_stage')!r}, expected {required_stage!r}",
            final_audit_source,
        ),
        _check(
            "final audit run profile matches report",
            not expected_profile or str(final_audit.get("run_profile", "") or "") == expected_profile,
            f"{final_audit.get('run_profile')!r}, expected {expected_profile!r}",
            final_audit_source,
        ),
        _check(
            "final audit generated-count thresholds match report",
            _as_int(final_audit.get("min_generated_per_family", 0)) == min_generated_per_family
            and _as_int(final_audit.get("max_generated_per_family", 0)) == max_generated_per_family,
            (
                f"min={final_audit.get('min_generated_per_family')!r}, "
                f"max={final_audit.get('max_generated_per_family')!r}; "
                f"expected min={min_generated_per_family}, max={max_generated_per_family}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit checkpoint sizes match report",
            actual_checkpoint_sizes == required_checkpoint_sizes,
            f"{actual_checkpoint_sizes!r}, expected {required_checkpoint_sizes!r}",
            final_audit_source,
        ),
        _check(
            "final audit completion checkpoint matches report",
            not require_selected_checkpoint
            or str(final_audit.get("required_completion_checkpoint_size", "") or "") == required_completion_size,
            f"{final_audit.get('required_completion_checkpoint_size')!r}, expected {required_completion_size!r}",
            final_audit_source,
        ),
        _check(
            "final audit reranker/comparison thresholds match report",
            _as_int(final_audit.get("min_reranker_count", 0)) == min_reranker_count
            and _as_int(final_audit.get("min_completion_compare_count", 0)) == min_completion_compare_count,
            (
                f"reranker={final_audit.get('min_reranker_count')!r}, "
                f"completion={final_audit.get('min_completion_compare_count')!r}; "
                f"expected reranker={min_reranker_count}, completion={min_completion_compare_count}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit train/device thresholds match report",
            _as_int(final_audit.get("min_train_epochs", 0)) == min_train_epochs
            and (
                not required_batch_size
                or _as_int(final_audit.get("required_batch_size", 0)) == required_batch_size
            )
            and str(final_audit.get("required_transformer_device", "") or "") == required_transformer_device,
            (
                f"epochs={final_audit.get('min_train_epochs')!r}, "
                f"batch={final_audit.get('required_batch_size')!r}, "
                f"device={final_audit.get('required_transformer_device')!r}; "
                f"expected epochs={min_train_epochs}, batch={required_batch_size}, "
                f"device={required_transformer_device!r}"
            ),
            final_audit_source,
        ),
        _check(
            "final audit readiness/source-bundle flags match report",
            final_audit.get("require_readiness") is require_readiness
            and final_audit.get("require_source_bundle_proof") is require_source_bundle_proof,
            (
                f"readiness={final_audit.get('require_readiness')!r}, "
                f"source_bundle={final_audit.get('require_source_bundle_proof')!r}; "
                f"expected readiness={require_readiness}, source_bundle={require_source_bundle_proof}"
            ),
            final_audit_source,
        ),
    ]


def _event_stage_checks(
    events_text: str,
    events_source: str,
    required_stage: str,
    require_readiness: bool,
    readiness: dict[str, object] | None = None,
    require_source_bundle_proof: bool = False,
    expected_completion_checkpoint: str = "",
) -> list[dict[str, object]]:
    if not require_readiness:
        return []
    positions: dict[str, int] = {}
    payloads: dict[str, dict[str, object]] = {}
    for index, line in enumerate(events_text.splitlines()):
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
            payloads[stage] = payload
    required_stages = ("generation_prepared", "checkpoint_audited", "comparisons_complete", required_stage)
    checks = [
        _check("run manifest event log present", bool(events_text), events_source or "missing", events_source),
    ]
    for stage in required_stages:
        checks.append(_check(
            f"run manifest event stage {stage}",
            stage in positions,
            f"positions={positions}",
            events_source,
        ))
    for earlier_stage, later_stage in (
        ("generation_prepared", "checkpoint_audited"),
        ("checkpoint_audited", "comparisons_complete"),
        ("comparisons_complete", required_stage),
    ):
        checks.append(_check(
            f"run manifest event order {earlier_stage} before {later_stage}",
            earlier_stage in positions and later_stage in positions and positions[earlier_stage] < positions[later_stage],
            f"positions={positions}",
            events_source,
        ))
    if require_source_bundle_proof:
        expected_bundle_hash = _source_bundle_hash(readiness or {})
        for stage in required_stages:
            payload = payloads.get(stage, {})
            source_bundle = payload.get("source_bundle", {}) if isinstance(payload, dict) else {}
            if not isinstance(source_bundle, dict):
                source_bundle = {}
            event_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
            checks.extend([
                _check(
                    f"run manifest event source-bundle proof {stage}",
                    bool(source_bundle) and source_bundle.get("require_source_bundle") is True,
                    json.dumps(source_bundle, sort_keys=True),
                    events_source,
                ),
                _check(
                    f"run manifest event source-bundle verified {stage}",
                    source_bundle.get("verified") is True and source_bundle.get("readiness_passed") is True,
                    json.dumps({
                        "verified": source_bundle.get("verified"),
                        "readiness_passed": source_bundle.get("readiness_passed"),
                    }, sort_keys=True),
                    events_source,
                ),
                _check(
                    f"run manifest event source-bundle hash {stage}",
                    bool(expected_bundle_hash) and event_bundle_hash == expected_bundle_hash,
                    f"event={event_bundle_hash or 'missing'}, readiness={expected_bundle_hash or 'missing'}",
                    events_source,
                ),
            ])
    if readiness:
        readiness_count = int(readiness.get("count_per_family", 0) or 0)
        readiness_profile = str(readiness.get("count_profile", "") or "")
        readiness_requires_eval = readiness.get("require_eval") is True
        readiness_requires_source = readiness.get("require_source_bundle") is True
        for stage in required_stages:
            payload = payloads.get(stage, {})
            parameters = payload.get("parameters", {}) if isinstance(payload, dict) else {}
            if not isinstance(parameters, dict):
                parameters = {}
            stage_profile = str(payload.get("run_profile", "") or "") if isinstance(payload, dict) else ""
            stage_count = str(parameters.get("COUNT_PER_FAMILY", "") or "")
            checks.extend([
                _check(
                    f"run manifest event run profile {stage}",
                    not readiness_profile or stage_profile == readiness_profile,
                    f"{stage_profile!r}, expected {readiness_profile!r}",
                    events_source,
                ),
                _check(
                    f"run manifest event COUNT_PER_FAMILY {stage}",
                    not readiness_count or stage_count == str(readiness_count),
                    f"{stage_count!r}, expected {readiness_count}",
                    events_source,
                ),
            ])
            if readiness_requires_source:
                checks.append(_check(
                    f"run manifest event REQUIRE_SOURCE_BUNDLE {stage}",
                    str(parameters.get("REQUIRE_SOURCE_BUNDLE", "") or "") == "1",
                    str(parameters.get("REQUIRE_SOURCE_BUNDLE", "") or "missing"),
                    events_source,
                ))
            if readiness_requires_eval:
                checks.append(_check(
                    f"run manifest event REQUIRE_EVAL {stage}",
                    str(parameters.get("REQUIRE_EVAL", "") or "") == "1",
                    str(parameters.get("REQUIRE_EVAL", "") or "missing"),
                    events_source,
                ))
            if expected_completion_checkpoint:
                completion_checkpoint = str(parameters.get("COMPLETION_CHECKPOINT", "") or "").replace("\\", "/")
                checks.append(_check(
                    f"run manifest event completion checkpoint {stage}",
                    completion_checkpoint == expected_completion_checkpoint,
                    f"{completion_checkpoint!r}, expected {expected_completion_checkpoint!r}",
                    events_source,
                ))
    return checks


def _package_checks(
    payload: dict[str, object],
    source: str,
    readiness: dict[str, object],
    required_sizes: list[str],
    expected_profile: str,
    required_size: str,
    required_batch_size: int,
    min_reranker_count: int,
    min_completion_compare_count: int,
    min_train_epochs: int,
    required_transformer_device: str,
    require_selected_checkpoint: bool,
    require_preflight_cuda: bool,
    require_preflight_eval: bool,
    require_generated_metadata: bool,
    require_readiness: bool,
    require_source_bundle_proof: bool,
) -> list[dict[str, object]]:
    sizes = {str(size) for size in payload.get("required_checkpoint_sizes", [])}
    source_bundle = payload.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    expected_bundle_hash = _source_bundle_hash(readiness)
    package_bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
    checks = [
        _check("package manifest present", bool(payload), source or "missing", source),
        _check("package requires evidence", payload.get("require_evidence") is True, str(payload.get("require_evidence")), source),
        _check(
            "package requires readiness",
            not require_readiness or payload.get("require_readiness") is True,
            str(payload.get("require_readiness")),
            source,
        ),
        _check(
            "package run profile",
            not expected_profile or str(payload.get("run_profile", "") or "") == expected_profile,
            f"{payload.get('run_profile')!r}" + (f", expected {expected_profile!r}" if expected_profile else ""),
            source,
        ),
        _check(
            "package checkpoint sizes",
            all(size in sizes for size in required_sizes),
            f"required_checkpoint_sizes={sorted(sizes)}",
            source,
        ),
        _check(
            "package completion checkpoint size",
            not require_readiness or str(payload.get("required_completion_checkpoint_size", "") or "") == required_size,
            f"{payload.get('required_completion_checkpoint_size')!r}, expected {required_size!r}",
            source,
        ),
        _check(
            "package requires CUDA preflight",
            not require_preflight_cuda or payload.get("require_preflight_cuda") is True,
            str(payload.get("require_preflight_cuda")),
            source,
        ),
        _check(
            "package requires eval preflight",
            not require_preflight_eval or payload.get("require_preflight_eval") is True,
            str(payload.get("require_preflight_eval")),
            source,
        ),
        _check(
            "package requires selected checkpoint",
            not require_selected_checkpoint or payload.get("require_selected_checkpoint") is True,
            str(payload.get("require_selected_checkpoint")),
            source,
        ),
        _check(
            "package requires generated metadata",
            not require_generated_metadata or payload.get("require_generated_metadata") is True,
            str(payload.get("require_generated_metadata")),
            source,
        ),
        _check(
            "package transformer device",
            not required_transformer_device
            or str(payload.get("required_transformer_device", "") or "") == required_transformer_device,
            f"{payload.get('required_transformer_device')!r}, expected {required_transformer_device!r}",
            source,
        ),
        _check(
            "package reranker threshold",
            _as_int(payload.get("required_min_reranker_count", 0)) >= min_reranker_count,
            f"{payload.get('required_min_reranker_count')!r}, expected at least {min_reranker_count}",
            source,
        ),
        _check(
            "package completion-comparison threshold",
            _as_int(payload.get("required_min_completion_compare_count", 0)) >= min_completion_compare_count,
            (
                f"{payload.get('required_min_completion_compare_count')!r}, "
                f"expected at least {min_completion_compare_count}"
            ),
            source,
        ),
        _check(
            "package train-epoch threshold",
            _as_int(payload.get("required_min_train_epochs", 0)) >= min_train_epochs,
            f"{payload.get('required_min_train_epochs')!r}, expected at least {min_train_epochs}",
            source,
        ),
    ]
    if required_batch_size:
        checks.append(_check(
            "package training batch size",
            _as_int(payload.get("required_batch_size", 0)) == required_batch_size,
            f"{payload.get('required_batch_size')!r}, expected {required_batch_size}",
            source,
        ))
    if require_readiness and require_source_bundle_proof:
        checks.extend([
            _check(
                "package manifest source-bundle proof recorded",
                bool(source_bundle) and source_bundle.get("required") is True,
                json.dumps(source_bundle, sort_keys=True),
                source,
            ),
            _check(
                "package manifest source-bundle verified",
                source_bundle.get("verified") is True and source_bundle.get("readiness_passed") is True,
                json.dumps({
                    "verified": source_bundle.get("verified"),
                    "readiness_passed": source_bundle.get("readiness_passed"),
                }, sort_keys=True),
                source,
            ),
            _check(
                "package manifest source-bundle hash matches readiness",
                bool(expected_bundle_hash) and package_bundle_hash == expected_bundle_hash,
                f"package={package_bundle_hash or 'missing'}, readiness={expected_bundle_hash or 'missing'}",
                source,
            ),
        ])
    return checks


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _rows_by_label(payload: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("label", "") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("label", "") or "")
    }


def _eval_staging_checks(
    staging: dict[str, object],
    staging_source: str,
    readiness: dict[str, object],
    preflight: dict[str, object],
    require_readiness: bool,
    require_preflight_eval: bool,
) -> list[dict[str, object]]:
    if not require_readiness or not require_preflight_eval:
        return []
    checks = [
        _check("eval staging manifest present", bool(staging), staging_source or "missing", staging_source),
        _check("eval staging manifest passed", staging.get("passed") is True, str(staging.get("passed")), staging_source),
    ]
    destinations = _rows_by_label(staging, "destinations")
    readiness_rows = _rows_by_label(readiness, "eval_inputs")
    preflight_rows = _rows_by_label(preflight, "eval_inputs")
    for label in ("valid", "anomaly"):
        row = destinations.get(label, {})
        staged_hash = str(row.get("sha256", "") or "")
        checks.extend([
            _check(
                f"{label} eval staging destination present",
                bool(row) and row.get("exists") is True,
                json.dumps(row, sort_keys=True) if row else "missing",
                staging_source,
            ),
            _check(
                f"{label} eval staging row count",
                _as_int(row.get("rows", 0)) > 0,
                f"rows={row.get('rows', 0)}",
                staging_source,
            ),
            _check(
                f"{label} eval staging hash recorded",
                _valid_sha256(staged_hash),
                staged_hash or "missing",
                staging_source,
            ),
            _check(
                f"{label} readiness eval hash matches staging",
                str(readiness_rows.get(label, {}).get("sha256", "") or "") == staged_hash,
                (
                    f"readiness={readiness_rows.get(label, {}).get('sha256', '') or 'missing'}, "
                    f"staging={staged_hash or 'missing'}"
                ),
                staging_source or "leonardo_readiness.json",
            ),
            _check(
                f"{label} preflight eval hash matches staging",
                str(preflight_rows.get(label, {}).get("sha256", "") or "") == staged_hash,
                (
                    f"preflight={preflight_rows.get(label, {}).get('sha256', '') or 'missing'}, "
                    f"staging={staged_hash or 'missing'}"
                ),
                staging_source or "preflight_full_pipeline.json",
            ),
        ])
    return checks


def _source_manifest_rows(readiness: dict[str, object]) -> dict[str, dict[str, object]]:
    source_bundle = readiness.get("source_bundle", {})
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


def _package_source_identity_checks(
    package_dir: Path,
    readiness: dict[str, object],
    require_readiness: bool,
    require_source_bundle_proof: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not require_readiness or not require_source_bundle_proof:
        return [], {}
    rows = _source_manifest_rows(readiness)
    script_paths = sorted(rel_path for rel_path in rows if rel_path.startswith("scripts/"))
    source_paths = sorted(rel_path for rel_path in rows if rel_path.startswith("industrial_ai/"))
    checks = [
        _check(
            "Source-bundle manifest file hashes recorded",
            bool(rows),
            f"{len(rows)} manifest file rows",
            "leonardo_readiness.json",
        ),
        _check(
            "Source-bundle manifest includes Leonardo scripts",
            bool(script_paths),
            f"{len(script_paths)} script rows",
            "leonardo_readiness.json",
        ),
        _check(
            "Source-bundle manifest includes Python source",
            bool(source_paths),
            f"{len(source_paths)} industrial_ai rows",
            "leonardo_readiness.json",
        ),
    ]

    def check_paths(label: str, paths: list[str], package_rel_path: callable) -> None:
        missing: list[str] = []
        mismatched: list[str] = []
        invalid: list[str] = []
        sources: list[str] = []
        for rel_path in paths:
            row = rows.get(rel_path, {})
            expected_hash = str(row.get("sha256", "") or "")
            if not _valid_sha256(expected_hash):
                invalid.append(rel_path)
                continue
            data, source = _read_package_bytes(package_dir, package_rel_path(rel_path))
            if not data:
                missing.append(rel_path)
                continue
            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                mismatched.append(rel_path)
            if source:
                sources.append(source)
        failures = []
        if invalid:
            failures.append("invalid hashes: " + ", ".join(invalid[:5]))
        if missing:
            failures.append("missing: " + ", ".join(missing[:5]))
        if mismatched:
            failures.append("hash mismatch: " + ", ".join(mismatched[:5]))
        evidence = sources[0] if sources else str(package_dir)
        checks.append(_check(
            label,
            not failures and bool(paths),
            "; ".join(failures) if failures else f"{len(paths)} files match source-bundle manifest",
            evidence,
        ))

    check_paths("Packaged Leonardo scripts match source bundle", script_paths, lambda rel_path: Path(rel_path))
    check_paths(
        "Packaged Python source snapshot matches source bundle",
        source_paths,
        lambda rel_path: SOURCE_SNAPSHOT_PREFIX / Path(rel_path),
    )
    return checks, {
        "manifest_file_count": len(rows),
        "script_count": len(script_paths),
        "source_snapshot_count": len(source_paths),
    }


def _readiness_checks(
    readiness: dict[str, object],
    readiness_source: str,
    selftest: dict[str, object],
    selftest_source: str,
    launch_text: str,
    launch_source: str,
    expected_profile: str,
    require_readiness: bool,
    require_source_bundle_proof: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not require_readiness:
        return [], {}
    source_bundle = readiness.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        source_bundle = {}
    verification_commands = readiness.get("verification_commands", [])
    if not isinstance(verification_commands, list):
        verification_commands = []
    readiness_commands = readiness.get("commands", {})
    command_text = "\n".join(
        str(command)
        for command_list in readiness_commands.values()
        if isinstance(command_list, list)
        for command in command_list
    ) if isinstance(readiness_commands, dict) else ""
    bundle_hash = str(source_bundle.get("bundle_sha256", "") or "")
    return_packet_commands = [
        str(command)
        for command in verification_commands
        if "leonardo_return_packet" in str(command)
    ]
    checks = [
        _check("Leonardo readiness present", bool(readiness), readiness_source or "missing", readiness_source),
        _check("Leonardo readiness passed", readiness.get("passed") is True, str(readiness.get("passed")), readiness_source),
        _check(
            "Leonardo readiness eval staging finalized",
            readiness.get("defer_eval_staging") is not True,
            str(readiness.get("defer_eval_staging", False)),
            readiness_source,
        ),
        _check(
            "Leonardo readiness run profile",
            not expected_profile or str(readiness.get("count_profile", "") or "") == expected_profile,
            f"{readiness.get('count_profile')!r}" + (f", expected {expected_profile!r}" if expected_profile else ""),
            readiness_source,
        ),
        _check(
            "Leonardo launch commands present",
            bool(launch_text),
            launch_source or "missing",
            launch_source,
        ),
        _check(
            "Leonardo launch commands include returned verification",
            not launch_text or (
                "verify_returned_package" in launch_text
                and "run_evidence_report" in launch_text
                and "leonardo_return_packet" in launch_text
            ),
            "verify_returned_package/run_evidence_report/leonardo_return_packet",
            launch_source,
        ),
        _check(
            "Leonardo readiness verification commands recorded",
            any("verify_returned_package" in str(command) for command in verification_commands)
            and any("run_evidence_report" in str(command) for command in verification_commands)
            and any("leonardo_return_packet" in str(command) for command in verification_commands),
            json.dumps(verification_commands),
            readiness_source,
        ),
    ]
    if readiness.get("require_eval") is True:
        checks.extend([
            _check(
                "Leonardo readiness required eval",
                readiness.get("require_eval") is True,
                str(readiness.get("require_eval")),
                readiness_source,
            ),
            _check(
                "Leonardo launch commands require eval",
                "REQUIRE_EVAL=1" in launch_text,
                "REQUIRE_EVAL=1",
                launch_source,
            ),
            _check(
                "Leonardo readiness command lists require eval",
                "REQUIRE_EVAL=1" in command_text,
                "REQUIRE_EVAL=1",
                readiness_source,
            ),
        ])
    if require_source_bundle_proof:
        checks.extend([
            _check(
                "Leonardo readiness required source bundle",
                readiness.get("require_source_bundle") is True,
                str(readiness.get("require_source_bundle")),
                readiness_source,
            ),
            _check(
                "Leonardo source bundle verified",
                source_bundle.get("verified") is True,
                str(source_bundle.get("verified")),
                readiness_source,
            ),
            _check(
                "Leonardo source bundle hash recorded",
                len(bundle_hash) == 64 and all(char in "0123456789abcdef" for char in bundle_hash.lower()),
                bundle_hash or "missing",
                readiness_source,
            ),
            _check(
                "Leonardo launch commands require source-bundle proof",
                "REQUIRE_SOURCE_BUNDLE=1" in launch_text,
                "REQUIRE_SOURCE_BUNDLE=1",
                launch_source,
            ),
            _check(
                "Leonardo launch commands require source-bundle proof for split generation",
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh" in launch_text,
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh",
                launch_source,
            ),
            _check(
                "Leonardo launch commands require source-bundle proof for split training",
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh" in launch_text,
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh",
                launch_source,
            ),
            _check(
                "Leonardo readiness command lists require source-bundle proof for split jobs",
                "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh" in command_text
                and "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh" in command_text,
                "split generate/train REQUIRE_SOURCE_BUNDLE=1",
                readiness_source,
            ),
            _check(
                "Leonardo source-bundle self-test command recorded",
                any("source_bundle_proof_selftest" in str(command) for command in verification_commands),
                json.dumps(verification_commands),
                readiness_source,
            ),
            _check(
                "Leonardo launch commands include source-bundle self-test",
                "source_bundle_proof_selftest" in launch_text,
                "source_bundle_proof_selftest",
                launch_source,
            ),
            _check(
                "Leonardo source-bundle self-test evidence present",
                bool(selftest),
                selftest_source or "missing",
                selftest_source,
            ),
            _check(
                "Leonardo source-bundle self-test passed",
                selftest.get("passed") is True,
                str(selftest.get("passed", "missing")),
                selftest_source,
            ),
        ])
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
        checks.extend([
            _check(
                "Leonardo readiness verify command disables source-bundle proof explicitly",
                not verify_commands
                or any("--no-require-source-bundle-proof" in command for command in verify_commands),
                json.dumps(verify_commands),
                readiness_source,
            ),
            _check(
                "Leonardo readiness evidence-report command disables source-bundle proof explicitly",
                not evidence_report_commands
                or any("--no-require-source-bundle-proof" in command for command in evidence_report_commands),
                json.dumps(evidence_report_commands),
                readiness_source,
            ),
        ])
    checks.append(_check(
        "Leonardo readiness return-packet command requires final objective",
        any("--require-final-leonardo-objective" in command for command in return_packet_commands),
        json.dumps(return_packet_commands),
        readiness_source,
    ))
    if launch_text and verification_commands:
        missing_commands = [str(command) for command in verification_commands if str(command) not in launch_text]
        checks.append(_check(
            "Leonardo launch commands contain recorded verification commands",
            not missing_commands,
            "; ".join(missing_commands) if missing_commands else "all recorded verification commands present",
            launch_source,
        ))
    if launch_text and isinstance(readiness_commands, dict):
        recorded_launch_commands = [
            str(command)
            for key in ("full_pipeline", "split_jobs_with_dependencies")
            for command in (
                readiness_commands.get(key, [])
                if isinstance(readiness_commands.get(key, []), list)
                else []
            )
            if str(command)
        ]
        missing_launch_commands = [
            command for command in recorded_launch_commands if command not in launch_text
        ]
        checks.append(_check(
            "Leonardo launch commands contain recorded readiness launch commands",
            not missing_launch_commands,
            "; ".join(missing_launch_commands)
            if missing_launch_commands
            else "all recorded readiness launch commands present",
            launch_source,
        ))
    return checks, {
        "source": readiness_source,
        "launch_commands_source": launch_source,
        "require_source_bundle": readiness.get("require_source_bundle"),
        "source_bundle_verified": source_bundle.get("verified"),
        "source_bundle_sha256": bundle_hash,
    }


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Run Evidence Report",
        "",
        f"- Created: {payload['created_at']}",
        f"- Objective ready: {payload['objective_ready']}",
        f"- Objective scope: {payload.get('objective_scope', '')}",
        f"- Final Leonardo objective ready: {payload.get('final_leonardo_objective_ready', False)}",
        f"- Artifacts: `{payload['artifacts_dir']}`",
        f"- Package: `{payload['package_dir']}`",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload["checks"]:
        result = "pass" if check["passed"] else "fail"
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | {result} | {detail} |")
    lines.extend([
        "",
        "## Key Evidence",
        "",
        "```json",
        json.dumps(payload["summary"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _is_final_leonardo_objective_scope(args: argparse.Namespace) -> bool:
    if args.min_generated_per_family != args.max_generated_per_family:
        return False
    if args.min_generated_per_family < 50000 or args.min_generated_per_family > 150000:
        return False
    return (
        str(args.required_manifest_stage) == "packaged_with_submissions"
        and list(args.required_checkpoint_sizes) == list(DEFAULT_CHECKPOINT_SIZES)
        and args.min_reranker_count >= 240
        and args.min_completion_compare_count >= 240
        and args.min_train_epochs >= 6
        and args.required_batch_size == 96
        and str(args.required_transformer_device) == "cuda"
        and args.require_selected_checkpoint is True
        and args.require_preflight_cuda is True
        and args.require_preflight_eval is True
        and args.require_generated_metadata is True
        and args.require_readiness is True
        and args.require_source_bundle_proof is True
        and args.prefer_package_evidence is True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whether returned Leonardo evidence satisfies the run objective.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--package-dir", type=Path, default=Path("artifacts") / "submission_package")
    parser.add_argument("--count-profile", choices=sorted(COUNT_PROFILES), default="")
    parser.add_argument("--min-generated-per-family", type=int, default=50000)
    parser.add_argument("--max-generated-per-family", type=int, default=150000)
    parser.add_argument("--min-reranker-count", type=int, default=240)
    parser.add_argument("--min-completion-compare-count", type=int, default=240)
    parser.add_argument("--min-train-epochs", type=int, default=6)
    parser.add_argument("--required-batch-size", type=int, default=0)
    parser.add_argument("--required-checkpoint-sizes", nargs="+", default=list(DEFAULT_CHECKPOINT_SIZES))
    parser.add_argument("--required-transformer-device", default="cuda")
    parser.add_argument("--required-manifest-stage", default="packaged_with_submissions")
    parser.add_argument("--require-selected-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-preflight-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-preflight-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-generated-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-readiness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-source-bundle-proof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prefer-package-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer package evidence over same-named local artifacts when both are present.",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "run_evidence_report.json")
    parser.add_argument("--markdown-out", type=Path, default=Path("artifacts") / "run_evidence_report.md")
    args = parser.parse_args()

    if args.count_profile:
        count = COUNT_PROFILES[args.count_profile]
        args.min_generated_per_family = count
        args.max_generated_per_family = count
    if args.require_source_bundle_proof and not args.require_readiness:
        parser.error("--require-source-bundle-proof requires --require-readiness")

    required_completion_size = str(args.required_checkpoint_sizes[-1])
    expected_completion_checkpoint = f"checkpoints/{required_completion_size}/model.pt"
    expected_profile = _expected_run_profile(args.min_generated_per_family, args.max_generated_per_family)

    prefer_package = args.prefer_package_evidence
    corpus, corpus_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("corpus_audit") / "summary.json",
        prefer_package,
    )
    validation, validation_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("validation_summary.json"),
        prefer_package,
    )
    preflight, preflight_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("preflight_full_pipeline.json"),
        prefer_package,
    )
    checkpoint, checkpoint_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("checkpoint_audit.json"),
        prefer_package,
    )
    reranker, reranker_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("reranker_compare") / "metrics.json",
        prefer_package,
    )
    completion, completion_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("completion_compare") / "metrics.json",
        prefer_package,
    )
    inference, inference_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("inference_summary.json"),
        prefer_package,
    )
    manifest, manifest_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("run_manifest.json"),
        prefer_package,
    )
    events_text, events_source = _read_evidence_text(
        args.artifacts_dir,
        args.package_dir,
        Path("run_manifest_events.jsonl"),
        prefer_package,
    )
    readiness, readiness_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("leonardo_readiness.json"),
        prefer_package,
    )
    source_bundle_selftest, source_bundle_selftest_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("source_bundle_proof_selftest.json"),
        prefer_package,
    )
    eval_staging, eval_staging_source = _read_evidence_json(
        args.artifacts_dir,
        args.package_dir,
        Path("eval_staging_manifest.json"),
        prefer_package,
    )
    launch_text, launch_source = _read_evidence_text(
        args.artifacts_dir,
        args.package_dir,
        Path("leonardo_launch_commands.sh"),
        prefer_package,
    )
    package_manifest, package_source = _read_package_json(args.package_dir, Path("package_manifest.json"))
    final_audit, final_audit_source = _read_json_if_exists(args.artifacts_dir / "final_audit_summary.json")

    checks: list[dict[str, object]] = []
    corpus_checks, family_counts = _corpus_checks(
        corpus,
        corpus_source,
        args.min_generated_per_family,
        args.max_generated_per_family,
    )
    checks.extend(corpus_checks)
    checks.extend(_checkpoint_checks(
        validation,
        validation_source,
        checkpoint,
        checkpoint_source,
        list(args.required_checkpoint_sizes),
        args.min_train_epochs,
        args.required_transformer_device,
        args.required_batch_size,
        _source_bundle_hash(readiness) if args.require_source_bundle_proof else "",
    ))
    checks.extend(_checkpoint_artifact_hash_checks(
        args.package_dir,
        list(args.required_checkpoint_sizes),
    ))
    reranker_checks, reranker_summary = _reranker_checks(
        reranker,
        reranker_source,
        list(args.required_checkpoint_sizes),
        args.min_reranker_count,
        args.require_selected_checkpoint,
    )
    checks.extend(reranker_checks)
    checks.extend(_completion_checks(
        completion,
        completion_source,
        required_completion_size,
        args.required_transformer_device,
        args.min_completion_compare_count,
        args.require_selected_checkpoint,
    ))
    checks.extend(_inference_checks(
        inference,
        inference_source,
        args.required_transformer_device,
        args.require_selected_checkpoint,
    ))
    checks.extend(_selected_inference_checkpoint_checks(
        reranker,
        reranker_source,
        inference,
        inference_source,
        args.require_selected_checkpoint,
    ))
    checks.extend(_manifest_checks(
        manifest,
        manifest_source,
        readiness,
        final_audit,
        final_audit_source,
        args.required_manifest_stage,
        expected_profile,
        expected_completion_checkpoint,
        args.require_readiness,
        args.require_source_bundle_proof,
    ))
    checks.extend(_final_audit_checks(
        final_audit,
        final_audit_source,
        args.artifacts_dir,
        args.package_dir,
        expected_profile,
        args.required_manifest_stage,
        list(args.required_checkpoint_sizes),
        required_completion_size,
        args.min_generated_per_family,
        args.max_generated_per_family,
        args.min_reranker_count,
        args.min_completion_compare_count,
        args.min_train_epochs,
        args.required_batch_size,
        args.required_transformer_device,
        args.require_selected_checkpoint,
        args.require_readiness,
        args.require_source_bundle_proof,
    ))
    checks.extend(_event_stage_checks(
        events_text,
        events_source,
        args.required_manifest_stage,
        args.require_readiness,
        readiness,
        args.require_source_bundle_proof,
        expected_completion_checkpoint,
    ))
    checks.extend(_package_checks(
        package_manifest,
        package_source,
        readiness,
        list(args.required_checkpoint_sizes),
        expected_profile,
        required_completion_size,
        args.required_batch_size,
        args.min_reranker_count,
        args.min_completion_compare_count,
        args.min_train_epochs,
        args.required_transformer_device,
        args.require_selected_checkpoint,
        args.require_preflight_cuda,
        args.require_preflight_eval,
        args.require_generated_metadata,
        args.require_readiness,
        args.require_source_bundle_proof,
    ))
    checks.extend(_eval_staging_checks(
        eval_staging,
        eval_staging_source,
        readiness,
        preflight,
        args.require_readiness,
        args.require_preflight_eval,
    ))
    readiness_checks, readiness_summary = _readiness_checks(
        readiness,
        readiness_source,
        source_bundle_selftest,
        source_bundle_selftest_source,
        launch_text,
        launch_source,
        expected_profile,
        args.require_readiness,
        args.require_source_bundle_proof,
    )
    checks.extend(readiness_checks)
    source_identity_checks, source_identity_summary = _package_source_identity_checks(
        args.package_dir,
        readiness,
        args.require_readiness,
        args.require_source_bundle_proof,
    )
    checks.extend(source_identity_checks)

    objective_ready = all(bool(check["passed"]) for check in checks)
    final_leonardo_scope = _is_final_leonardo_objective_scope(args)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective_ready": objective_ready,
        "objective_scope": "final_leonardo" if final_leonardo_scope else "custom_verification",
        "final_leonardo_objective_ready": objective_ready and final_leonardo_scope,
        "artifacts_dir": str(args.artifacts_dir),
        "package_dir": str(args.package_dir),
        "expected": {
            "min_generated_per_family": args.min_generated_per_family,
            "max_generated_per_family": args.max_generated_per_family,
            "min_reranker_count": args.min_reranker_count,
            "min_completion_compare_count": args.min_completion_compare_count,
            "min_train_epochs": args.min_train_epochs,
            "required_batch_size": args.required_batch_size,
            "required_checkpoint_sizes": args.required_checkpoint_sizes,
            "required_completion_checkpoint": expected_completion_checkpoint,
            "required_manifest_stage": args.required_manifest_stage,
            "required_transformer_device": args.required_transformer_device,
            "require_selected_checkpoint": args.require_selected_checkpoint,
            "require_preflight_cuda": args.require_preflight_cuda,
            "require_preflight_eval": args.require_preflight_eval,
            "require_generated_metadata": args.require_generated_metadata,
            "require_readiness": args.require_readiness,
            "require_source_bundle_proof": args.require_source_bundle_proof,
            "prefer_package_evidence": args.prefer_package_evidence,
            "run_profile": expected_profile,
        },
        "summary": {
            "generated_per_family": family_counts,
            "reranker": reranker_summary,
            "completion_checkpoint_used": completion.get("checkpoint_used", ""),
            "inference_checkpoint_used": inference.get("checkpoint_used", ""),
            "submission_rows": {
                "nextstep": _as_int(inference.get("nextstep_rows", 0)),
                "completion": _as_int(inference.get("completion_rows", 0)),
                "anomaly": _as_int(inference.get("anomaly_rows", 0)),
            },
            "readiness": readiness_summary,
            "package_source_identity": source_identity_summary,
            "run_manifest_source_bundle": manifest.get("source_bundle", {}),
            "sources": {
                "corpus_audit": corpus_source,
                "validation": validation_source,
                "preflight": preflight_source,
                "checkpoint_audit": checkpoint_source,
                "reranker": reranker_source,
                "completion": completion_source,
                "inference": inference_source,
                "run_manifest": manifest_source,
                "run_manifest_events": events_source,
                "leonardo_readiness": readiness_source,
                "source_bundle_proof_selftest": source_bundle_selftest_source,
                "eval_staging": eval_staging_source,
                "leonardo_launch_commands": launch_source,
                "package_manifest": package_source,
                "final_audit": final_audit_source,
            },
        },
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(args.markdown_out, payload)
    print(f"Wrote {args.out}")
    if args.markdown_out:
        print(f"Wrote {args.markdown_out}")
    if not payload["objective_ready"]:
        print("Run evidence report failed:")
        for check in checks:
            if not check["passed"]:
                print(f"- {check['name']}: {check['detail']}")
        raise SystemExit(2)
    print("Run evidence report passed")


if __name__ == "__main__":
    main()
