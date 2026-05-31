from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import FAMILY_FILES
from .hashing import file_sha256
from .paths import PROJECT_ROOT
from .run_profiles import profile_for_count
from .train import MODEL_CONFIGS


REQUIRED_SUBMISSIONS = ["nextstep.csv", "completion.csv", "anomaly.csv"]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _resolve_optional_path(value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def _best_reranker_score_failures(
    payload: dict[str, object],
    best_row: dict[str, object] | None,
) -> list[str]:
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        return ["Reranker metrics runs is not a list"]
    eligible_rows = [row for row in runs if isinstance(row, dict) and _truthy(row.get("selection_eligible", False))]
    if not eligible_rows:
        return ["Reranker metrics has no selection-eligible runs"]
    if best_row is None:
        return []
    best_score = _as_float(best_row.get("selection_score", 0.0))
    max_score = max(_as_float(row.get("selection_score", 0.0)) for row in eligible_rows)
    if best_score + 1e-12 < max_score:
        return [
            "Selected reranker is not the highest-scoring eligible run: "
            f"{best_score:.6g} < {max_score:.6g}"
        ]
    return []


def _required_checkpoint_reranker_failures(
    payload: dict[str, object],
    required_sizes: list[str],
) -> list[str]:
    runs_by_label = {
        str(row.get("reranker", "") or ""): row
        for row in payload.get("runs", [])
        if isinstance(row, dict)
    }
    failures: list[str] = []
    for size in required_sizes:
        row = runs_by_label.get(size)
        if row is None:
            failures.append(f"Reranker metrics missing required checkpoint run: {size}")
            continue
        checkpoint = _resolve_optional_path(row.get("checkpoint"))
        if checkpoint is None:
            failures.append(f"Reranker metrics required run {size} has no checkpoint")
        elif checkpoint.parent.name != size:
            failures.append(
                f"Reranker metrics required run {size} uses checkpoint size {checkpoint.parent.name!r}"
            )
        elif not checkpoint.exists():
            failures.append(f"Reranker metrics required run {size} checkpoint does not exist: {checkpoint}")
        if not _truthy(row.get("available", False)):
            failures.append(f"Reranker metrics required run {size} was not available")
        if not _truthy(row.get("selection_eligible", False)):
            failures.append(f"Reranker metrics required run {size} was not selection eligible")
        checkpoint_sha256 = str(row.get("checkpoint_sha256", "") or "").strip()
        if not checkpoint_sha256:
            failures.append(f"Reranker metrics required run {size} has no checkpoint_sha256")
        elif checkpoint is not None and checkpoint.exists() and checkpoint_sha256 != file_sha256(checkpoint):
            failures.append(f"Reranker metrics required run {size} checkpoint_sha256 does not match file")
    return failures


def _failures_from_preflight(path: Path, require_torch: bool, require_cuda: bool, require_eval: bool) -> list[str]:
    if not path.exists():
        return [f"Missing preflight output: {path}"]
    payload = _read_json(path)
    failures = [f"Preflight recorded failure: {failure}" for failure in payload.get("failures", [])]
    if payload.get("official_generator_loads") is not True:
        failures.append("Preflight did not confirm official generator loading")
    data_files = payload.get("data_files", [])
    if not isinstance(data_files, list) or not data_files:
        failures.append("Preflight has no data file checks")
    else:
        for row in data_files:
            if not isinstance(row, dict):
                continue
            if row.get("exists") is not True:
                failures.append(f"Preflight missing data file: {row.get('path')}")
            if int(row.get("bytes", 0) or 0) <= 0:
                failures.append(f"Preflight saw empty data file: {row.get('path')}")
    torch_info = payload.get("torch", {})
    if not isinstance(torch_info, dict):
        torch_info = {}
    if require_torch and torch_info.get("available") is not True:
        failures.append("Preflight did not confirm PyTorch availability")
    if require_cuda and torch_info.get("cuda_available") is not True:
        failures.append("Preflight did not confirm CUDA availability")
    if require_cuda:
        if int(torch_info.get("cuda_device_count", 0) or 0) <= 0:
            failures.append("Preflight did not report a CUDA device")
        devices = torch_info.get("devices", [])
        if not isinstance(devices, list) or not devices:
            failures.append("Preflight did not report CUDA device details")
    preflight_requires_eval = payload.get("require_eval") is True
    if require_eval and not preflight_requires_eval:
        failures.append("Preflight did not require eval inputs")
    eval_inputs = payload.get("eval_inputs", [])
    if (preflight_requires_eval or require_eval) and isinstance(eval_inputs, list):
        if require_eval and not eval_inputs:
            failures.append("Preflight did not record eval input checks")
        for row in eval_inputs:
            if not isinstance(row, dict):
                continue
            if row.get("exists") is not True:
                failures.append(f"Preflight missing eval input: {row.get('path')}")
            elif int(row.get("bytes", 0) or 0) <= 0:
                failures.append(f"Preflight saw empty eval input: {row.get('path')}")
            elif row.get("missing_columns"):
                failures.append(
                    "Preflight saw eval input with missing columns: "
                    f"{row.get('path')} ({', '.join(map(str, row.get('missing_columns', [])))})"
                )
            elif int(row.get("rows", 0) or 0) <= 0:
                failures.append(f"Preflight saw eval input with no rows: {row.get('path')}")
    return failures


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _source_bundle_expectation(readiness_path: Path) -> tuple[list[str], str]:
    if not readiness_path.exists():
        return [], ""
    try:
        payload = _read_json(readiness_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Leonardo readiness is not readable JSON for checkpoint source-bundle validation: {readiness_path} ({exc})"], ""
    if payload.get("require_source_bundle") is not True:
        return [], ""
    source_bundle = payload.get("source_bundle", {})
    if not isinstance(source_bundle, dict):
        return ["Leonardo readiness required source-bundle proof but has no source_bundle object"], ""
    if source_bundle.get("verified") is not True:
        return ["Leonardo readiness source_bundle.verified is not true"], ""
    bundle_sha256 = str(source_bundle.get("bundle_sha256", "") or "")
    if not bundle_sha256:
        return ["Leonardo readiness source_bundle has no bundle_sha256"], ""
    return [], bundle_sha256


def _run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    count = max_generated_per_family or min_generated_per_family
    if count <= 0:
        return ""
    return profile_for_count(count)


def _corpus_audit_expectations(
    path: Path,
    min_generated_per_family: int,
    max_generated_per_family: int,
) -> tuple[list[str], dict[str, int], str]:
    failures: list[str] = []
    if not path.exists():
        return [f"Missing corpus audit summary: {path}"], {}, ""
    payload = _read_json(path)
    for failure in payload.get("failures", []):
        failures.append(str(failure))
    payload_min = _as_int(payload.get("min_generated_per_family", 0))
    if payload_min < min_generated_per_family:
        failures.append(
            "Corpus audit min_generated_per_family is "
            f"{payload_min}; expected at least {min_generated_per_family}"
        )
    payload_max = _as_int(payload.get("max_generated_per_family", 0))
    if max_generated_per_family and payload_max != max_generated_per_family:
        failures.append(
            "Corpus audit max_generated_per_family is "
            f"{payload_max}; expected {max_generated_per_family}"
        )
    expected_family_counts: dict[str, int] = {}
    seen_families: set[str] = set()
    for row in payload.get("families", []):
        family = str(row.get("family", "UNKNOWN"))
        seen_families.add(family)
        generated = int(row.get("generated_sequences", 0))
        total = int(row.get("total_sequences", 0))
        if total > 0:
            expected_family_counts[family] = total
        if generated < min_generated_per_family:
            failures.append(
                f"{family} has {generated} generated sequences; expected at least {min_generated_per_family}"
            )
        if max_generated_per_family and generated > max_generated_per_family:
            failures.append(
                f"{family} has {generated} generated sequences; expected at most {max_generated_per_family}"
            )
    missing_families = sorted(set(FAMILY_FILES) - seen_families)
    if missing_families:
        failures.append("Corpus audit missing families: " + ", ".join(missing_families))
    missing_counts = sorted(set(FAMILY_FILES) - set(expected_family_counts))
    if missing_counts:
        failures.append("Corpus audit missing positive total counts for families: " + ", ".join(missing_counts))
    return failures, expected_family_counts, str(payload.get("corpus_fingerprint", "") or "")


def _failures_from_checkpoints(
    checkpoint_dir: Path,
    model_sizes: list[str],
    expected_family_counts: dict[str, int],
    expected_corpus_fingerprint: str,
    required_checkpoint_device: str,
    min_train_epochs: int,
    required_batch_size: int = 0,
    expected_source_bundle_sha256: str = "",
) -> list[str]:
    failures: list[str] = []
    min_total_sequences = sum(expected_family_counts.values())
    for model_size in model_sizes:
        run_dir = checkpoint_dir / model_size
        model_path = run_dir / "model.pt"
        summary_path = run_dir / "train_summary.json"
        log_path = run_dir / "train_log.json"
        if not model_path.exists():
            failures.append(f"Missing checkpoint for {model_size}: {model_path}")
        if not summary_path.exists():
            failures.append(f"Missing train summary for {model_size}: {summary_path}")
            continue
        if not log_path.exists():
            failures.append(f"Missing train log for {model_size}: {log_path}")
            log_rows: list[object] = []
        else:
            try:
                loaded_log = _read_json(log_path)
            except (OSError, json.JSONDecodeError):
                failures.append(f"Train log is not readable JSON for {model_size}: {log_path}")
                loaded_log = []
            log_rows = loaded_log if isinstance(loaded_log, list) else []
        summary = _read_json(summary_path)
        epochs = int(summary.get("epochs", 0))
        if epochs < min_train_epochs:
            failures.append(
                f"{model_size} trained for {epochs} epochs; expected at least {min_train_epochs}"
            )
        if log_rows and len(log_rows) < epochs:
            failures.append(
                f"{model_size} train log has {len(log_rows)} rows; expected at least {epochs}"
            )
        if int(summary.get("num_sequences", 0)) != min_total_sequences:
            failures.append(
                f"{model_size} trained on {summary.get('num_sequences', 0)} sequences; "
                f"expected audited corpus total {min_total_sequences}"
            )
        model_sha256 = str(summary.get("model_sha256", "") or "")
        if not model_sha256:
            failures.append(f"{model_size} train summary has no model_sha256")
        elif model_path.exists() and model_sha256 != file_sha256(model_path):
            failures.append(f"{model_size} model_sha256 does not match model.pt")
        train_log_sha256 = str(summary.get("train_log_sha256", "") or "")
        if not train_log_sha256:
            failures.append(f"{model_size} train summary has no train_log_sha256")
        elif log_path.exists() and train_log_sha256 != file_sha256(log_path):
            failures.append(f"{model_size} train_log_sha256 does not match train_log.json")
        summary_fingerprint = str(summary.get("corpus_fingerprint", "") or "")
        if expected_corpus_fingerprint and summary_fingerprint != expected_corpus_fingerprint:
            failures.append(
                f"{model_size} corpus_fingerprint does not match corpus audit fingerprint"
            )
        family_counts = summary.get("family_counts", {})
        if not isinstance(family_counts, dict):
            failures.append(f"{model_size} train summary has no family_counts object")
        else:
            for family, expected in sorted(expected_family_counts.items()):
                count = int(family_counts.get(family, 0))
                if count != expected:
                    failures.append(
                        f"{model_size} trained on {count} {family} sequences; "
                        f"expected audited corpus total {expected}"
                    )
        if summary.get("final_loss") is None:
            failures.append(f"{model_size} train summary has no final_loss")
        if summary.get("model_size") != model_size:
            failures.append(f"{model_size} summary model_size mismatch: {summary.get('model_size')}")
        expected_config = MODEL_CONFIGS.get(model_size)
        if expected_config is not None and summary.get("config") != expected_config:
            failures.append(
                f"{model_size} train summary config does not match expected {model_size} architecture"
            )
        if required_batch_size:
            actual_batch_size = int(summary.get("batch_size", 0))
            if actual_batch_size != required_batch_size:
                failures.append(
                    f"{model_size} trained with batch_size {actual_batch_size}; "
                    f"expected {required_batch_size}"
                )
        if expected_source_bundle_sha256:
            actual_source_bundle_sha = str(summary.get("source_bundle_sha256", "") or "")
            if actual_source_bundle_sha != expected_source_bundle_sha256:
                failures.append(
                    f"{model_size} source_bundle_sha256 {actual_source_bundle_sha!r} "
                    f"does not match readiness source bundle {expected_source_bundle_sha256!r}"
                )
            if summary.get("source_bundle_required") is not True:
                failures.append(f"{model_size} train summary did not require source-bundle proof")
            if summary.get("source_bundle_verified") is not True:
                failures.append(f"{model_size} train summary did not verify source-bundle proof")
        if required_checkpoint_device:
            actual_device = str(summary.get("device", "") or "")
            requested_device = str(summary.get("requested_device", "") or "")
            if actual_device != required_checkpoint_device:
                failures.append(
                    f"{model_size} trained on device {actual_device!r}; expected {required_checkpoint_device!r}"
                )
            if requested_device and requested_device != required_checkpoint_device:
                failures.append(
                    f"{model_size} requested device {requested_device!r}; expected {required_checkpoint_device!r}"
                )
            if summary.get("device_fallback") is True:
                failures.append(f"{model_size} training summary reports device_fallback=true")
    return failures


def _failures_from_rerankers(
    path: Path,
    model_sizes: list[str],
    min_reranker_count: int,
    expected_family_counts: dict[str, int],
    expected_corpus_fingerprint: str,
    required_transformer_device: str,
    require_selected_checkpoint: bool,
) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return [f"Missing reranker comparison metrics: {path}"]
    payload = _read_json(path)
    runs = payload.get("runs", [])
    if not runs:
        return [f"Reranker metrics has no runs: {path}"]
    expected_total = sum(expected_family_counts.values())
    corpus_sequences = int(payload.get("num_corpus_sequences", 0))
    if corpus_sequences != expected_total:
        failures.append(
            f"Reranker comparison used {corpus_sequences} corpus sequences; expected audited corpus total {expected_total}"
        )
    if expected_corpus_fingerprint and str(payload.get("corpus_fingerprint", "") or "") != expected_corpus_fingerprint:
        failures.append("Reranker comparison corpus_fingerprint does not match corpus audit fingerprint")
    if required_transformer_device:
        actual_device = str(payload.get("transformer_device", "") or "")
        if actual_device != required_transformer_device:
            failures.append(
                f"Reranker comparison transformer_device is {actual_device!r}; "
                f"expected {required_transformer_device!r}"
            )
    if require_selected_checkpoint:
        actual_scope = str(payload.get("selection_scope", "") or "")
        if actual_scope != "checkpoints":
            failures.append(
                f"Reranker comparison selection_scope is {actual_scope!r}; expected 'checkpoints'"
            )
        failures.extend(_required_checkpoint_reranker_failures(payload, model_sizes))
    corpus_family_counts = payload.get("corpus_family_counts", {})
    if not isinstance(corpus_family_counts, dict):
        failures.append("Reranker metrics has no corpus_family_counts object")
    else:
        for family, expected in sorted(expected_family_counts.items()):
            count = int(corpus_family_counts.get(family, 0))
            if count != expected:
                failures.append(
                    f"Reranker comparison used {count} {family} sequences; "
                    f"expected audited corpus total {expected}"
                )
    labels = {str(row.get("reranker", "")) for row in runs}
    runs_by_label = {str(row.get("reranker", "")): row for row in runs}
    if "baseline" not in labels:
        failures.append("Reranker metrics missing baseline run")
    missing_sizes = [size for size in model_sizes if size not in labels]
    if missing_sizes:
        failures.append(f"Reranker metrics missing model runs: {', '.join(missing_sizes)}")
    for size in model_sizes:
        row = runs_by_label.get(size)
        if row is not None and not _truthy(row.get("available", False)):
            failures.append(f"Reranker metrics show checkpoint {size} was not available during comparison")
    if min_reranker_count > 0:
        for row in runs:
            label = str(row.get("reranker", "UNKNOWN"))
            next_count = _as_int(row.get("nextstep_count", 0))
            completion_count = _as_int(row.get("completion_count", 0))
            if next_count < min_reranker_count:
                failures.append(
                    f"Reranker {label} nextstep_count is {next_count}; expected at least {min_reranker_count}"
                )
            if completion_count < min_reranker_count:
                failures.append(
                    f"Reranker {label} completion_count is {completion_count}; expected at least {min_reranker_count}"
                )
    best_reranker = str(payload.get("best_reranker", "") or "")
    if best_reranker not in labels:
        failures.append(f"best_reranker is not one of the recorded runs: {payload.get('best_reranker')}")
    best_checkpoint = str(payload.get("best_checkpoint", "") or "").strip()
    if require_selected_checkpoint:
        if best_reranker == "baseline":
            failures.append("Reranker comparison selected baseline; expected a checkpoint reranker")
        if not best_checkpoint:
            failures.append("Reranker comparison did not select a checkpoint")
    best_row = runs_by_label.get(best_reranker)
    if best_row is not None:
        failures.extend(_best_reranker_score_failures(payload, best_row))
        row_checkpoint = str(best_row.get("checkpoint", "") or "").strip()
        if row_checkpoint != best_checkpoint:
            failures.append(
                f"best_checkpoint {best_checkpoint!r} does not match winning run checkpoint {row_checkpoint!r}"
            )
        if "selection_eligible" in best_row and not _truthy(best_row.get("selection_eligible", False)):
            failures.append(f"Winning reranker {best_reranker} was not selection eligible")
        if best_checkpoint:
            if not _truthy(best_row.get("available", False)):
                failures.append(f"Winning checkpoint reranker {best_reranker} was not available")
            row_checkpoint_sha = str(best_row.get("checkpoint_sha256", "") or "")
            checkpoint_path = Path(best_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            if not row_checkpoint_sha:
                failures.append(f"Winning checkpoint reranker {best_reranker} has no checkpoint_sha256")
            elif checkpoint_path.exists() and row_checkpoint_sha != file_sha256(checkpoint_path):
                failures.append(f"Winning checkpoint reranker {best_reranker} checkpoint_sha256 does not match file")
        elif best_reranker != "baseline":
            failures.append(f"Winning model reranker {best_reranker} has no best_checkpoint")
    if best_checkpoint:
        checkpoint_path = Path(best_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            failures.append(f"Selected checkpoint does not exist: {checkpoint_path}")
    return failures


def _failures_from_completion_compare(
    path: Path,
    expected_family_counts: dict[str, int],
    expected_corpus_fingerprint: str,
    min_completion_compare_count: int,
    required_transformer_device: str,
    required_completion_checkpoint_size: str,
) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return [f"Missing completion comparison metrics: {path}"]
    payload = _read_json(path)
    expected_total = sum(expected_family_counts.values())
    corpus_sequences = int(payload.get("num_corpus_sequences", 0))
    if corpus_sequences != expected_total:
        failures.append(
            f"Completion comparison used {corpus_sequences} corpus sequences; expected audited corpus total {expected_total}"
        )
    if expected_corpus_fingerprint and str(payload.get("corpus_fingerprint", "") or "") != expected_corpus_fingerprint:
        failures.append("Completion comparison corpus_fingerprint does not match corpus audit fingerprint")
    if required_transformer_device:
        actual_device = str(payload.get("transformer_device", "") or "")
        if actual_device != required_transformer_device:
            failures.append(
                f"Completion comparison transformer_device is {actual_device!r}; "
                f"expected {required_transformer_device!r}"
            )
    corpus_family_counts = payload.get("corpus_family_counts", {})
    if not isinstance(corpus_family_counts, dict):
        failures.append("Completion comparison metrics has no corpus_family_counts object")
    else:
        for family, expected in sorted(expected_family_counts.items()):
            count = int(corpus_family_counts.get(family, 0))
            if count != expected:
                failures.append(
                    f"Completion comparison used {count} {family} sequences; "
                    f"expected audited corpus total {expected}"
                )
    modes = payload.get("modes", [])
    if not isinstance(modes, list) or not modes:
        failures.append(f"Completion comparison metrics has no mode runs: {path}")
        return failures
    mode_names = {str(row.get("mode", "")) for row in modes if isinstance(row, dict)}
    expected_modes = {"prefix", "retrieval", "beam", "ensemble"}
    missing_modes = sorted(expected_modes - mode_names)
    if missing_modes:
        failures.append(f"Completion comparison missing modes: {', '.join(missing_modes)}")
    if min_completion_compare_count > 0:
        for row in modes:
            if not isinstance(row, dict):
                continue
            mode = str(row.get("mode", "UNKNOWN"))
            count = _as_int(row.get("count", 0))
            if count < min_completion_compare_count:
                failures.append(
                    f"Completion comparison mode {mode} count is {count}; "
                    f"expected at least {min_completion_compare_count}"
                )
    checkpoint_used = _resolve_optional_path(payload.get("checkpoint_used"))
    if checkpoint_used is not None and payload.get("transformer_available") is not True:
        failures.append("Completion comparison used a checkpoint but transformer_available is false")
    checkpoint_sha = str(payload.get("checkpoint_sha256", "") or "")
    if checkpoint_used is not None:
        if required_completion_checkpoint_size and checkpoint_used.parent.name != required_completion_checkpoint_size:
            failures.append(
                "Completion comparison checkpoint size is "
                f"{checkpoint_used.parent.name!r}; expected {required_completion_checkpoint_size!r}"
            )
        if not checkpoint_sha:
            failures.append("Completion comparison used a checkpoint but has no checkpoint_sha256")
        elif checkpoint_used.exists() and checkpoint_sha != file_sha256(checkpoint_used):
            failures.append("Completion comparison checkpoint_sha256 does not match checkpoint file")
    elif required_completion_checkpoint_size:
        failures.append(
            "Completion comparison did not use the required checkpoint size "
            f"{required_completion_checkpoint_size!r}"
        )
    return failures


def _failures_from_submissions(
    submission_dir: Path,
    reranker_metrics_path: Path,
    min_corpus_sequences: int,
    expected_corpus_fingerprint: str,
    required_transformer_device: str,
    require_selected_checkpoint: bool,
) -> list[str]:
    failures: list[str] = []
    row_counts: dict[str, int] = {}
    for filename in REQUIRED_SUBMISSIONS:
        path = submission_dir / filename
        if not path.exists():
            failures.append(f"Missing submission file: {path}")
            continue
        row_counts[filename] = _csv_count(path)
        if row_counts[filename] == 0:
            failures.append(f"Submission file has no prediction rows: {path}")
    summary_path = submission_dir / "inference_summary.json"
    if not summary_path.exists():
        failures.append(f"Missing inference summary: {summary_path}")
        return failures
    summary = _read_json(summary_path)
    num_corpus_sequences = int(summary.get("num_corpus_sequences", 0))
    if num_corpus_sequences != min_corpus_sequences:
        failures.append(
            f"Inference used {num_corpus_sequences} corpus sequences; expected audited corpus total {min_corpus_sequences}"
        )
    if expected_corpus_fingerprint and str(summary.get("corpus_fingerprint", "") or "") != expected_corpus_fingerprint:
        failures.append("Inference corpus_fingerprint does not match corpus audit fingerprint")
    if required_transformer_device:
        actual_device = str(summary.get("transformer_device", "") or "")
        if actual_device != required_transformer_device:
            failures.append(
                f"Inference transformer_device is {actual_device!r}; "
                f"expected {required_transformer_device!r}"
            )
    expected_rows = {
        "nextstep.csv": int(summary.get("nextstep_rows", -1)),
        "completion.csv": int(summary.get("completion_rows", -1)),
        "anomaly.csv": int(summary.get("anomaly_rows", -1)),
    }
    for filename, expected in expected_rows.items():
        if filename in row_counts and row_counts[filename] != expected:
            failures.append(
                f"{filename} has {row_counts[filename]} rows but inference_summary reports {expected}"
            )
    if reranker_metrics_path.exists():
        reranker_payload = _read_json(reranker_metrics_path)
        best_checkpoint = _resolve_optional_path(reranker_payload.get("best_checkpoint"))
        checkpoint_used = _resolve_optional_path(summary.get("checkpoint_used"))
        checkpoint_sha = str(summary.get("checkpoint_sha256", "") or "")
        if require_selected_checkpoint and best_checkpoint is None:
            failures.append("Reranker comparison did not select a checkpoint for inference")
        if require_selected_checkpoint and checkpoint_used is None:
            failures.append("Inference did not use a selected checkpoint")
        if require_selected_checkpoint:
            if best_checkpoint is None and checkpoint_used is not None:
                failures.append(
                    f"Reranker comparison selected baseline, but inference used checkpoint {checkpoint_used}"
                )
            if best_checkpoint is not None and checkpoint_used != best_checkpoint:
                failures.append(
                    f"Inference checkpoint {checkpoint_used} does not match selected checkpoint {best_checkpoint}"
                )
            if best_checkpoint is not None and not bool(summary.get("transformer_available", False)):
                failures.append("Inference used a selected checkpoint but transformer_available is false")
        if checkpoint_used is not None:
            if not checkpoint_sha:
                failures.append("Inference used a checkpoint but has no checkpoint_sha256")
            elif checkpoint_used.exists() and checkpoint_sha != file_sha256(checkpoint_used):
                failures.append("Inference checkpoint_sha256 does not match checkpoint file")
        if require_selected_checkpoint and best_checkpoint is not None:
            best_row = next(
                (
                    row for row in reranker_payload.get("runs", [])
                    if row.get("reranker") == reranker_payload.get("best_reranker")
                ),
                {},
            )
            best_checkpoint_sha = str(best_row.get("checkpoint_sha256", "") or "")
            if checkpoint_sha and best_checkpoint_sha and checkpoint_sha != best_checkpoint_sha:
                failures.append("Inference checkpoint_sha256 does not match selected reranker checkpoint_sha256")
        expected_mode = str(reranker_payload.get("completion_mode", "") or "")
        if expected_mode and str(summary.get("completion_mode", "")) != expected_mode:
            failures.append(
                f"Inference completion mode {summary.get('completion_mode')} does not match reranker comparison mode {expected_mode}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that a generated/trained run has required evidence artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--submission-dir", type=Path, default=PROJECT_ROOT / "submissions")
    parser.add_argument("--out", type=Path, help="Write validation summary JSON. Defaults to <artifacts-dir>/validation_summary.json.")
    parser.add_argument(
        "--readiness",
        type=Path,
        help="Leonardo readiness JSON for source-bundle checkpoint provenance. Defaults to <artifacts-dir>/leonardo_readiness.json.",
    )
    parser.add_argument("--model-sizes", nargs="*", default=["tiny", "small", "medium"])
    parser.add_argument(
        "--required-completion-checkpoint-size",
        default="",
        help="Require completion comparison to use this model-size checkpoint. Defaults to the last --model-sizes entry when transformer evidence is required.",
    )
    parser.add_argument("--min-generated-per-family", type=int, default=0)
    parser.add_argument("--max-generated-per-family", type=int, default=0)
    parser.add_argument(
        "--preflight",
        type=Path,
        help="Preflight JSON to validate. Defaults to <artifacts-dir>/preflight_full_pipeline.json when present.",
    )
    parser.add_argument("--require-preflight", action="store_true")
    parser.add_argument("--require-preflight-torch", action="store_true")
    parser.add_argument("--require-preflight-cuda", action="store_true")
    parser.add_argument("--require-preflight-eval", action="store_true")
    parser.add_argument(
        "--require-checkpoint-device",
        default="",
        help="Require every validated checkpoint train_summary.json to report this actual device.",
    )
    parser.add_argument(
        "--require-transformer-device",
        default="",
        help="Require reranker comparison, completion comparison, and inference summaries to report this transformer device.",
    )
    parser.add_argument(
        "--require-selected-checkpoint",
        action="store_true",
        help="Require reranker comparison to select an actual checkpoint instead of the baseline.",
    )
    parser.add_argument(
        "--min-train-epochs",
        type=int,
        default=0,
        help="Require every validated checkpoint train_summary.json to report at least this many epochs.",
    )
    parser.add_argument(
        "--required-batch-size",
        type=int,
        default=0,
        help="Require every validated checkpoint train_summary.json to report this batch_size. Zero disables the check.",
    )
    parser.add_argument(
        "--min-reranker-count",
        type=int,
        default=0,
        help="Require each reranker run to score at least this many next-step and completion examples.",
    )
    parser.add_argument(
        "--min-completion-compare-count",
        type=int,
        default=0,
        help="Require each completion comparison mode to score at least this many examples.",
    )
    parser.add_argument("--require-submissions", action="store_true")
    args = parser.parse_args()
    required_completion_checkpoint_size = args.required_completion_checkpoint_size
    if not required_completion_checkpoint_size and args.require_transformer_device and args.model_sizes:
        required_completion_checkpoint_size = str(args.model_sizes[-1])
    if args.min_generated_per_family < 0:
        raise SystemExit("--min-generated-per-family must be non-negative")
    if args.max_generated_per_family < 0:
        raise SystemExit("--max-generated-per-family must be non-negative")
    if (
        args.max_generated_per_family
        and args.min_generated_per_family
        and args.max_generated_per_family < args.min_generated_per_family
    ):
        raise SystemExit("--max-generated-per-family cannot be less than --min-generated-per-family")

    failures: list[str] = []
    preflight_path = args.preflight or (args.artifacts_dir / "preflight_full_pipeline.json")
    if args.require_preflight or args.require_preflight_eval or preflight_path.exists():
        failures.extend(_failures_from_preflight(
            preflight_path,
            args.require_preflight_torch,
            args.require_preflight_cuda,
            args.require_preflight_eval,
        ))
    audit_failures, audit_family_counts, audit_corpus_fingerprint = _corpus_audit_expectations(
        args.artifacts_dir / "corpus_audit" / "summary.json",
        args.min_generated_per_family,
        args.max_generated_per_family,
    )
    failures.extend(audit_failures)
    readiness_path = args.readiness or (args.artifacts_dir / "leonardo_readiness.json")
    source_bundle_failures, expected_source_bundle_sha256 = _source_bundle_expectation(readiness_path)
    failures.extend(source_bundle_failures)
    expected_family_counts = dict(audit_family_counts)
    failures.extend(_failures_from_checkpoints(
        args.checkpoint_dir,
        args.model_sizes,
        expected_family_counts,
        audit_corpus_fingerprint,
        args.require_checkpoint_device,
        args.min_train_epochs,
        args.required_batch_size,
        expected_source_bundle_sha256,
    ))
    failures.extend(_failures_from_rerankers(
        args.artifacts_dir / "reranker_compare" / "metrics.json",
        args.model_sizes,
        args.min_reranker_count,
        expected_family_counts,
        audit_corpus_fingerprint,
        args.require_transformer_device,
        args.require_selected_checkpoint,
    ))
    failures.extend(_failures_from_completion_compare(
        args.artifacts_dir / "completion_compare" / "metrics.json",
        expected_family_counts,
        audit_corpus_fingerprint,
        args.min_completion_compare_count,
        args.require_transformer_device,
        required_completion_checkpoint_size,
    ))
    if args.require_submissions:
        failures.extend(_failures_from_submissions(
            args.submission_dir,
            args.artifacts_dir / "reranker_compare" / "metrics.json",
            sum(expected_family_counts.values()),
            audit_corpus_fingerprint,
            args.require_transformer_device,
            args.require_selected_checkpoint,
        ))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "artifacts_dir": str(args.artifacts_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "submission_dir": str(args.submission_dir),
        "model_sizes": args.model_sizes,
        "min_generated_per_family": args.min_generated_per_family,
        "max_generated_per_family": args.max_generated_per_family,
        "run_profile": _run_profile(args.min_generated_per_family, args.max_generated_per_family),
        "preflight": str(preflight_path),
        "readiness": str(readiness_path),
        "require_preflight": args.require_preflight,
        "require_preflight_torch": args.require_preflight_torch,
        "require_preflight_cuda": args.require_preflight_cuda,
        "require_preflight_eval": args.require_preflight_eval,
        "required_checkpoint_device": args.require_checkpoint_device,
        "required_transformer_device": args.require_transformer_device,
        "required_completion_checkpoint_size": required_completion_checkpoint_size,
        "require_selected_checkpoint": args.require_selected_checkpoint,
        "min_train_epochs": args.min_train_epochs,
        "required_batch_size": args.required_batch_size,
        "min_reranker_count": args.min_reranker_count,
        "min_completion_compare_count": args.min_completion_compare_count,
        "require_submissions": args.require_submissions,
        "expected_family_counts": expected_family_counts,
        "corpus_fingerprint": audit_corpus_fingerprint,
        "source_bundle_sha256": expected_source_bundle_sha256,
        "failures": failures,
    }
    out_path = args.out or (args.artifacts_dir / "validation_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")

    if failures:
        print("Run validation failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)

    print("Run validation passed")
    print(f"Artifacts: {args.artifacts_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    if args.require_submissions:
        print(f"Submissions: {args.submission_dir}")


if __name__ == "__main__":
    main()
