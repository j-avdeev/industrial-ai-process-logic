from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .data import FAMILY_FILES
from .hashing import file_sha256
from .leonardo_bundle import DEFAULT_BUNDLE_MANIFEST_PATH, DEFAULT_BUNDLE_PATH, verify_bundle
from .leonardo_shell_audit import audit_scripts
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT
from .preflight import EXPECTED_EVAL_COLUMNS
from .run_profiles import COUNT_PROFILES, profile_for_count


LEONARDO_SCRIPTS = [
    "scripts/leonardo_common.sh",
    "scripts/leonardo_probe.sh",
    "scripts/leonardo_generate.sh",
    "scripts/leonardo_train.sh",
    "scripts/leonardo_train_scaling.sh",
    "scripts/leonardo_infer.sh",
    "scripts/leonardo_finalize.sh",
    "scripts/leonardo_full_pipeline.sh",
]
def _path_status(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _eval_input_status(label: str, path: Path) -> tuple[dict[str, object], list[str]]:
    status = _path_status(path)
    expected_columns = EXPECTED_EVAL_COLUMNS[label]
    status["label"] = label
    status["required_columns"] = expected_columns
    status["fieldnames"] = []
    status["missing_columns"] = expected_columns
    status["rows"] = 0
    status["sha256"] = ""
    failures: list[str] = []
    if not path.exists():
        return status, failures
    if not path.is_file():
        failures.append(f"Eval input is not a file: {path}")
        return status, failures
    if not status["bytes"]:
        failures.append(f"Eval input is empty: {path}")
        return status, failures
    try:
        status["sha256"] = file_sha256(path)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = sum(1 for _ in reader)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        failures.append(f"Eval input is not readable CSV for {label}: {path} ({exc})")
        return status, failures
    missing_columns = [column for column in expected_columns if column not in fieldnames]
    status["fieldnames"] = fieldnames
    status["missing_columns"] = missing_columns
    status["rows"] = rows
    if not fieldnames:
        failures.append(f"Eval input has no CSV header for {label}: {path}")
    elif missing_columns:
        failures.append(f"Eval input is missing columns for {label}: {', '.join(missing_columns)}")
    elif rows <= 0:
        failures.append(f"Eval input has no rows for {label}: {path}")
    return status, failures


def _rows_by_label(payload: dict[str, object], key: str) -> dict[str, dict[str, object]]:
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("label", "") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("label", "") or "")
    }


def _eval_staging_status(
    manifest_path: Path,
    eval_inputs: list[dict[str, object]],
) -> tuple[dict[str, object], list[str]]:
    status: dict[str, object] = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
        "bytes": manifest_path.stat().st_size if manifest_path.exists() and manifest_path.is_file() else 0,
        "passed": False,
        "destinations": [],
        "failures": [],
    }
    failures: list[str] = []
    if not manifest_path.exists():
        failures.append(f"Missing eval staging manifest: {manifest_path}")
        status["failures"] = failures
        return status, failures
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"Eval staging manifest is not readable JSON: {manifest_path} ({exc})")
        status["failures"] = failures
        return status, failures
    status["passed"] = payload.get("passed") is True
    status["destinations"] = payload.get("destinations", [])
    status["manifest_sha256"] = file_sha256(manifest_path)
    if payload.get("passed") is not True:
        failures.append("Eval staging manifest did not pass")
    recorded_failures = payload.get("failures", [])
    if isinstance(recorded_failures, list) and recorded_failures:
        failures.append("Eval staging manifest recorded failures: " + "; ".join(map(str, recorded_failures)))
    destination_rows = _rows_by_label(payload, "destinations")
    input_rows = {
        str(row.get("label", "") or ""): row
        for row in eval_inputs
        if isinstance(row, dict) and str(row.get("label", "") or "")
    }
    for label in ("valid", "anomaly"):
        destination = destination_rows.get(label)
        if destination is None:
            failures.append(f"Eval staging manifest missing {label} destination row")
            continue
        if destination.get("exists") is not True:
            failures.append(f"Eval staging manifest says {label} destination is missing")
        if destination.get("missing_columns"):
            failures.append(
                f"Eval staging manifest says {label} destination is missing columns: "
                + ", ".join(map(str, destination.get("missing_columns", [])))
            )
        if int(destination.get("rows", 0) or 0) <= 0:
            failures.append(f"Eval staging manifest says {label} destination has no rows")
        destination_hash = str(destination.get("sha256", "") or "")
        current_hash = str(input_rows.get(label, {}).get("sha256", "") or "")
        if not destination_hash:
            failures.append(f"Eval staging manifest has no {label} destination SHA-256")
        elif current_hash != destination_hash:
            failures.append(
                f"Eval staging manifest {label} SHA-256 does not match current eval input: "
                f"{destination_hash} != {current_hash or 'missing'}"
            )
    status["failures"] = failures
    return status, failures


def _check_required_files(paths: list[Path], failures: list[str]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        status = _path_status(path)
        rows.append(status)
        if not status["exists"]:
            failures.append(f"Missing required file: {path}")
        elif status["bytes"] == 0:
            failures.append(f"Required file is empty: {path}")
    return rows


def _source_bundle_status(
    bundle_path: Path,
    manifest_path: Path,
    require_source_bundle: bool,
) -> tuple[dict[str, object], list[str]]:
    sidecar_path = bundle_path.with_suffix(bundle_path.suffix + ".sha256")
    status: dict[str, object] = {
        "bundle_path": str(bundle_path),
        "bundle_exists": bundle_path.exists(),
        "bundle_bytes": bundle_path.stat().st_size if bundle_path.exists() and bundle_path.is_file() else None,
        "bundle_sha256": file_sha256(bundle_path) if bundle_path.exists() and bundle_path.is_file() else "",
        "sidecar_path": str(sidecar_path),
        "sidecar_exists": sidecar_path.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_source": "",
        "manifest_file_count": None,
        "verified": False,
        "failures": [],
    }
    failures: list[str] = []
    manifest: dict[str, object] = {}
    if manifest_path.exists() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            status["manifest_source"] = str(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"Source bundle manifest is not readable JSON: {manifest_path} ({exc})")
    elif bundle_path.exists() and bundle_path.is_file():
        try:
            with zipfile.ZipFile(bundle_path, "r") as zf:
                manifest = json.loads(zf.read("leonardo_source_bundle_manifest.json").decode("utf-8-sig"))
            status["manifest_source"] = f"{bundle_path}!leonardo_source_bundle_manifest.json"
        except KeyError:
            failures.append(f"Source bundle ZIP is missing embedded manifest: {bundle_path}")
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"Source bundle embedded manifest is not readable: {bundle_path} ({exc})")
    if manifest:
        status["manifest_file_count"] = manifest.get("file_count")
        manifest_files = manifest.get("files", [])
        status["manifest_files"] = manifest_files if isinstance(manifest_files, list) else []
        handoff = manifest.get("handoff", {})
        if isinstance(handoff, dict):
            status["handoff_upload_files"] = handoff.get("upload_files", [])
            status["handoff_verify_commands"] = handoff.get("verify_commands", [])
            status["handoff_readiness_commands"] = handoff.get("readiness_commands", [])
            status["handoff_deferred_eval_readiness_commands"] = handoff.get("deferred_eval_readiness_commands", [])
            status["handoff_selftest_commands"] = handoff.get("selftest_commands", [])
            status["handoff_audit_commands"] = handoff.get("handoff_audit_commands", [])
            if require_source_bundle:
                upload_files = {str(item) for item in handoff.get("upload_files", [])}
                required_uploads = {
                    bundle_path.name,
                    sidecar_path.name,
                    manifest_path.name,
                }
                missing_uploads = sorted(required_uploads - upload_files)
                if missing_uploads:
                    failures.append(
                        "Source bundle manifest handoff is missing upload files: "
                        + ", ".join(missing_uploads)
                    )
                verify_commands = "\n".join(str(item) for item in handoff.get("verify_commands", []))
                if "--verify-bundle" not in verify_commands or "--verify-root" not in verify_commands:
                    failures.append("Source bundle manifest handoff does not record bundle/root verify commands")
                readiness_commands = "\n".join(str(item) for item in handoff.get("readiness_commands", []))
                if "--require-source-bundle" not in readiness_commands:
                    failures.append("Source bundle manifest handoff does not record source-bundle readiness command")
                if "--require-eval" not in readiness_commands:
                    failures.append("Source bundle manifest handoff does not record strict eval readiness command")
                if "--defer-eval-staging" in readiness_commands:
                    failures.append("Source bundle manifest handoff strict readiness command still defers eval staging")
                deferred_readiness_commands = "\n".join(
                    str(item) for item in handoff.get("deferred_eval_readiness_commands", [])
                )
                if "--defer-eval-staging" not in deferred_readiness_commands:
                    failures.append("Source bundle manifest handoff does not record deferred-eval readiness command")
                if "--require-eval" not in deferred_readiness_commands:
                    failures.append("Source bundle manifest handoff deferred readiness command does not require eval")
                selftest_commands = "\n".join(str(item) for item in handoff.get("selftest_commands", []))
                if "source_bundle_proof_selftest" not in selftest_commands:
                    failures.append("Source bundle manifest handoff does not record source-bundle self-test command")
                audit_commands = "\n".join(str(item) for item in handoff.get("handoff_audit_commands", []))
                if "leonardo_handoff" not in audit_commands or "--require-source-bundle" not in audit_commands:
                    failures.append("Source bundle manifest handoff does not record source-bundle handoff audit command")
        elif require_source_bundle:
            failures.append("Source bundle manifest has no handoff object")
    has_bundle_evidence = bundle_path.exists() or manifest_path.exists()
    if require_source_bundle and not has_bundle_evidence:
        failures.append(
            "Missing required Leonardo source bundle evidence: "
            f"{bundle_path} or {manifest_path}"
        )
    if has_bundle_evidence:
        verify_bundle_path = bundle_path if bundle_path.exists() else None
        verify_sidecar_path = sidecar_path if bundle_path.exists() else None
        bundle_failures = verify_bundle(
            manifest_path,
            PROJECT_ROOT,
            verify_bundle_path,
            verify_sidecar_path,
        )
        failures.extend(bundle_failures)
        status["verified"] = not bundle_failures
    status["failures"] = failures
    return status, failures


def _effective_source_bundle_paths(bundle_path: Path, manifest_path: Path) -> tuple[Path, Path]:
    effective_bundle = bundle_path
    effective_manifest = manifest_path
    root_bundle = PROJECT_ROOT / DEFAULT_BUNDLE_PATH.name
    root_manifest = PROJECT_ROOT / DEFAULT_BUNDLE_MANIFEST_PATH.name
    if bundle_path == DEFAULT_BUNDLE_PATH and not bundle_path.exists() and root_bundle.exists():
        effective_bundle = root_bundle
    if manifest_path == DEFAULT_BUNDLE_MANIFEST_PATH and not manifest_path.exists() and root_manifest.exists():
        effective_manifest = root_manifest
    return effective_bundle, effective_manifest


def _script_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _script_tail_after(path: Path, needle: str) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    index = text.find(needle)
    if index < 0:
        return ""
    return text[index:]


def _sbatch_export(assignments: dict[str, object]) -> str:
    return ",".join(["ALL", *(f"{key}={value}" for key, value in assignments.items())])


def _command_value(value: object) -> str:
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            return str(value).replace("\\", "/")
    return str(value)


def _commands(args: argparse.Namespace) -> dict[str, list[str]]:
    full_common = {
        "COUNT_PER_FAMILY": args.count_per_family,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "RERANKER_VALID_PER_FAMILY": args.reranker_valid_per_family,
    }
    source_bundle_assignments: dict[str, object] = {}
    if args.require_source_bundle:
        source_bundle_assignments["REQUIRE_SOURCE_BUNDLE"] = 1
    if args.source_bundle != DEFAULT_BUNDLE_PATH:
        source_bundle_assignments["SOURCE_BUNDLE"] = args.source_bundle
    if args.source_bundle_manifest != DEFAULT_BUNDLE_MANIFEST_PATH:
        source_bundle_assignments["SOURCE_BUNDLE_MANIFEST"] = args.source_bundle_manifest
    full_common.update(source_bundle_assignments)
    finalize_common = {
        "COUNT_PER_FAMILY": args.count_per_family,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "RERANKER_VALID_PER_FAMILY": args.reranker_valid_per_family,
        **source_bundle_assignments,
    }
    if args.require_eval:
        finalize_common["REQUIRE_EVAL"] = 1
    include_eval = args.require_eval or (args.valid_input.exists() and args.anomaly_input.exists())
    eval_args = {
        "VALID_INPUT": args.valid_input,
        "ANOMALY_INPUT": args.anomaly_input,
    }
    full_assignments: dict[str, object] = dict(full_common)
    if args.require_eval:
        full_assignments["REQUIRE_EVAL"] = 1
    if include_eval:
        full_assignments.update(eval_args)
    generate_assignments: dict[str, object] = {
        "COUNT_PER_FAMILY": args.count_per_family,
        **source_bundle_assignments,
    }
    train_assignments: dict[str, object] = {
        "COUNT_PER_FAMILY": args.count_per_family,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        **source_bundle_assignments,
    }
    split_jobs = [
        "sbatch scripts/leonardo_probe.sh",
        "sbatch --export="
        + _sbatch_export({key: _command_value(value) for key, value in generate_assignments.items()})
        + " scripts/leonardo_generate.sh",
        "sbatch --export="
        + _sbatch_export({key: _command_value(value) for key, value in train_assignments.items()})
        + " scripts/leonardo_train_scaling.sh",
    ]
    if include_eval:
        split_jobs.append(
            "sbatch --export="
            + _sbatch_export({key: _command_value(value) for key, value in {**finalize_common, **eval_args}.items()})
            + " scripts/leonardo_finalize.sh"
        )
    else:
        split_jobs.append(
            "# Add official eval files, then run: sbatch --export="
            + _sbatch_export({key: _command_value(value) for key, value in {**finalize_common, **eval_args}.items()})
            + " scripts/leonardo_finalize.sh"
        )
    return {
        "full_pipeline": [
            "sbatch scripts/leonardo_probe.sh",
            "sbatch --export="
            + _sbatch_export({key: _command_value(value) for key, value in full_assignments.items()})
            + " scripts/leonardo_full_pipeline.sh",
        ],
        "split_jobs": split_jobs,
        "split_jobs_with_dependencies": [
            "PROBE_JOB=$(sbatch --parsable scripts/leonardo_probe.sh)",
            "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export="
            + _sbatch_export({key: _command_value(value) for key, value in generate_assignments.items()})
            + " scripts/leonardo_generate.sh)",
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export="
            + _sbatch_export({key: _command_value(value) for key, value in train_assignments.items()})
            + " scripts/leonardo_train_scaling.sh)",
            (
                "FINAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export="
                + _sbatch_export({key: _command_value(value) for key, value in {**finalize_common, **eval_args}.items()})
                + " scripts/leonardo_finalize.sh)"
                if include_eval
                else "# Add official eval files, then run FINAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export="
                + _sbatch_export({key: _command_value(value) for key, value in {**finalize_common, **eval_args}.items()})
                + " scripts/leonardo_finalize.sh)"
            ),
            'echo "probe=${PROBE_JOB} generate=${GEN_JOB} train=${TRAIN_JOB} finalize=${FINAL_JOB:-pending-eval}"',
        ],
    }


def _verification_commands(args: argparse.Namespace) -> list[str]:
    returned_package_command = "python -m industrial_ai.verify_returned_package"
    evidence_report_command = "python -m industrial_ai.run_evidence_report"
    if args.count_profile in COUNT_PROFILES:
        returned_package_command += f" --count-profile {args.count_profile}"
        evidence_report_command += f" --count-profile {args.count_profile}"
    else:
        returned_package_command += (
            f" --min-generated-per-family {args.count_per_family}"
            f" --max-generated-per-family {args.count_per_family}"
        )
        evidence_report_command += (
            f" --min-generated-per-family {args.count_per_family}"
            f" --max-generated-per-family {args.count_per_family}"
        )
    strict_flags = (
        " --required-manifest-stage packaged_with_submissions"
        " --require-selected-checkpoint"
        " --require-preflight-cuda"
        " --require-preflight-eval"
        " --require-generated-metadata"
    )
    returned_package_command += strict_flags + " --require-readiness"
    evidence_report_command += strict_flags + " --require-readiness"
    returned_package_command += " --require-final-leonardo-objective"
    returned_package_command += (
        " --require-source-bundle-proof"
        if args.require_source_bundle
        else " --no-require-source-bundle-proof"
    )
    returned_package_command += f" --required-batch-size {args.batch_size}"
    evidence_report_command += (
        " --require-source-bundle-proof"
        if args.require_source_bundle
        else " --no-require-source-bundle-proof"
    )
    evidence_report_command += f" --required-batch-size {args.batch_size}"
    evidence_report_command += " --prefer-package-evidence"
    readiness_command = "python -m industrial_ai.leonardo_readiness"
    if args.count_profile in COUNT_PROFILES:
        readiness_command += f" --count-profile {args.count_profile}"
    else:
        readiness_command += f" --count-per-family {args.count_per_family}"
    if args.epochs != 6:
        readiness_command += f" --epochs {args.epochs}"
    if args.batch_size != 96:
        readiness_command += f" --batch-size {args.batch_size}"
    if args.reranker_valid_per_family != 40:
        readiness_command += f" --reranker-valid-per-family {args.reranker_valid_per_family}"
    if args.require_eval:
        readiness_command += " --require-eval"
    if args.require_source_bundle:
        readiness_command += " --require-source-bundle"
    if args.source_bundle != DEFAULT_BUNDLE_PATH:
        readiness_command += f" --source-bundle {_command_value(args.source_bundle)}"
    if args.source_bundle_manifest != DEFAULT_BUNDLE_MANIFEST_PATH:
        readiness_command += f" --source-bundle-manifest {_command_value(args.source_bundle_manifest)}"
    commands = [
        "python -m industrial_ai.leonardo_shell_audit",
        readiness_command,
        "python -m industrial_ai.verify_package --package-dir artifacts/submission_package",
        returned_package_command,
        evidence_report_command,
        "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
    ]
    if args.require_source_bundle:
        commands.insert(2, "python -m industrial_ai.source_bundle_proof_selftest")
    return commands


def _resume_guidance(commands: dict[str, list[str]], verification_commands: list[str]) -> list[str]:
    full_pipeline = commands.get("full_pipeline", [])
    split_jobs = commands.get("split_jobs", [])
    verify_returned = next(
        (command for command in verification_commands if "verify_returned_package" in command),
        "python -m industrial_ai.verify_returned_package",
    )
    evidence_report = next(
        (command for command in verification_commands if "run_evidence_report" in command),
        "python -m industrial_ai.run_evidence_report",
    )
    guidance = [
        "Run scripts/leonardo_probe.sh first; only submit the full or split final run after the probe validates CUDA training and checkpoint scoring.",
    ]
    if len(full_pipeline) >= 2:
        guidance.append(
            "If the single full-pipeline job is interrupted, rerun the same command: "
            f"{full_pipeline[1]}. Generation and checkpoint steps reuse only exact complete artifacts; "
            "use generation_prepared, checkpoint_audited, comparisons_complete, and the terminal package stage "
            "in artifacts/run_manifest_events.jsonl to see how far the rerun progressed."
        )
    if len(split_jobs) >= 4:
        guidance.append(
            "For split jobs, submit generation and training first, then finalization only after "
            "artifacts/run_manifest_events.jsonl contains generation_prepared and train_scaling_complete "
            f"or all three train_<size>_complete stages. Finalization command: {split_jobs[-1]}"
        )
    dependency_split_jobs = commands.get("split_jobs_with_dependencies", [])
    if dependency_split_jobs:
        guidance.append(
            "When queueing split jobs together, prefer the split_jobs_with_dependencies commands so Slurm runs "
            "generation, training, and finalization only after the previous job exits successfully."
        )
    guidance.append(
        "After copying artifacts/leonardo_return_packet.zip plus its .sha256 and manifest, or the full "
        "artifacts/submission_package directory, back from Leonardo, verify the returned archive and write "
        "artifacts/returned_package_verification.json with: "
        f"{verify_returned}"
    )
    guidance.append(
        "The returned-package verifier runs the objective evidence summary by default; rerun it directly if you "
        "need to refresh artifacts/run_evidence_report.json with: "
        f"{evidence_report}"
    )
    return guidance


def _write_command_script(
    path: Path,
    commands: dict[str, list[str]],
    verification_commands: list[str],
    resume_guidance: list[str],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by python -m industrial_ai.leonardo_readiness.",
        "# Review the eval paths before submitting jobs.",
        "",
        "# Full pipeline",
    ]
    lines.extend(commands.get("full_pipeline", []))
    lines.extend([
        "",
        "# Dependency-safe split jobs",
    ])
    lines.extend(commands.get("split_jobs_with_dependencies", []))
    lines.extend([
        "",
        "# Post-run verification after copying artifacts/submission_package back",
    ])
    lines.extend(verification_commands)
    lines.extend([
        "",
        "# Resume guidance",
    ])
    lines.extend(f"# - {item}" for item in resume_guidance)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check repo readiness before launching the Leonardo scaling run.")
    parser.add_argument(
        "--count-profile",
        choices=sorted(COUNT_PROFILES),
        default=None,
        help="Named generation target: standard=50k/family, max=150k/family.",
    )
    parser.add_argument(
        "--count-per-family",
        type=int,
        default=None,
        help="Generated sequence target per family. Defaults to the selected count profile.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--reranker-valid-per-family", type=int, default=40)
    parser.add_argument("--valid-input", type=Path, default=PROJECT_ROOT / "data" / "eval" / "eval_input_valid.csv")
    parser.add_argument("--anomaly-input", type=Path, default=PROJECT_ROOT / "data" / "eval" / "eval_input_anomaly.csv")
    parser.add_argument(
        "--eval-staging-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "eval_staging_manifest.json",
        help="Manifest written by industrial_ai.stage_eval_inputs for official eval CSV provenance.",
    )
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument(
        "--defer-eval-staging",
        action="store_true",
        help=(
            "Allow pre-upload readiness to pass before official eval CSVs are staged. "
            "Launch commands still require eval, and job-time readiness must run without this flag."
        ),
    )
    parser.add_argument("--require-sbatch", action="store_true")
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--source-bundle-manifest", type=Path, default=DEFAULT_BUNDLE_MANIFEST_PATH)
    parser.add_argument(
        "--require-source-bundle",
        action="store_true",
        help="Fail if Leonardo source bundle evidence is missing or stale.",
    )
    parser.add_argument(
        "--commands-out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "leonardo_launch_commands.sh",
        help="Write generated launch, split-job, and post-run verification commands.",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "leonardo_readiness.json")
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    if args.count_per_family is None:
        args.count_profile = args.count_profile or "standard"
        args.count_per_family = COUNT_PROFILES[args.count_profile]
    elif args.count_profile is None:
        args.count_profile = profile_for_count(args.count_per_family)
    elif args.count_per_family != COUNT_PROFILES[args.count_profile]:
        failures.append(
            "Conflicting corpus size settings: "
            f"--count-profile {args.count_profile} expects COUNT_PER_FAMILY={COUNT_PROFILES[args.count_profile]}, "
            f"but --count-per-family is {args.count_per_family}"
        )
    if args.defer_eval_staging and not args.require_eval:
        failures.append("--defer-eval-staging is only valid together with --require-eval")

    required_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "REPORT.md",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / ".env.example",
        *(PROJECT_ROOT / script for script in LEONARDO_SCRIPTS),
    ]
    raw_data_files = [DEFAULT_DATA_DIR / filename for filename in FAMILY_FILES.values()]
    raw_data_files.append(DEFAULT_DATA_DIR / "generate_sequences.py")

    args.source_bundle, args.source_bundle_manifest = _effective_source_bundle_paths(
        args.source_bundle,
        args.source_bundle_manifest,
    )

    file_rows = _check_required_files(required_files + raw_data_files, failures)
    source_bundle, source_bundle_failures = _source_bundle_status(
        args.source_bundle,
        args.source_bundle_manifest,
        args.require_source_bundle,
    )
    if args.require_source_bundle:
        failures.extend(source_bundle_failures)
    else:
        warnings.extend(f"Source bundle evidence: {failure}" for failure in source_bundle_failures)

    slurm_dir = PROJECT_ROOT / "artifacts" / "slurm"
    if not slurm_dir.exists():
        failures.append(f"Missing Slurm output directory: {slurm_dir}")
    if not (slurm_dir / ".gitkeep").exists():
        failures.append(f"Missing tracked Slurm output placeholder: {slurm_dir / '.gitkeep'}")
    failures.extend(audit_scripts())

    for script in ("scripts/leonardo_full_pipeline.sh", "scripts/leonardo_finalize.sh"):
        path = PROJECT_ROOT / script
        if not _script_contains(path, "--selection-scope checkpoints"):
            failures.append(f"{script} does not force checkpoint-only reranker selection")
        if not _script_contains(path, "--require-selected-checkpoint"):
            failures.append(f"{script} does not fail on missing selected checkpoint")
        if not _script_contains(path, "--require-checkpoints-available"):
            failures.append(f"{script} does not require all checkpoint rerankers to load")
        if not _script_contains(path, "--require-transformer-available"):
            failures.append(f"{script} does not require completion comparison checkpoint scoring")
        if not _script_contains(path, "--checkpoint checkpoints/medium/model.pt"):
            failures.append(f"{script} does not use the medium checkpoint for completion comparison")
        if not _script_contains(path, '--set "COMPLETION_CHECKPOINT=checkpoints/medium/model.pt"'):
            failures.append(f"{script} does not record the medium completion checkpoint in run manifests")
        if not _script_contains(path, "--required-transformer-device cuda"):
            failures.append(f"{script} does not require CUDA transformer-device evidence during packaging")
        if not _script_contains(path, "industrial_ai.leonardo_shell_audit --out artifacts/leonardo_shell_audit.json"):
            failures.append(f"{script} does not write Leonardo shell audit evidence")
        if not _script_contains(path, "industrial_ai.leonardo_readiness"):
            failures.append(f"{script} does not write Leonardo readiness evidence")
        if not _script_contains(path, "industrial_ai.source_bundle_proof_selftest --out artifacts/source_bundle_proof_selftest.json"):
            failures.append(f"{script} does not write source-bundle proof self-test evidence")
        if not _script_contains(path, "REQUIRE_SOURCE_BUNDLE"):
            failures.append(f"{script} does not support source-bundle readiness proof")
        if not _script_contains(path, "--require-source-bundle"):
            failures.append(f"{script} does not pass source-bundle proof into readiness when requested")
        if not _script_contains(path, "SOURCE_BUNDLE_PROOF_ARGS=(--no-require-source-bundle-proof)"):
            failures.append(f"{script} does not default final source-bundle proof checks to disabled")
        if not _script_contains(path, "SOURCE_BUNDLE_PROOF_ARGS=(--require-source-bundle-proof)"):
            failures.append(f"{script} does not enable final source-bundle proof checks when requested")
        if not _script_contains(path, '"${SOURCE_BUNDLE_PROOF_ARGS[@]}"'):
            failures.append(f"{script} does not pass conditional source-bundle proof flags into final checks")
        if not _script_contains(path, "industrial_ai.checkpoint_audit"):
            failures.append(f"{script} does not run checkpoint audit before final comparisons")
        if not _script_contains(path, "--stage checkpoint_audited"):
            failures.append(f"{script} does not record checkpoint_audited manifest stage")
        if not _script_contains(path, "--stage comparisons_complete"):
            failures.append(f"{script} does not record comparisons_complete manifest stage")
        final_audit_tail = _script_tail_after(path, "python -m industrial_ai.final_audit")
        if not final_audit_tail:
            failures.append(f"{script} does not run final_audit")
        else:
            final_audit_needles = {
                '--min-generated-per-family "${COUNT_PER_FAMILY}"': "COUNT_PER_FAMILY",
                '--max-generated-per-family "${COUNT_PER_FAMILY}"': "generated corpus upper bound",
                '--min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))"': "reranker example count",
                '--min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))"': (
                    "completion comparison count"
                ),
                '--min-train-epochs "${EPOCHS}"': "EPOCHS",
                '--required-batch-size "${BATCH_SIZE}"': "BATCH_SIZE",
                "--required-transformer-device cuda": "CUDA transformer device",
                "--require-selected-checkpoint": "selected-checkpoint requirement",
                "--require-preflight-eval": "eval preflight requirement",
                "--require-generated-metadata": "generated metadata requirement",
                "--require-readiness": "readiness evidence requirement",
                '"${SOURCE_BUNDLE_PROOF_ARGS[@]}"': "conditional source-bundle proof",
            }
            for final_audit_needle, label in final_audit_needles.items():
                if final_audit_needle not in final_audit_tail:
                    failures.append(f"{script} does not pass {label} into final_audit")
        evidence_report_tail = _script_tail_after(path, "python -m industrial_ai.run_evidence_report")
        if not evidence_report_tail:
            failures.append(f"{script} does not run run_evidence_report after final_audit")
        else:
            evidence_report_needles = {
                '--min-generated-per-family "${COUNT_PER_FAMILY}"': "COUNT_PER_FAMILY",
                '--max-generated-per-family "${COUNT_PER_FAMILY}"': "generated corpus upper bound",
                '--min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))"': "reranker example count",
                '--min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))"': (
                    "completion comparison count"
                ),
                '--min-train-epochs "${EPOCHS}"': "EPOCHS",
                '--required-batch-size "${BATCH_SIZE}"': "BATCH_SIZE",
                "--required-checkpoint-sizes tiny small medium": "required checkpoint sizes",
                "--required-transformer-device cuda": "CUDA transformer device",
                "--required-manifest-stage packaged_with_submissions": "terminal manifest stage",
                "--require-readiness": "readiness evidence",
                '"${SOURCE_BUNDLE_PROOF_ARGS[@]}"': "conditional source-bundle proof",
                "--prefer-package-evidence": "returned package evidence precedence",
            }
            for evidence_report_needle, label in evidence_report_needles.items():
                if evidence_report_needle not in evidence_report_tail:
                    failures.append(f"{script} does not pass {label} into run_evidence_report")
            if "python -m industrial_ai.verify_returned_package" not in evidence_report_tail:
                failures.append(f"{script} does not run verify_returned_package after run_evidence_report")
        returned_package_tail = _script_tail_after(path, "python -m industrial_ai.verify_returned_package")
        if not returned_package_tail:
            failures.append(f"{script} does not write returned-package verification summary")
        else:
            returned_package_needles = {
                '--min-generated-per-family "${COUNT_PER_FAMILY}"': "COUNT_PER_FAMILY",
                '--max-generated-per-family "${COUNT_PER_FAMILY}"': "generated corpus upper bound",
                '--min-reranker-count "$((RERANKER_VALID_PER_FAMILY * 6))"': "reranker example count",
                '--min-completion-compare-count "$((RERANKER_VALID_PER_FAMILY * 6))"': (
                    "completion comparison count"
                ),
                '--min-train-epochs "${EPOCHS}"': "EPOCHS",
                '--required-batch-size "${BATCH_SIZE}"': "BATCH_SIZE",
                "--required-checkpoint-sizes tiny small medium": "required checkpoint sizes",
                "--required-transformer-device cuda": "CUDA transformer device",
                "--required-manifest-stage packaged_with_submissions": "terminal manifest stage",
                "--require-selected-checkpoint": "selected-checkpoint requirement",
                "--require-preflight-cuda": "CUDA preflight requirement",
                "--require-preflight-eval": "eval preflight requirement",
                "--require-generated-metadata": "generated metadata requirement",
                "--require-readiness": "readiness evidence",
                '"${SOURCE_BUNDLE_PROOF_ARGS[@]}"': "conditional source-bundle proof",
            }
            for returned_package_needle, label in returned_package_needles.items():
                if returned_package_needle not in returned_package_tail:
                    failures.append(f"{script} does not pass {label} into verify_returned_package")
            if "python -m industrial_ai.leonardo_return_packet" not in returned_package_tail:
                failures.append(f"{script} does not create Leonardo return packet after returned-package verification")
            if (
                "python -m industrial_ai.leonardo_return_packet --require-final-leonardo-objective"
                not in returned_package_tail
            ):
                failures.append(f"{script} does not require final Leonardo objective when creating return packet")

    for script in ("scripts/leonardo_train.sh", "scripts/leonardo_train_scaling.sh"):
        path = PROJECT_ROOT / script
        if not _script_contains(path, 'source "${SCRIPT_DIR}/leonardo_common.sh"'):
            failures.append(f"{script} does not source Leonardo launch guards")
        if not _script_contains(path, 'require_min_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 50000'):
            failures.append(f"{script} does not reject COUNT_PER_FAMILY below 50000")
        if not _script_contains(path, 'require_max_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 150000'):
            failures.append(f"{script} does not reject COUNT_PER_FAMILY above 150000")
        if not _script_contains(path, 'require_min_int EPOCHS "${EPOCHS}" 6'):
            failures.append(f"{script} does not reject EPOCHS below 6")
        if not _script_contains(path, 'require_positive_int BATCH_SIZE "${BATCH_SIZE}"'):
            failures.append(f"{script} does not validate BATCH_SIZE")
        if not _script_contains(path, 'COUNT_PER_FAMILY="${COUNT_PER_FAMILY:-50000}"'):
            failures.append(f"{script} does not default COUNT_PER_FAMILY to the final-run lower bound")
        if not _script_contains(path, "industrial_ai.leonardo_readiness"):
            failures.append(f"{script} does not verify Leonardo readiness before training")
        if not _script_contains(path, "REQUIRE_SOURCE_BUNDLE"):
            failures.append(f"{script} does not support source-bundle readiness proof before training")
        if not _script_contains(path, "--require-source-bundle"):
            failures.append(f"{script} does not pass source-bundle proof into training readiness when requested")
        if not _script_contains(path, "--require-source-bundle-proof"):
            failures.append(f"{script} does not bind source-bundle proof into checkpoint training when requested")
        if not _script_contains(path, '--set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"'):
            failures.append(f"{script} does not record source-bundle proof mode in run manifests")
        if not _script_contains(path, "industrial_ai.audit_corpus"):
            failures.append(f"{script} does not audit generated corpus size before training")
        if not _script_contains(path, '--min-generated-per-family "${COUNT_PER_FAMILY}"'):
            failures.append(f"{script} does not require generated corpus lower bound before training")
        if not _script_contains(path, '--max-generated-per-family "${COUNT_PER_FAMILY}"'):
            failures.append(f"{script} does not reject stale over-sized generated corpus before training")
        if script == "scripts/leonardo_train_scaling.sh" and not _script_contains(
            path,
            "industrial_ai.checkpoint_audit",
        ):
            failures.append(f"{script} does not audit checkpoints after scaling training")
        if script == "scripts/leonardo_train_scaling.sh" and not _script_contains(
            path,
            "--stage train_scaling_checkpoint_audited",
        ):
            failures.append(f"{script} does not record train_scaling_checkpoint_audited stage")

    probe_script = PROJECT_ROOT / "scripts/leonardo_probe.sh"
    probe_needles = {
        'source "${SCRIPT_DIR}/leonardo_common.sh"': "Leonardo launch guards",
        "#SBATCH --gres=gpu:1": "GPU request",
        "industrial_ai.leonardo_shell_audit --out artifacts/probe/leonardo_shell_audit.json": (
            "Leonardo shell audit evidence"
        ),
        "--require-cuda": "CUDA preflight",
        "--device cuda": "CUDA training",
        "--require-device": "strict training device",
        "--require-transformer-available": "strict checkpoint scorer availability",
        "--selection-scope checkpoints": "checkpoint-only reranker selection",
        "--require-selected-checkpoint": "selected checkpoint gate",
        "--require-checkpoints-available": "checkpoint load gate",
        "--require-checkpoint-device cuda": "CUDA checkpoint validation",
        "--require-transformer-device cuda": "CUDA scorer validation",
        "--stage probe_validated": "validated terminal manifest stage",
    }
    for needle, label in probe_needles.items():
        if not _script_contains(probe_script, needle):
            failures.append(f"scripts/leonardo_probe.sh does not include {label}")

    for script in ("scripts/leonardo_generate.sh", "scripts/leonardo_full_pipeline.sh", "scripts/leonardo_finalize.sh"):
        path = PROJECT_ROOT / script
        if not _script_contains(path, 'source "${SCRIPT_DIR}/leonardo_common.sh"'):
            failures.append(f"{script} does not source Leonardo launch guards")
        if not _script_contains(path, 'require_min_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 50000'):
            failures.append(f"{script} does not reject COUNT_PER_FAMILY below 50000")
        if not _script_contains(path, 'require_max_int COUNT_PER_FAMILY "${COUNT_PER_FAMILY}" 150000'):
            failures.append(f"{script} does not reject COUNT_PER_FAMILY above 150000")
        if not _script_contains(path, '--max-generated-per-family "${COUNT_PER_FAMILY}"'):
            failures.append(f"{script} does not reject generated corpora above COUNT_PER_FAMILY")
        if not _script_contains(path, "--stage generation_prepared"):
            failures.append(f"{script} does not record generation_prepared manifest stage")
        if script == "scripts/leonardo_generate.sh":
            if not _script_contains(path, "industrial_ai.leonardo_readiness"):
                failures.append(f"{script} does not verify Leonardo readiness before generation")
            if not _script_contains(path, "REQUIRE_SOURCE_BUNDLE"):
                failures.append(f"{script} does not support source-bundle readiness proof before generation")
            if not _script_contains(path, "--require-source-bundle"):
                failures.append(f"{script} does not pass source-bundle proof into generation readiness when requested")
            if not _script_contains(path, '--set "REQUIRE_SOURCE_BUNDLE=${REQUIRE_SOURCE_BUNDLE}"'):
                failures.append(f"{script} does not record source-bundle proof mode in run manifests")
        if script in {"scripts/leonardo_generate.sh", "scripts/leonardo_full_pipeline.sh"}:
            if not _script_contains(path, "--exact-count"):
                failures.append(f"{script} does not require exact generated counts during skip-if-complete reuse")
            post_audit_tail = _script_tail_after(path, '--max-generated-per-family "${COUNT_PER_FAMILY}"')
            if "industrial_ai.prepare" not in post_audit_tail:
                failures.append(f"{script} does not prepare generated-aware corpus stats after generation audit")
        if script in {"scripts/leonardo_full_pipeline.sh", "scripts/leonardo_finalize.sh"}:
            if not _script_contains(path, 'require_min_int EPOCHS "${EPOCHS}" 6'):
                failures.append(f"{script} does not reject EPOCHS below 6")
            if not _script_contains(path, 'require_min_int RERANKER_VALID_PER_FAMILY "${RERANKER_VALID_PER_FAMILY}" 40'):
                failures.append(f"{script} does not reject too-small reranker validation sets")
            if script == "scripts/leonardo_full_pipeline.sh" and not _script_contains(
                path,
                'require_positive_int BATCH_SIZE "${BATCH_SIZE}"',
            ):
                failures.append(f"{script} does not validate BATCH_SIZE")
            if not _script_contains(path, '--required-max-generated-per-family "${COUNT_PER_FAMILY}"'):
                failures.append(f"{script} does not require packaged max generated threshold evidence")
            if not _script_contains(path, "--require-readiness"):
                failures.append(f"{script} does not require packaged readiness evidence")
            if script == "scripts/leonardo_full_pipeline.sh" and not _script_contains(
                path,
                "--require-source-bundle-proof",
            ):
                failures.append(f"{script} does not bind source-bundle proof into checkpoint training when requested")

    infer_script = PROJECT_ROOT / "scripts/leonardo_infer.sh"
    if not _script_contains(infer_script, 'source "${SCRIPT_DIR}/leonardo_common.sh"'):
        failures.append("scripts/leonardo_infer.sh does not source Leonardo launch guards")
    if not _script_contains(infer_script, "#SBATCH --gres=gpu:1"):
        failures.append("scripts/leonardo_infer.sh does not request a GPU for CUDA checkpoint inference")
    if not _script_contains(infer_script, 'TRANSFORMER_DEVICE="${TRANSFORMER_DEVICE:-cuda}"'):
        failures.append("scripts/leonardo_infer.sh does not default transformer inference to CUDA")
    if not _script_contains(infer_script, 'VALID_INPUT="${VALID_INPUT:-data/eval/eval_input_valid.csv}"'):
        failures.append("scripts/leonardo_infer.sh does not default valid input to the official eval drop path")
    if not _script_contains(infer_script, 'ANOMALY_INPUT="${ANOMALY_INPUT:-data/eval/eval_input_anomaly.csv}"'):
        failures.append("scripts/leonardo_infer.sh does not default anomaly input to the official eval drop path")
    if not _script_contains(infer_script, 'require_choice TRANSFORMER_DEVICE "${TRANSFORMER_DEVICE}" cuda'):
        failures.append("scripts/leonardo_infer.sh does not reject non-CUDA transformer inference")
    if not _script_contains(infer_script, "--require-cuda"):
        failures.append("scripts/leonardo_infer.sh does not require CUDA preflight")
    if not _script_contains(infer_script, "--require-transformer-available"):
        failures.append("scripts/leonardo_infer.sh does not fail when checkpoint scorer is unavailable")
    if not _script_contains(infer_script, "--require-selected-checkpoint"):
        failures.append("scripts/leonardo_infer.sh does not require inference to match selected checkpoint")

    for script in ("scripts/leonardo_full_pipeline.sh", "scripts/leonardo_finalize.sh"):
        path = PROJECT_ROOT / script
        infer_tail = _script_tail_after(path, "python -m industrial_ai.infer")
        if not infer_tail:
            failures.append(f"{script} does not run inference")
            continue
        for needle, label in {
            "--require-checkpoint": "checkpoint existence",
            "--require-transformer-available": "checkpoint scorer availability",
            "--require-selected-checkpoint": "selected checkpoint match",
        }.items():
            if needle not in infer_tail:
                failures.append(f"{script} does not require {label} during inference")

    if args.count_per_family < 50000:
        failures.append(f"COUNT_PER_FAMILY is {args.count_per_family}; expected at least 50000 for final Leonardo run")
    if args.count_per_family > 150000:
        failures.append(f"COUNT_PER_FAMILY is {args.count_per_family}; expected at most 150000 for final Leonardo run")
    if args.epochs < 6:
        failures.append(f"EPOCHS is {args.epochs}; expected at least 6 for final Leonardo run")
    if args.batch_size <= 0:
        failures.append(f"BATCH_SIZE is {args.batch_size}; expected a positive integer")
    reranker_examples = args.reranker_valid_per_family * 6
    if reranker_examples < 240:
        failures.append(
            f"Reranker comparison examples would be {reranker_examples}; expected at least 240"
        )

    eval_inputs = []
    for label, path in (("valid", args.valid_input), ("anomaly", args.anomaly_input)):
        status, eval_failures = _eval_input_status(label, path)
        eval_inputs.append(status)
        if args.require_eval and not path.exists() and args.defer_eval_staging:
            warnings.append(
                f"Deferred required {label} eval input staging: {path}. "
                "Stage the official CSV before running Leonardo jobs; job-time readiness will fail until it exists."
            )
        elif args.require_eval and not path.exists():
            failures.append(f"Missing required {label} eval input: {path}")
        elif not path.exists():
            warnings.append(f"Eval input not present yet: {path}")
        elif eval_failures:
            failures.extend(eval_failures)
    eval_staging: dict[str, object] = {
        "path": str(args.eval_staging_manifest),
        "exists": args.eval_staging_manifest.exists(),
        "bytes": (
            args.eval_staging_manifest.stat().st_size
            if args.eval_staging_manifest.exists() and args.eval_staging_manifest.is_file()
            else 0
        ),
        "passed": False,
        "destinations": [],
        "failures": [],
    }
    if args.require_eval and not args.defer_eval_staging and all(Path(str(row["path"])).exists() for row in eval_inputs):
        eval_staging, eval_staging_failures = _eval_staging_status(args.eval_staging_manifest, eval_inputs)
        failures.extend(eval_staging_failures)
    elif args.require_eval and args.defer_eval_staging:
        warnings.append(
            f"Deferred eval staging manifest check: {args.eval_staging_manifest}. "
            "Run stage_eval_inputs and rerun readiness without --defer-eval-staging on Leonardo."
        )

    sbatch_path = shutil.which("sbatch")
    if args.require_sbatch and not sbatch_path:
        failures.append("sbatch is not on PATH")
    elif not sbatch_path:
        warnings.append("sbatch is not on PATH; expected on Leonardo login nodes, not local Windows")

    commands = _commands(args)
    if args.require_source_bundle:
        command_text = "\n".join(
            str(command)
            for command_list in commands.values()
            if isinstance(command_list, list)
            for command in command_list
        )
        for needle, label in {
            "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh": "split generation source-bundle export",
            "REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh": (
                "split training source-bundle export"
            ),
            "REQUIRE_SOURCE_BUNDLE=1,REQUIRE_EVAL=1": "strict final source-bundle and eval exports",
        }.items():
            if needle not in command_text:
                failures.append(f"Generated launch commands missing {label}: {needle}")
    verification_commands = _verification_commands(args)
    resume_guidance = _resume_guidance(commands, verification_commands)
    command_script_path = str(args.commands_out)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "project_root": str(PROJECT_ROOT),
        "count_profile": args.count_profile,
        "count_per_family": args.count_per_family,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "reranker_valid_per_family": args.reranker_valid_per_family,
        "reranker_examples": reranker_examples,
        "require_eval": args.require_eval,
        "defer_eval_staging": args.defer_eval_staging,
        "require_sbatch": args.require_sbatch,
        "require_source_bundle": args.require_source_bundle,
        "sbatch": sbatch_path or "",
        "source_bundle": source_bundle,
        "files": file_rows,
        "eval_inputs": eval_inputs,
        "eval_staging": eval_staging,
        "commands": commands,
        "commands_out": command_script_path,
        "verification_commands": verification_commands,
        "resume_guidance": resume_guidance,
        "warnings": warnings,
        "failures": failures,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_command_script(args.commands_out, commands, verification_commands, resume_guidance)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.commands_out}")
    if warnings:
        print("Readiness warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if failures:
        print("Readiness failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    for command in commands["full_pipeline"]:
        print(command)
    print("Dependency-safe split-job commands:")
    for command in commands.get("split_jobs_with_dependencies", []):
        print(command)
    print("Post-run verification commands:")
    for command in verification_commands:
        print(command)
    print("Resume guidance:")
    for item in resume_guidance:
        print(f"- {item}")
    print("Leonardo readiness passed")


if __name__ == "__main__":
    main()
