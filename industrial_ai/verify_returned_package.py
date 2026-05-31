from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .hashing import file_sha256
from .paths import PROJECT_ROOT
from .run_profiles import COUNT_PROFILES, profile_for_count


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_json_or_empty(path: Path) -> dict[str, object]:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _optional_sha256(path: Path) -> str:
    return file_sha256(path) if path.exists() else ""


def _expected_run_profile(min_generated_per_family: int, max_generated_per_family: int) -> str:
    if max_generated_per_family and max_generated_per_family != min_generated_per_family:
        return ""
    count = max_generated_per_family or min_generated_per_family
    return profile_for_count(count) if count > 0 else ""


def _generated_count_args_failure(min_generated_per_family: int, max_generated_per_family: int) -> str:
    if not max_generated_per_family or not min_generated_per_family:
        return ""
    if max_generated_per_family < min_generated_per_family:
        return "--max-generated-per-family cannot be less than --min-generated-per-family"
    if max_generated_per_family != min_generated_per_family:
        return (
            "returned-package verification requires an exact generated-count target; "
            "use --count-profile standard/max or set --min-generated-per-family and "
            "--max-generated-per-family to the same COUNT_PER_FAMILY"
        )
    return ""


def _package_manifest_status(package_dir: Path) -> tuple[dict[str, object], list[str]]:
    manifest_path = package_dir / "package_manifest.json"
    zip_path = package_dir / "track1_submission.zip"
    status: dict[str, object] = {
        "path": str(manifest_path),
        "exists": manifest_path.exists(),
        "source": "",
        "sha256": "",
    }
    if manifest_path.exists():
        status["source"] = str(manifest_path)
        status["sha256"] = file_sha256(manifest_path)
        return status, []
    if not zip_path.exists():
        return status, [f"Package manifest is missing: {manifest_path}; package ZIP also missing: {zip_path}"]
    try:
        with zipfile.ZipFile(zip_path) as zf:
            raw = zf.read("package_manifest.json")
    except KeyError:
        return status, [f"Package manifest is missing: {manifest_path} and ZIP entry package_manifest.json"]
    except (OSError, zipfile.BadZipFile) as exc:
        return status, [f"Package ZIP is not readable while checking package_manifest.json: {zip_path} ({exc})"]
    status["source"] = f"{zip_path}!package_manifest.json"
    status["sha256"] = hashlib.sha256(raw).hexdigest()
    return status, []


def _sidecar_status(zip_path: Path, sidecar_path: Path) -> tuple[dict[str, object], list[str]]:
    status: dict[str, object] = {
        "sidecar": str(sidecar_path),
        "sidecar_exists": sidecar_path.exists(),
        "zip": str(zip_path),
        "zip_exists": zip_path.exists(),
        "recorded_zip_sha256": "",
        "recorded_zip_name": "",
        "actual_zip_sha256": _optional_sha256(zip_path),
        "matches_zip": False,
    }
    failures: list[str] = []
    if not zip_path.exists():
        failures.append(f"Package ZIP is missing: {zip_path}")
    if not sidecar_path.exists():
        failures.append(f"Package ZIP checksum sidecar is missing: {sidecar_path}")
        return status, failures
    try:
        parts = sidecar_path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        failures.append(f"Package ZIP checksum sidecar is not readable: {sidecar_path} ({exc})")
        return status, failures
    if len(parts) < 2:
        failures.append(f"Package ZIP checksum sidecar is malformed: {sidecar_path}")
        return status, failures
    recorded_hash, recorded_name = parts[0], parts[1]
    status["recorded_zip_sha256"] = recorded_hash
    status["recorded_zip_name"] = recorded_name
    if recorded_name != zip_path.name:
        failures.append(f"Package ZIP checksum sidecar names {recorded_name!r}; expected {zip_path.name!r}")
    actual_hash = str(status["actual_zip_sha256"])
    if not actual_hash or recorded_hash != actual_hash:
        failures.append(
            f"Package ZIP checksum sidecar hash mismatch: recorded {recorded_hash or 'missing'}, "
            f"actual {actual_hash or 'missing'}"
        )
    status["matches_zip"] = not failures
    return status, failures


def _report_failures(
    report_path: Path,
    require_report: bool,
    require_final_leonardo_objective: bool,
) -> list[str]:
    if not require_report:
        return []
    if not report_path.exists():
        return [f"Run evidence report was not written: {report_path}"]
    try:
        report = _read_json(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Run evidence report is not readable JSON: {report_path} ({exc})"]
    if report.get("objective_ready") is not True:
        return [f"Run evidence report objective_ready is not true: {report.get('objective_ready')!r}"]
    if require_final_leonardo_objective and report.get("final_leonardo_objective_ready") is not True:
        return [
            "Run evidence report final_leonardo_objective_ready is not true: "
            f"{report.get('final_leonardo_objective_ready')!r}"
        ]
    return []


def _final_audit_failures(final_audit_path: Path) -> list[str]:
    if not final_audit_path.exists():
        return [f"Final audit summary was not written: {final_audit_path}"]
    try:
        final_audit = _read_json(final_audit_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Final audit summary is not readable JSON: {final_audit_path} ({exc})"]
    if final_audit.get("passed") is not True:
        return [f"Final audit summary passed is not true: {final_audit.get('passed')!r}"]
    return []


def _expected_value_failures(
    actual: dict[str, object],
    expected: dict[str, object],
    label: str,
    nested_key: str | None = None,
) -> list[str]:
    failures: list[str] = []
    source = actual.get(nested_key, {}) if nested_key else actual
    if not isinstance(source, dict):
        return [f"{label} does not record expected values"]
    keys = [
        "min_generated_per_family",
        "max_generated_per_family",
        "min_reranker_count",
        "min_completion_compare_count",
        "min_train_epochs",
        "required_batch_size",
        "required_checkpoint_sizes",
        "required_manifest_stage",
        "required_transformer_device",
        "require_selected_checkpoint",
        "require_preflight_cuda",
        "require_preflight_eval",
        "require_generated_metadata",
        "require_readiness",
        "require_source_bundle_proof",
    ]
    if nested_key:
        keys.extend(["run_profile", "prefer_package_evidence"])
    else:
        keys.extend(["count_profile", "run_profile"])
    for key in keys:
        if key not in expected:
            continue
        actual_value = source.get(key)
        expected_value = expected[key]
        if actual_value != expected_value:
            failures.append(
                f"{label} expected {key} is {actual_value!r}; wrapper expected {expected_value!r}"
            )
    return failures


def _write_verification_summary(
    path: Path,
    artifacts_dir: Path,
    package_dir: Path,
    final_audit_path: Path,
    report_path: Path,
    require_report: bool,
    require_final_leonardo_objective: bool,
    expected: dict[str, object],
    raise_on_failure: bool = True,
) -> dict[str, object]:
    final_failures = _final_audit_failures(final_audit_path)
    report_failures = _report_failures(report_path, require_report, require_final_leonardo_objective)
    sidecar_status, package_failures = _sidecar_status(
        package_dir / "track1_submission.zip",
        package_dir / "track1_submission.zip.sha256",
    )
    manifest_status, manifest_failures = _package_manifest_status(package_dir)
    package_failures.extend(manifest_failures)
    report_payload = _read_json_or_empty(report_path) if require_report else {}
    final_payload = _read_json_or_empty(final_audit_path)
    expected_failures = []
    if not final_failures:
        expected_failures.extend(_expected_value_failures(final_payload, expected, "Final audit summary"))
    if require_report and not report_failures:
        expected_failures.extend(_expected_value_failures(
            report_payload,
            expected,
            "Run evidence report",
            nested_key="expected",
        ))
    failures = [*package_failures, *final_failures, *report_failures, *expected_failures]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "artifacts_dir": str(artifacts_dir),
        "package_dir": str(package_dir),
        "package_zip": str(package_dir / "track1_submission.zip"),
        "package_zip_sha256": _optional_sha256(package_dir / "track1_submission.zip"),
        "package_sidecar_sha256": _optional_sha256(package_dir / "track1_submission.zip.sha256"),
        "package_sidecar_status": sidecar_status,
        "package_manifest_sha256": str(manifest_status.get("sha256", "") or ""),
        "package_manifest_status": manifest_status,
        "final_audit_summary": str(final_audit_path),
        "final_audit_summary_sha256": _optional_sha256(final_audit_path),
        "final_audit_passed": final_payload.get("passed"),
        "run_evidence_report": str(report_path),
        "run_evidence_report_sha256": _optional_sha256(report_path),
        "objective_ready": report_payload.get("objective_ready") if require_report else None,
        "objective_scope": report_payload.get("objective_scope") if require_report else None,
        "final_leonardo_objective_ready": (
            report_payload.get("final_leonardo_objective_ready") if require_report else None
        ),
        "require_evidence_report": require_report,
        "require_final_leonardo_objective": require_final_leonardo_objective,
        "expected": expected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if failures and raise_on_failure:
        print("Returned package verification summary failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(2)
    return payload


def _run(args: list[str]) -> None:
    print("+", " ".join(args))
    result = subprocess.run(args, cwd=PROJECT_ROOT)
    if result.returncode:
        print(f"Command failed with exit code {result.returncode}")
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a returned Leonardo submission package end to end.")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--package-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "submission_package")
    parser.add_argument("--required-manifest-stage", default="packaged_with_submissions")
    parser.add_argument("--required-checkpoint-sizes", nargs="*", default=["tiny", "small", "medium"])
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
    parser.add_argument("--required-batch-size", type=int, default=0)
    parser.add_argument("--required-transformer-device", default="cuda")
    parser.add_argument("--require-selected-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-preflight-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-preflight-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-generated-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-readiness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-source-bundle-proof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-evidence-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run industrial_ai.run_evidence_report after final_audit.",
    )
    parser.add_argument(
        "--require-final-leonardo-objective",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail unless the evidence report proves the strict final Leonardo objective scope.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write a returned-package verification summary JSON.")
    args = parser.parse_args()
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
    count_args_failure = _generated_count_args_failure(
        args.min_generated_per_family,
        args.max_generated_per_family,
    )
    if count_args_failure:
        parser.error(count_args_failure)
    if args.require_source_bundle_proof and not args.require_readiness:
        parser.error("--require-source-bundle-proof requires --require-readiness")
    if args.require_final_leonardo_objective and not args.require_evidence_report:
        parser.error("--require-final-leonardo-objective requires --require-evidence-report")
    if args.out is None:
        args.out = args.artifacts_dir / "returned_package_verification.json"

    _run([
        sys.executable,
        "-m",
        "industrial_ai.verify_package",
        "--package-dir",
        str(args.package_dir),
    ])
    final_audit_args = [
        sys.executable,
        "-m",
        "industrial_ai.final_audit",
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--package-dir",
        str(args.package_dir),
        "--required-manifest-stage",
        args.required_manifest_stage,
        "--required-checkpoint-sizes",
        *args.required_checkpoint_sizes,
        "--min-generated-per-family",
        str(args.min_generated_per_family),
        "--max-generated-per-family",
        str(args.max_generated_per_family),
        "--min-reranker-count",
        str(args.min_reranker_count),
        "--min-completion-compare-count",
        str(args.min_completion_compare_count),
        "--min-train-epochs",
        str(args.min_train_epochs),
        "--required-batch-size",
        str(args.required_batch_size),
        f"--required-transformer-device={args.required_transformer_device}",
    ]
    if args.count_profile is not None:
        final_audit_args.extend(["--count-profile", args.count_profile])
    final_audit_args.append("--require-selected-checkpoint" if args.require_selected_checkpoint else "--no-require-selected-checkpoint")
    final_audit_args.append("--require-preflight-cuda" if args.require_preflight_cuda else "--no-require-preflight-cuda")
    final_audit_args.append("--require-preflight-eval" if args.require_preflight_eval else "--no-require-preflight-eval")
    final_audit_args.append("--require-generated-metadata" if args.require_generated_metadata else "--no-require-generated-metadata")
    final_audit_args.append("--require-readiness" if args.require_readiness else "--no-require-readiness")
    final_audit_args.append(
        "--require-source-bundle-proof"
        if args.require_source_bundle_proof
        else "--no-require-source-bundle-proof"
    )
    _run(final_audit_args)
    if args.require_evidence_report:
        report_args = [
            sys.executable,
            "-m",
            "industrial_ai.run_evidence_report",
            "--artifacts-dir",
            str(args.artifacts_dir),
            "--package-dir",
            str(args.package_dir),
            "--required-manifest-stage",
            args.required_manifest_stage,
            "--required-checkpoint-sizes",
            *args.required_checkpoint_sizes,
            "--min-generated-per-family",
            str(args.min_generated_per_family),
            "--max-generated-per-family",
            str(args.max_generated_per_family),
            "--min-reranker-count",
            str(args.min_reranker_count),
            "--min-completion-compare-count",
            str(args.min_completion_compare_count),
            "--min-train-epochs",
            str(args.min_train_epochs),
            "--required-batch-size",
            str(args.required_batch_size),
            f"--required-transformer-device={args.required_transformer_device}",
            "--require-selected-checkpoint"
            if args.require_selected_checkpoint
            else "--no-require-selected-checkpoint",
            "--require-preflight-cuda" if args.require_preflight_cuda else "--no-require-preflight-cuda",
            "--require-preflight-eval" if args.require_preflight_eval else "--no-require-preflight-eval",
            "--require-generated-metadata"
            if args.require_generated_metadata
            else "--no-require-generated-metadata",
            "--require-readiness" if args.require_readiness else "--no-require-readiness",
            "--require-source-bundle-proof"
            if args.require_source_bundle_proof
            else "--no-require-source-bundle-proof",
            "--prefer-package-evidence",
            "--out",
            str(args.artifacts_dir / "run_evidence_report.json"),
            "--markdown-out",
            str(args.artifacts_dir / "run_evidence_report.md"),
        ]
        if args.count_profile is not None:
            report_args.extend(["--count-profile", args.count_profile])
        _run(report_args)
    summary = _write_verification_summary(
        args.out,
        args.artifacts_dir,
        args.package_dir,
        args.artifacts_dir / "final_audit_summary.json",
        args.artifacts_dir / "run_evidence_report.json",
        args.require_evidence_report,
        args.require_final_leonardo_objective,
        {
            "count_profile": args.count_profile
            or _expected_run_profile(args.min_generated_per_family, args.max_generated_per_family)
            or "custom",
            "run_profile": _expected_run_profile(args.min_generated_per_family, args.max_generated_per_family),
            "min_generated_per_family": args.min_generated_per_family,
            "max_generated_per_family": args.max_generated_per_family,
            "min_reranker_count": args.min_reranker_count,
            "min_completion_compare_count": args.min_completion_compare_count,
            "min_train_epochs": args.min_train_epochs,
            "required_batch_size": args.required_batch_size,
            "required_checkpoint_sizes": args.required_checkpoint_sizes,
            "required_manifest_stage": args.required_manifest_stage,
            "required_transformer_device": args.required_transformer_device,
            "require_selected_checkpoint": args.require_selected_checkpoint,
            "require_preflight_cuda": args.require_preflight_cuda,
            "require_preflight_eval": args.require_preflight_eval,
            "require_generated_metadata": args.require_generated_metadata,
            "require_readiness": args.require_readiness,
            "require_source_bundle_proof": args.require_source_bundle_proof,
            "prefer_package_evidence": True,
        },
    )
    print(f"Wrote {args.out}")
    if args.require_evidence_report:
        print(f"Objective ready: {summary['objective_ready']}")
    print("Returned package verification passed")


if __name__ == "__main__":
    main()
