from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import (
    final_audit,
    infer,
    leonardo_handoff,
    leonardo_readiness,
    leonardo_return_packet,
    package_submission,
    run_evidence_report,
    validate_run,
    verify_package,
    verify_returned_package,
)


GOOD_SHA = "a" * 64
BAD_SHA = "b" * 64


def _readiness() -> dict[str, object]:
    return {
        "passed": True,
        "require_source_bundle": True,
        "source_bundle": {
            "bundle_sha256": GOOD_SHA,
            "bundle_path": "artifacts/leonardo_source_bundle.zip",
            "verified": True,
            "manifest_source": "artifacts/leonardo_source_bundle_manifest.json",
            "manifest_file_count": 61,
            "manifest_files": [],
            "failures": [],
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "stage": "packaged_with_submissions",
        "run_profile": "max",
        "parameters": {"COMPLETION_CHECKPOINT": "checkpoints/medium/model.pt"},
        "source_bundle": {
            "bundle_sha256": GOOD_SHA,
            "bundle_path": "artifacts/leonardo_source_bundle.zip",
            "verified": True,
            "readiness_passed": True,
            "require_source_bundle": True,
            "manifest_source": "artifacts/leonardo_source_bundle_manifest.json",
            "failures": [],
        },
    }


def _package(readiness: dict[str, object]) -> dict[str, object]:
    return {
        "require_evidence": True,
        "require_readiness": True,
        "require_preflight_cuda": True,
        "require_preflight_eval": True,
        "require_selected_checkpoint": True,
        "require_generated_metadata": True,
        "run_profile": "max",
        "required_checkpoint_sizes": ["tiny", "small", "medium"],
        "required_completion_checkpoint_size": "medium",
        "required_transformer_device": "cuda",
        "required_min_reranker_count": 240,
        "required_min_completion_compare_count": 240,
        "required_min_train_epochs": 6,
        "source_bundle": package_submission._source_bundle_summary(readiness),
    }


def _reranker_payload(best_score: float = 2.0, other_score: float = 1.0) -> dict[str, object]:
    return {
        "best_reranker": "tiny",
        "best_checkpoint": "checkpoints/tiny/model.pt",
        "selection_scope": "checkpoints",
        "runs": [
            {
                "reranker": "tiny",
                "checkpoint": "checkpoints/tiny/model.pt",
                "available": True,
                "selection_eligible": True,
                "selection_score": best_score,
                "checkpoint_sha256": GOOD_SHA,
                "nextstep_count": 240,
                "completion_count": 240,
            },
            {
                "reranker": "small",
                "checkpoint": "checkpoints/small/model.pt",
                "available": True,
                "selection_eligible": True,
                "selection_score": other_score,
                "checkpoint_sha256": GOOD_SHA,
                "nextstep_count": 240,
                "completion_count": 240,
            },
            {
                "reranker": "medium",
                "checkpoint": "checkpoints/medium/model.pt",
                "available": True,
                "selection_eligible": True,
                "selection_score": 0.5,
                "checkpoint_sha256": GOOD_SHA,
                "nextstep_count": 240,
                "completion_count": 240,
            },
        ],
    }


def _infer_selected_checkpoint_failures(
    checkpoint_sha: str | None = None,
    omit_checkpoint_sha: bool = False,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="infer_selected_checkpoint_") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoints" / "tiny" / "model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"selected checkpoint bytes")
        actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        payload = _reranker_payload()
        payload["best_checkpoint"] = str(checkpoint)
        first_run = payload["runs"][0]
        assert isinstance(first_run, dict)
        first_run["checkpoint"] = str(checkpoint)
        if omit_checkpoint_sha:
            first_run.pop("checkpoint_sha256", None)
        else:
            first_run["checkpoint_sha256"] = checkpoint_sha or actual_sha

        metrics_path = Path(temp_dir) / "metrics.json"
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
        _selected_checkpoint, _selected_reranker, failures = infer._read_selected_checkpoint(metrics_path)
        return failures


def _infer_checkpoint_choice(
    requested_checkpoint: Path | None,
    selected_checkpoint: Path | None,
) -> tuple[Path, list[str]]:
    return infer._resolve_inference_checkpoint(
        requested_checkpoint,
        True,
        selected_checkpoint,
    )


def _report_selected_inference_failed_checks(
    checkpoint_used: str = "checkpoints/tiny/model.pt",
    selected_checkpoint: str = "checkpoints/tiny/model.pt",
    checkpoint_sha: str = GOOD_SHA,
) -> list[dict[str, object]]:
    inference_payload = {
        "checkpoint_used": checkpoint_used,
        "selected_checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "require_selected_checkpoint": True,
    }
    return _failed_checks(run_evidence_report._selected_inference_checkpoint_checks(
        _reranker_payload(),
        "reranker",
        inference_payload,
        "inference",
        True,
    ))


def _failed_checks(checks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [check for check in checks if not check["passed"]]


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _event_text(stages: list[str]) -> str:
    return "\n".join(json.dumps({"stage": stage}, sort_keys=True) for stage in stages) + "\n"


def _eval_rows(eval_hash: str = GOOD_SHA) -> list[dict[str, object]]:
    return [
        {
            "label": "valid",
            "exists": True,
            "bytes": 12,
            "rows": 3,
            "sha256": eval_hash,
            "missing_columns": [],
        },
        {
            "label": "anomaly",
            "exists": True,
            "bytes": 12,
            "rows": 3,
            "sha256": eval_hash,
            "missing_columns": [],
        },
    ]


def _eval_staging_payload(eval_hash: str = GOOD_SHA) -> dict[str, object]:
    return {
        "passed": True,
        "failures": [],
        "destinations": _eval_rows(eval_hash),
    }


def _final_audit_eval_staging_failures(
    readiness: dict[str, object],
    eval_hash: str = GOOD_SHA,
    preflight_hash: str = GOOD_SHA,
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        package_dir = Path(tmp)
        evidence_dir = package_dir / "evidence"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "eval_staging_manifest.json").write_text(
            json.dumps(_eval_staging_payload(eval_hash)),
            encoding="utf-8",
        )
        (evidence_dir / "preflight_full_pipeline.json").write_text(
            json.dumps({"eval_inputs": _eval_rows(preflight_hash)}),
            encoding="utf-8",
        )
        return final_audit._check_packaged_eval_staging_evidence(package_dir, readiness)


def _readiness_eval_staging_failures(eval_hash: str = GOOD_SHA, manifest_hash: str = GOOD_SHA) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "eval_staging_manifest.json"
        manifest_path.write_text(json.dumps(_eval_staging_payload(manifest_hash)), encoding="utf-8")
        _status, failures = leonardo_readiness._eval_staging_status(manifest_path, _eval_rows(eval_hash))
        return failures


def _package_submission_eval_staging_failures(
    readiness: dict[str, object],
    eval_hash: str = GOOD_SHA,
    preflight_hash: str = GOOD_SHA,
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        artifacts_dir = Path(tmp)
        (artifacts_dir / "eval_staging_manifest.json").write_text(
            json.dumps(_eval_staging_payload(eval_hash)),
            encoding="utf-8",
        )
        (artifacts_dir / "preflight_full_pipeline.json").write_text(
            json.dumps({"require_eval": True, "eval_inputs": _eval_rows(preflight_hash)}),
            encoding="utf-8",
        )
        return package_submission._check_eval_staging_evidence(artifacts_dir, readiness)


def _returned_package_summary_failures(
    objective_ready: bool = True,
    final_leonardo_objective_ready: bool | None = None,
    require_final_leonardo_objective: bool = False,
    final_passed: bool = True,
    sidecar_hash: str | None = None,
    zip_only_manifest: bool = False,
    stale_report_expected: bool = False,
    stale_final_expected: bool = False,
    min_generated_per_family: int = 150000,
    max_generated_per_family: int = 150000,
) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifacts_dir = root / "artifacts"
        package_dir = artifacts_dir / "submission_package"
        artifacts_dir.mkdir(parents=True)
        package_dir.mkdir(parents=True)
        if zip_only_manifest:
            with zipfile.ZipFile(package_dir / "track1_submission.zip", "w") as zf:
                zf.writestr("package_manifest.json", b"{}")
        else:
            (package_dir / "track1_submission.zip").write_bytes(b"zip")
        zip_sha = hashlib.sha256((package_dir / "track1_submission.zip").read_bytes()).hexdigest()
        (package_dir / "track1_submission.zip.sha256").write_text(
            f"{sidecar_hash or zip_sha}  track1_submission.zip\n",
            encoding="utf-8",
        )
        if not zip_only_manifest:
            (package_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
        run_profile = (
            "max"
            if min_generated_per_family == max_generated_per_family == 150000
            else "standard"
            if min_generated_per_family == max_generated_per_family == 50000
            else ""
        )
        count_profile = run_profile or "custom"
        if final_leonardo_objective_ready is None:
            final_leonardo_objective_ready = (
                objective_ready
                and min_generated_per_family == max_generated_per_family
                and 50000 <= min_generated_per_family <= 150000
            )
        expected = {
            "count_profile": count_profile,
            "run_profile": run_profile,
            "min_generated_per_family": min_generated_per_family,
            "max_generated_per_family": max_generated_per_family,
            "min_reranker_count": 240,
            "min_completion_compare_count": 240,
            "min_train_epochs": 6,
            "required_batch_size": 96,
            "required_checkpoint_sizes": ["tiny", "small", "medium"],
            "required_manifest_stage": "packaged_with_submissions",
            "required_transformer_device": "cuda",
            "require_selected_checkpoint": True,
            "require_preflight_cuda": True,
            "require_preflight_eval": True,
            "require_generated_metadata": True,
            "require_readiness": True,
            "require_source_bundle_proof": True,
            "prefer_package_evidence": True,
        }
        final_expected = dict(expected)
        report_expected = dict(expected)
        if stale_final_expected:
            final_expected["min_generated_per_family"] = 50000
        if stale_report_expected:
            report_expected["required_batch_size"] = 32
        final_payload = {"passed": final_passed, **final_expected}
        (artifacts_dir / "final_audit_summary.json").write_text(
            json.dumps(final_payload),
            encoding="utf-8",
        )
        (artifacts_dir / "run_evidence_report.json").write_text(
            json.dumps({
                "objective_ready": objective_ready,
                "objective_scope": "final_leonardo" if final_leonardo_objective_ready else "custom_verification",
                "final_leonardo_objective_ready": final_leonardo_objective_ready,
                "expected": report_expected,
            }),
            encoding="utf-8",
        )
        payload = verify_returned_package._write_verification_summary(
            artifacts_dir / "returned_package_verification.json",
            artifacts_dir,
            package_dir,
            artifacts_dir / "final_audit_summary.json",
            artifacts_dir / "run_evidence_report.json",
            True,
            require_final_leonardo_objective,
            expected,
            raise_on_failure=False,
        )
        if payload.get("passed") is not True:
            if objective_ready is False and payload.get("objective_ready") is not False:
                return [f"objective_ready false was not recorded: {payload.get('objective_ready')!r}"]
            if final_passed is False and payload.get("final_audit_passed") is not False:
                return [f"final_audit_passed false was not recorded: {payload.get('final_audit_passed')!r}"]
            sidecar_status = payload.get("package_sidecar_status", {})
            if sidecar_hash is not None and isinstance(sidecar_status, dict) and sidecar_status.get("matches_zip") is not False:
                return [f"sidecar mismatch was not recorded: {sidecar_status!r}"]
            if stale_final_expected and not any("Final audit summary expected" in failure for failure in payload.get("failures", [])):
                return [f"final-audit expected mismatch was not recorded: {payload.get('failures', [])!r}"]
            if stale_report_expected and not any("Run evidence report expected" in failure for failure in payload.get("failures", [])):
                return [f"report expected mismatch was not recorded: {payload.get('failures', [])!r}"]
            return list(map(str, payload.get("failures", [])))
        if payload.get("objective_ready") is not True:
            return [f"objective_ready was not true: {payload.get('objective_ready')!r}"]
        if payload.get("final_leonardo_objective_ready") is not final_leonardo_objective_ready:
            return [
                "final_leonardo_objective_ready mismatch: "
                f"{payload.get('final_leonardo_objective_ready')!r}"
            ]
        sidecar_status = payload.get("package_sidecar_status", {})
        if not isinstance(sidecar_status, dict) or sidecar_status.get("matches_zip") is not True:
            return [f"matching package sidecar was not recorded: {sidecar_status!r}"]
        manifest_status = payload.get("package_manifest_status", {})
        if not isinstance(manifest_status, dict) or not manifest_status.get("sha256"):
            return [f"package manifest hash was not recorded: {manifest_status!r}"]
        if zip_only_manifest and "!package_manifest.json" not in str(manifest_status.get("source", "")):
            return [f"ZIP-only package manifest source was not recorded: {manifest_status!r}"]
        return []


def _event_text_with_source(
    stages: list[str],
    include_source: bool = True,
    bundle_sha: str = GOOD_SHA,
    count_per_family: int = 150000,
    require_eval: str = "1",
    require_source_bundle: str = "1",
    completion_checkpoint: str = "checkpoints/medium/model.pt",
) -> str:
    rows = []
    for stage in stages:
        row: dict[str, object] = {
            "stage": stage,
            "run_profile": "max" if count_per_family == 150000 else "custom",
            "parameters": {
                "COUNT_PER_FAMILY": str(count_per_family),
                "REQUIRE_EVAL": require_eval,
                "REQUIRE_SOURCE_BUNDLE": require_source_bundle,
                "COMPLETION_CHECKPOINT": completion_checkpoint,
            },
        }
        if include_source:
            row["source_bundle"] = {
                "bundle_sha256": bundle_sha,
                "require_source_bundle": True,
                "verified": True,
                "readiness_passed": True,
                "manifest_source": "artifacts/leonardo_source_bundle_manifest.json",
                "failures": [],
            }
        rows.append(json.dumps(row, sort_keys=True))
    return "\n".join(rows) + "\n"


def _event_order_failed_checks(events_text: str) -> list[dict[str, object]]:
    return _failed_checks(run_evidence_report._event_stage_checks(
        events_text,
        "events",
        "packaged_with_submissions",
        True,
    ))


def _event_source_failed_checks(events_text: str) -> list[dict[str, object]]:
    readiness = _readiness()
    readiness["count_per_family"] = 150000
    readiness["count_profile"] = "max"
    readiness["require_eval"] = True
    readiness["require_source_bundle"] = True
    return _failed_checks(run_evidence_report._event_stage_checks(
        events_text,
        "events",
        "packaged_with_submissions",
        True,
        readiness,
        True,
        "checkpoints/medium/model.pt",
    ))


def _package_stage_positions(events_text: str) -> dict[str, int]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(events_text)
        temp_path = Path(handle.name)
    try:
        return package_submission._event_log_stage_positions(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _packaged_readiness_failures(
    require_source_bundle: bool,
    require_source_bundle_proof: bool,
    include_eval_launch: bool = True,
    include_split_source_launch: bool = True,
    include_recorded_launch_commands: bool = True,
    include_batch_proof: bool = True,
    include_no_source_proof_flag: bool = True,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="readiness_evidence_") as temp_dir:
        package_dir = Path(temp_dir)
        evidence_dir = package_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        readiness = _readiness()
        readiness["require_eval"] = True
        readiness["count_per_family"] = 150000
        readiness["epochs"] = 6
        readiness["batch_size"] = 96
        readiness["reranker_valid_per_family"] = 40
        readiness["reranker_examples"] = 240
        readiness["count_profile"] = "max"
        readiness["commands_out"] = "artifacts/leonardo_launch_commands.sh"
        readiness["commands"] = {
            "full_pipeline": [
                "sbatch --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,"
                + ("REQUIRE_EVAL=1," if include_eval_launch else "")
                + "RERANKER_VALID_PER_FAMILY=40 scripts/leonardo_full_pipeline.sh"
            ],
            "split_jobs_with_dependencies": [
                "PROBE_JOB=$(sbatch --parsable scripts/leonardo_probe.sh)",
                "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000"
                + (",REQUIRE_SOURCE_BUNDLE=1" if require_source_bundle and include_split_source_launch else "")
                + " scripts/leonardo_generate.sh)",
                "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96"
                + (",REQUIRE_SOURCE_BUNDLE=1" if require_source_bundle and include_split_source_launch else "")
                + " scripts/leonardo_train_scaling.sh)",
                "FINAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,"
                + ("REQUIRE_SOURCE_BUNDLE=1," if require_source_bundle else "")
                + ("REQUIRE_EVAL=1," if include_eval_launch else "")
                + "RERANKER_VALID_PER_FAMILY=40 scripts/leonardo_finalize.sh)",
            ],
        }
        readiness["resume_guidance"] = [
            "Run scripts/leonardo_probe.sh before final run",
            "If interrupted, reuse exact artifacts",
            "Wait for train_scaling_complete before scripts/leonardo_finalize.sh",
            "Run verify_returned_package and run_evidence_report",
        ]
        batch_flag = " --required-batch-size 96" if include_batch_proof else ""
        readiness["verification_commands"] = [
            "python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof" + batch_flag,
            "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness --require-source-bundle-proof" + batch_flag + " --prefer-package-evidence",
            "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
            "python -m industrial_ai.source_bundle_proof_selftest",
        ]
        if not require_source_bundle:
            readiness["require_source_bundle"] = False
            readiness["source_bundle"] = {}
            readiness["verification_commands"] = [
                command.replace(
                    " --require-source-bundle-proof",
                    " --no-require-source-bundle-proof" if include_no_source_proof_flag else "",
                )
                for command in readiness["verification_commands"]
                if "source_bundle_proof_selftest" not in command
            ]
        launch_text = "\n".join([
            "sbatch scripts/leonardo_probe.sh",
            "scripts/leonardo_full_pipeline.sh",
            "--dependency=afterok:${PROBE_JOB}",
            "--dependency=afterok:${GEN_JOB}",
            "--dependency=afterok:${TRAIN_JOB}",
            "verify_returned_package",
            "run_evidence_report",
            "REQUIRE_EVAL=1" if include_eval_launch else "",
            "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh)"
            if require_source_bundle and include_split_source_launch
            else "",
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh)"
            if require_source_bundle and include_split_source_launch
            else "",
            *(
                [
                    *readiness["commands"]["full_pipeline"],
                    *readiness["commands"]["split_jobs_with_dependencies"],
                ]
                if include_recorded_launch_commands
                else []
            ),
            "REQUIRE_SOURCE_BUNDLE=1" if require_source_bundle else "",
            "source_bundle_proof_selftest" if require_source_bundle else "",
            *readiness["verification_commands"],
        ])
        (evidence_dir / "leonardo_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
        (evidence_dir / "leonardo_launch_commands.sh").write_text(launch_text, encoding="utf-8")
        (evidence_dir / "source_bundle_proof_selftest.json").write_text(
            json.dumps({"passed": True, "failures": []}),
            encoding="utf-8",
        )
        return final_audit._check_packaged_readiness_evidence(
            package_dir,
            150000,
            150000,
            240,
            6,
            "max",
            True,
            require_source_bundle_proof,
        )


def _package_submission_launch_failures(
    include_eval_launch: bool = True,
    include_recorded_launch_commands: bool = True,
    include_no_source_proof_flag: bool = True,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="launch_evidence_") as temp_dir:
        artifacts_dir = Path(temp_dir)
        readiness = {
            "commands_out": "artifacts/leonardo_launch_commands.sh",
            "defer_eval_staging": False,
            "require_eval": True,
            "require_source_bundle": False,
            "commands": {
                "full_pipeline": [
                    "sbatch --export=ALL,COUNT_PER_FAMILY=150000,"
                    + ("REQUIRE_EVAL=1," if include_eval_launch else "")
                    + "scripts/leonardo_full_pipeline.sh"
                ],
            },
            "verification_commands": [
                "python -m industrial_ai.verify_returned_package --count-profile max"
                + " --require-final-leonardo-objective"
                + (" --no-require-source-bundle-proof" if include_no_source_proof_flag else "")
                + " --required-batch-size 96",
                "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness"
                + (" --no-require-source-bundle-proof" if include_no_source_proof_flag else "")
                + " --required-batch-size 96 --prefer-package-evidence",
                "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
            ],
        }
        launch_text = "\n".join([
            "sbatch scripts/leonardo_probe.sh",
            "scripts/leonardo_full_pipeline.sh",
            "--dependency=afterok:${PROBE_JOB}",
            "--dependency=afterok:${GEN_JOB}",
            "--dependency=afterok:${TRAIN_JOB}",
            "verify_returned_package",
            "run_evidence_report",
            "REQUIRE_EVAL=1" if include_eval_launch else "",
            *(
                [*readiness["commands"]["full_pipeline"]]
                if include_recorded_launch_commands
                else []
            ),
            *readiness["verification_commands"],
        ])
        (artifacts_dir / "leonardo_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
        (artifacts_dir / "leonardo_launch_commands.sh").write_text(launch_text, encoding="utf-8")
        return package_submission._check_readiness_launch_command_script(artifacts_dir)


def _package_submission_source_launch_failures(
    include_split_source_launch: bool = True,
    include_recorded_launch_commands: bool = True,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="launch_source_evidence_") as temp_dir:
        artifacts_dir = Path(temp_dir)
        readiness = {
            "commands_out": "artifacts/leonardo_launch_commands.sh",
            "defer_eval_staging": False,
            "require_eval": False,
            "require_source_bundle": True,
            "commands": {
                "full_pipeline": [
                    "sbatch --export=ALL,COUNT_PER_FAMILY=150000,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_full_pipeline.sh"
                ],
                "split_jobs_with_dependencies": [
                    "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000"
                    + (",REQUIRE_SOURCE_BUNDLE=1" if include_split_source_launch else "")
                    + " scripts/leonardo_generate.sh)",
                    "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96"
                    + (",REQUIRE_SOURCE_BUNDLE=1" if include_split_source_launch else "")
                    + " scripts/leonardo_train_scaling.sh)",
                ],
            },
            "verification_commands": [
                "python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96",
                "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness --require-source-bundle-proof --required-batch-size 96 --prefer-package-evidence",
                "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
                "python -m industrial_ai.source_bundle_proof_selftest",
            ],
        }
        launch_text = "\n".join([
            "sbatch scripts/leonardo_probe.sh",
            "scripts/leonardo_full_pipeline.sh",
            "--dependency=afterok:${PROBE_JOB}",
            "--dependency=afterok:${GEN_JOB}",
            "--dependency=afterok:${TRAIN_JOB}",
            "verify_returned_package",
            "run_evidence_report",
            "REQUIRE_SOURCE_BUNDLE=1",
            "source_bundle_proof_selftest",
            "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh)"
            if include_split_source_launch
            else "",
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh)"
            if include_split_source_launch
            else "",
            *(
                [
                    *readiness["commands"]["full_pipeline"],
                    *readiness["commands"]["split_jobs_with_dependencies"],
                ]
                if include_recorded_launch_commands
                else []
            ),
            *readiness["verification_commands"],
        ])
        (artifacts_dir / "leonardo_readiness.json").write_text(json.dumps(readiness), encoding="utf-8")
        (artifacts_dir / "leonardo_launch_commands.sh").write_text(launch_text, encoding="utf-8")
        return package_submission._check_readiness_launch_command_script(artifacts_dir)


def _handoff_launch_failures(include_eval_launch: bool = True) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="handoff_evidence_") as temp_dir:
        root = Path(temp_dir)
        bundle = root / "leonardo_source_bundle.zip"
        manifest = root / "leonardo_source_bundle_manifest.json"
        sidecar = root / "leonardo_source_bundle.zip.sha256"
        readiness_path = root / "leonardo_readiness.json"
        launch_path = root / "leonardo_launch_commands.sh"
        selftest_path = root / "source_bundle_proof_selftest.json"
        bundle.write_bytes(b"not a real bundle")
        sidecar.write_text(f"{BAD_SHA}  leonardo_source_bundle.zip\n", encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")
        readiness = {
            "passed": True,
            "defer_eval_staging": True,
            "require_eval": True,
            "require_source_bundle": False,
            "batch_size": 96,
            "source_bundle": {},
            "commands": {
                "full_pipeline": [
                    "sbatch --export=ALL,COUNT_PER_FAMILY=150000,BATCH_SIZE=96,"
                    + ("REQUIRE_EVAL=1," if include_eval_launch else "")
                    + "VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_full_pipeline.sh"
                ],
                "split_jobs_with_dependencies": [
                    "FINAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,BATCH_SIZE=96,"
                    + ("REQUIRE_EVAL=1," if include_eval_launch else "")
                    + "VALID_INPUT=data/eval/eval_input_valid.csv,ANOMALY_INPUT=data/eval/eval_input_anomaly.csv scripts/leonardo_finalize.sh)",
                ],
            },
            "verification_commands": [
                "python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --no-require-source-bundle-proof --required-batch-size 96",
                "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness --no-require-source-bundle-proof --required-batch-size 96 --prefer-package-evidence",
            ],
        }
        launch_text = "\n".join([
            "verify_returned_package",
            "run_evidence_report",
            "REQUIRE_EVAL=1" if include_eval_launch else "",
            *readiness["commands"]["full_pipeline"],
            *readiness["commands"]["split_jobs_with_dependencies"],
            *readiness["verification_commands"],
        ])
        readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
        launch_path.write_text(launch_text, encoding="utf-8")
        selftest_path.write_text(json.dumps({"passed": True, "failures": []}), encoding="utf-8")
        payload = leonardo_handoff.audit_handoff(
            bundle,
            manifest,
            readiness_path,
            launch_path,
            selftest_path,
            False,
        )
        return [
            failure
            for failure in payload["failures"]
            if "eval export" in str(failure)
        ]


def _handoff_strict_readiness_failures(
    include_strict_eval: bool = True,
    include_defer_in_strict: bool = False,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="handoff_readiness_") as temp_dir:
        root = Path(temp_dir)
        bundle = root / "leonardo_source_bundle.zip"
        manifest = root / "leonardo_source_bundle_manifest.json"
        sidecar = root / "leonardo_source_bundle.zip.sha256"
        readiness_path = root / "leonardo_readiness.json"
        launch_path = root / "leonardo_launch_commands.sh"
        selftest_path = root / "source_bundle_proof_selftest.json"
        bundle.write_bytes(b"not a real bundle")
        sidecar.write_text(f"{BAD_SHA}  leonardo_source_bundle.zip\n", encoding="utf-8")
        manifest.write_text("{}", encoding="utf-8")
        strict_command = "python -m industrial_ai.leonardo_readiness --count-profile max --require-source-bundle"
        if include_strict_eval:
            strict_command += " --require-eval"
        if include_defer_in_strict:
            strict_command += " --defer-eval-staging"
        readiness = {
            "passed": True,
            "defer_eval_staging": True,
            "require_eval": True,
            "require_source_bundle": True,
            "batch_size": 96,
            "source_bundle": {
                "bundle_sha256": GOOD_SHA,
                "verified": True,
                "failures": [],
                "handoff_upload_files": [
                    "leonardo_source_bundle.zip",
                    "leonardo_source_bundle.zip.sha256",
                    "leonardo_source_bundle_manifest.json",
                ],
                "handoff_readiness_commands": [strict_command],
                "handoff_deferred_eval_readiness_commands": [
                    "python -m industrial_ai.leonardo_readiness --count-profile max --require-eval --defer-eval-staging --require-source-bundle"
                ],
                "handoff_selftest_commands": ["python -m industrial_ai.source_bundle_proof_selftest"],
                "handoff_audit_commands": ["python -m industrial_ai.leonardo_handoff --require-source-bundle"],
            },
            "commands": {
                "full_pipeline": [
                    "sbatch --export=ALL,COUNT_PER_FAMILY=150000,BATCH_SIZE=96,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_full_pipeline.sh"
                ],
                "split_jobs_with_dependencies": [
                    "FINAL_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,BATCH_SIZE=96,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_finalize.sh)",
                ],
            },
            "verification_commands": [
                "python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96",
                "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness --require-source-bundle-proof --required-batch-size 96 --prefer-package-evidence",
                "python -m industrial_ai.source_bundle_proof_selftest",
            ],
        }
        launch_path.write_text(
            "\n".join([
                *readiness["commands"]["full_pipeline"],
                *readiness["commands"]["split_jobs_with_dependencies"],
                *readiness["verification_commands"],
            ]),
            encoding="utf-8",
        )
        readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
        selftest_path.write_text(json.dumps({"passed": True, "failures": []}), encoding="utf-8")
        payload = leonardo_handoff.audit_handoff(
            bundle,
            manifest,
            readiness_path,
            launch_path,
            selftest_path,
            True,
        )
        return [
            failure
            for failure in payload["failures"]
            if "strict" in str(failure)
        ]


def _handoff_checklist_missing_needles() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="handoff_checklist_") as temp_dir:
        checklist = Path(temp_dir) / "checklist.md"
        payload = {
            "passed": True,
            "bundle_sha256": GOOD_SHA,
            "upload_files": [
                "C:/tmp/leonardo_source_bundle.zip",
                "C:/tmp/leonardo_source_bundle.zip.sha256",
                "C:/tmp/leonardo_source_bundle_manifest.json",
            ],
            "handoff_commands": [
                "sbatch scripts/leonardo_probe.sh",
                "sbatch --export=ALL,COUNT_PER_FAMILY=150000,REQUIRE_EVAL=1,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_full_pipeline.sh",
                "python -m industrial_ai.verify_returned_package --count-profile max --required-manifest-stage packaged_with_submissions --require-selected-checkpoint --require-preflight-cuda --require-preflight-eval --require-generated-metadata --require-readiness --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96",
                "python -m industrial_ai.run_evidence_report --count-profile max --required-manifest-stage packaged_with_submissions --require-selected-checkpoint --require-preflight-cuda --require-preflight-eval --require-generated-metadata --require-readiness --require-source-bundle-proof --required-batch-size 96 --prefer-package-evidence",
            ],
            "warnings": ["stage eval before launch"],
        }
        leonardo_handoff._write_checklist(checklist, payload)
        text = checklist.read_text(encoding="utf-8")
        needles = [
            GOOD_SHA,
            "leonardo_source_bundle.zip",
            "sha256sum -c leonardo_source_bundle.zip.sha256",
            "unzip -o leonardo_source_bundle.zip",
            "leonardo_bundle --verify-bundle leonardo_source_bundle.zip",
            "python -m pip install -r requirements.txt",
            "preflight --require-torch --require-cuda",
            "stage_eval_inputs",
            "leonardo_readiness --count-profile max --require-eval --require-source-bundle",
            "sbatch --export=ALL,COUNT_PER_FAMILY=150000",
            "verify_returned_package --count-profile max",
            "run_evidence_report --count-profile max",
            "--require-readiness",
            "--require-source-bundle-proof",
            "--required-manifest-stage packaged_with_submissions",
            "--require-selected-checkpoint",
            "--require-preflight-cuda",
            "--require-preflight-eval",
            "--require-generated-metadata",
            "--require-final-leonardo-objective",
            "stage eval before launch",
        ]
        return [needle for needle in needles if needle not in text]


def _transfer_packet_failures() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="handoff_packet_") as temp_dir:
        root = Path(temp_dir)
        bundle = root / "leonardo_source_bundle.zip"
        sidecar = root / "leonardo_source_bundle.zip.sha256"
        manifest = root / "leonardo_source_bundle_manifest.json"
        handoff = root / "leonardo_handoff.json"
        checklist = root / "leonardo_handoff_checklist.md"
        readiness = root / "leonardo_readiness.json"
        launch = root / "leonardo_launch_commands.sh"
        selftest = root / "source_bundle_proof_selftest.json"
        packet = root / "leonardo_transfer_packet.zip"
        source_manifest = {"files": []}
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("leonardo_source_bundle_manifest.json", json.dumps(source_manifest))
        bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
        for path, text in {
            sidecar: f"{bundle_hash}  leonardo_source_bundle.zip\n",
            manifest: json.dumps(source_manifest),
            handoff: "{}",
            checklist: "# checklist\n",
            readiness: "{}",
            launch: "sbatch scripts/leonardo_probe.sh\n",
            selftest: json.dumps({"passed": True, "failures": []}),
        }.items():
            path.write_text(text, encoding="utf-8")
        payload = {
            "passed": True,
            "bundle_sha256": bundle_hash,
            "upload_files": [str(bundle), str(sidecar), str(manifest)],
        }
        packet_manifest = leonardo_handoff._write_transfer_packet(
            packet,
            payload,
            checklist,
            handoff,
            readiness,
            launch,
            selftest,
        )
        failures: list[str] = []
        sidecar_path = packet.with_suffix(packet.suffix + ".sha256")
        manifest_path = packet.with_name(packet.stem + "_manifest.json")
        if not packet.exists():
            failures.append("transfer packet ZIP was not written")
        if not sidecar_path.exists():
            failures.append("transfer packet checksum sidecar was not written")
        if not manifest_path.exists():
            failures.append("transfer packet external manifest was not written")
        if sidecar_path.exists():
            recorded = sidecar_path.read_text(encoding="utf-8").split()[0]
            if recorded != hashlib.sha256(packet.read_bytes()).hexdigest():
                failures.append("transfer packet checksum sidecar does not match packet ZIP")
        expected_entries = {
            "leonardo_source_bundle.zip",
            "leonardo_source_bundle.zip.sha256",
            "leonardo_source_bundle_manifest.json",
            "leonardo_handoff.json",
            "leonardo_handoff_checklist.md",
            "leonardo_readiness.json",
            "leonardo_launch_commands.sh",
            "source_bundle_proof_selftest.json",
            "leonardo_transfer_packet_manifest.json",
        }
        with zipfile.ZipFile(packet, "r") as zf:
            names = set(zf.namelist())
            missing = sorted(expected_entries - names)
            if missing:
                failures.append("transfer packet missing entries: " + ", ".join(missing))
            embedded = json.loads(zf.read("leonardo_transfer_packet_manifest.json").decode("utf-8-sig"))
        if packet_manifest.get("entry_count") != len(expected_entries) - 1:
            failures.append(f"transfer packet entry_count is wrong: {packet_manifest.get('entry_count')!r}")
        if embedded.get("bundle_sha256") != bundle_hash:
            failures.append("transfer packet embedded manifest lost bundle SHA")
        unpack_commands = [str(command) for command in embedded.get("unpack_commands", [])]
        unpack_text = "\n".join(unpack_commands)
        if "sha256sum -c leonardo_source_bundle.zip.sha256" not in unpack_text:
            failures.append("transfer packet embedded manifest missing bundle sidecar verification command")
        try:
            source_unzip_index = unpack_commands.index("unzip -o leonardo_source_bundle.zip")
            packet_verify_index = unpack_commands.index(
                "python -m industrial_ai.leonardo_handoff --verify-transfer-packet leonardo_transfer_packet.zip --verify-fresh-unpack"
            )
        except ValueError as exc:
            failures.append(f"transfer packet embedded manifest missing fresh-dir-safe command: {exc}")
        else:
            if packet_verify_index <= source_unzip_index:
                failures.append("transfer packet verifier command appears before source bundle unpack")
        external = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
        packet_info = external.get("packet", {}) if isinstance(external, dict) else {}
        if not isinstance(packet_info, dict) or packet_info.get("sha256") != hashlib.sha256(packet.read_bytes()).hexdigest():
            failures.append("transfer packet external manifest does not record packet SHA")
        verify_failures = leonardo_handoff.verify_transfer_packet(
            packet,
            sidecar_path,
            manifest_path,
            verify_fresh_unpack=True,
        )
        if verify_failures:
            failures.append("transfer packet verifier rejected valid packet: " + "; ".join(verify_failures))
        tampered = root / "tampered_transfer_packet.zip"
        shutil.copyfile(packet, tampered)
        with zipfile.ZipFile(tampered, "a") as zf:
            zf.writestr("unexpected.txt", "tamper")
        tampered_sidecar = tampered.with_suffix(tampered.suffix + ".sha256")
        tampered_manifest = tampered.with_name(tampered.stem + "_manifest.json")
        tampered_sidecar.write_text(
            f"{hashlib.sha256(tampered.read_bytes()).hexdigest()}  {tampered.name}\n",
            encoding="utf-8",
        )
        shutil.copyfile(manifest_path, tampered_manifest)
        if not leonardo_handoff.verify_transfer_packet(tampered, tampered_sidecar, tampered_manifest):
            failures.append("transfer packet verifier accepted unexpected ZIP entry")
        return failures


def _return_packet_failures() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="return_packet_") as temp_dir:
        root = Path(temp_dir)
        artifacts_dir = root / "artifacts"
        package_dir = artifacts_dir / "submission_package"
        package_dir.mkdir(parents=True)
        manifest = {"files": [], "evidence": []}
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        (package_dir / "package_manifest.json").write_bytes(manifest_bytes)
        zip_path = package_dir / "track1_submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("package_manifest.json", manifest_bytes)
        zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        (package_dir / "track1_submission.zip.sha256").write_text(
            f"{zip_sha}  track1_submission.zip\n",
            encoding="utf-8",
        )
        (artifacts_dir / "final_audit_summary.json").write_text(
            json.dumps({"passed": True}),
            encoding="utf-8",
        )
        (artifacts_dir / "run_evidence_report.json").write_text(
            json.dumps({"objective_ready": True, "final_leonardo_objective_ready": True}),
            encoding="utf-8",
        )
        (artifacts_dir / "run_evidence_report.md").write_text("# report\n", encoding="utf-8")
        (artifacts_dir / "returned_package_verification.json").write_text(
            json.dumps({
                "passed": True,
                "objective_ready": True,
                "objective_scope": "final_leonardo",
                "final_leonardo_objective_ready": True,
                "package_zip_sha256": zip_sha,
                "package_sidecar_sha256": hashlib.sha256(
                    (package_dir / "track1_submission.zip.sha256").read_bytes()
                ).hexdigest(),
                "package_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            }),
            encoding="utf-8",
        )
        packet = artifacts_dir / "leonardo_return_packet.zip"
        packet_manifest = leonardo_return_packet.create_return_packet(artifacts_dir, package_dir, packet)
        failures: list[str] = []
        if not packet.exists():
            failures.append("return packet ZIP was not written")
        sidecar = packet.with_suffix(packet.suffix + ".sha256")
        manifest_path = packet.with_name(packet.stem + "_manifest.json")
        if not sidecar.exists():
            failures.append("return packet checksum sidecar was not written")
        if not manifest_path.exists():
            failures.append("return packet external manifest was not written")
        if packet_manifest.get("entry_count") != 7:
            failures.append(f"return packet entry_count is wrong: {packet_manifest.get('entry_count')!r}")
        verify_failures = leonardo_return_packet.verify_return_packet(
            packet,
            sidecar,
            manifest_path,
            require_final_leonardo_objective=True,
        )
        if verify_failures:
            failures.append("return packet verifier rejected valid packet: " + "; ".join(verify_failures))
        tampered = artifacts_dir / "tampered_return_packet.zip"
        shutil.copyfile(packet, tampered)
        with zipfile.ZipFile(tampered, "a") as zf:
            zf.writestr("unexpected.txt", "tamper")
        tampered_sidecar = tampered.with_suffix(tampered.suffix + ".sha256")
        tampered_manifest = tampered.with_name(tampered.stem + "_manifest.json")
        tampered_sidecar.write_text(
            f"{hashlib.sha256(tampered.read_bytes()).hexdigest()}  {tampered.name}\n",
            encoding="utf-8",
        )
        shutil.copyfile(manifest_path, tampered_manifest)
        if not leonardo_return_packet.verify_return_packet(tampered, tampered_sidecar, tampered_manifest):
            failures.append("return packet verifier accepted unexpected ZIP entry")
        return failures


def _readiness_report_failed_checks(
    include_eval_launch: bool = True,
    include_split_source_launch: bool = True,
    include_recorded_launch_commands: bool = True,
) -> list[dict[str, object]]:
    readiness = _readiness()
    readiness["require_eval"] = True
    readiness["batch_size"] = 96
    readiness["commands"] = {
        "full_pipeline": [
            "sbatch --export=ALL,"
            + "REQUIRE_SOURCE_BUNDLE=1,"
            + ("REQUIRE_EVAL=1," if include_eval_launch else "")
            + "COUNT_PER_FAMILY=150000,BATCH_SIZE=96 scripts/leonardo_full_pipeline.sh"
        ],
        "split_jobs_with_dependencies": [
            "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000"
            + (",REQUIRE_SOURCE_BUNDLE=1" if include_split_source_launch else "")
            + " scripts/leonardo_generate.sh)",
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96"
            + (",REQUIRE_SOURCE_BUNDLE=1" if include_split_source_launch else "")
            + " scripts/leonardo_train_scaling.sh)",
        ],
    }
    readiness["verification_commands"] = [
        "python -m industrial_ai.verify_returned_package --count-profile max --require-final-leonardo-objective --require-source-bundle-proof --required-batch-size 96",
        "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness --require-source-bundle-proof --required-batch-size 96 --prefer-package-evidence",
        "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
        "python -m industrial_ai.source_bundle_proof_selftest",
    ]
    launch_text = "\n".join([
        "verify_returned_package",
        "run_evidence_report",
        "REQUIRE_SOURCE_BUNDLE=1",
        "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_generate.sh)"
        if include_split_source_launch
        else "",
        "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96,REQUIRE_SOURCE_BUNDLE=1 scripts/leonardo_train_scaling.sh)"
        if include_split_source_launch
        else "",
        "source_bundle_proof_selftest",
        "REQUIRE_EVAL=1" if include_eval_launch else "",
        *(
            [
                *readiness["commands"]["full_pipeline"],
                *readiness["commands"]["split_jobs_with_dependencies"],
            ]
            if include_recorded_launch_commands
            else []
        ),
        *readiness["verification_commands"],
    ])
    checks, _summary = run_evidence_report._readiness_checks(
        readiness,
        "readiness",
        {"passed": True},
        "source_bundle_proof_selftest",
        launch_text,
        "launch",
        "",
        True,
        True,
    )
    return _failed_checks(checks)


def _readiness_report_no_source_failed_checks(
    include_no_source_proof_flag: bool = True,
) -> list[dict[str, object]]:
    readiness = _readiness()
    readiness["require_source_bundle"] = False
    readiness["source_bundle"] = {}
    readiness["require_eval"] = True
    readiness["count_profile"] = "max"
    readiness["count_per_family"] = 150000
    readiness["batch_size"] = 96
    readiness["commands"] = {
        "full_pipeline": [
            "sbatch --export=ALL,REQUIRE_EVAL=1,COUNT_PER_FAMILY=150000,BATCH_SIZE=96 scripts/leonardo_full_pipeline.sh"
        ],
        "split_jobs_with_dependencies": [
            "GEN_JOB=$(sbatch --parsable --dependency=afterok:${PROBE_JOB} --export=ALL,COUNT_PER_FAMILY=150000 scripts/leonardo_generate.sh)",
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:${GEN_JOB} --export=ALL,COUNT_PER_FAMILY=150000,EPOCHS=6,BATCH_SIZE=96 scripts/leonardo_train_scaling.sh)",
        ],
    }
    readiness["verification_commands"] = [
        "python -m industrial_ai.verify_returned_package --count-profile max"
        + " --require-final-leonardo-objective"
        + (" --no-require-source-bundle-proof" if include_no_source_proof_flag else "")
        + " --required-batch-size 96",
        "python -m industrial_ai.run_evidence_report --count-profile max --require-readiness"
        + (" --no-require-source-bundle-proof" if include_no_source_proof_flag else "")
        + " --required-batch-size 96 --prefer-package-evidence",
        "python -m industrial_ai.leonardo_return_packet --verify artifacts/leonardo_return_packet.zip --require-final-leonardo-objective",
    ]
    launch_text = "\n".join([
        "verify_returned_package",
        "run_evidence_report",
        "REQUIRE_EVAL=1",
        *readiness["commands"]["full_pipeline"],
        *readiness["commands"]["split_jobs_with_dependencies"],
        *readiness["verification_commands"],
    ])
    checks, _summary = run_evidence_report._readiness_checks(
        readiness,
        "readiness",
        {},
        "",
        launch_text,
        "launch",
        "max",
        True,
        False,
    )
    return _failed_checks(checks)


def _packaged_script_source_bundle_failures(
    tamper_script: bool = False,
    omit_manifest_files: bool = False,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="script_source_bundle_") as temp_dir:
        package_dir = Path(temp_dir)
        evidence_dir = package_dir / "evidence"
        readiness = _readiness()
        manifest_files = []
        for script in sorted(final_audit.LEONARDO_SCRIPT_PATHS):
            data = f"#!/usr/bin/env bash\n# {script}\n".encode("utf-8")
            path = evidence_dir / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest_files.append({
                "path": script,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        if tamper_script:
            (evidence_dir / "scripts" / "leonardo_finalize.sh").write_text(
                "#!/usr/bin/env bash\n# tampered\n",
                encoding="utf-8",
            )
        if not omit_manifest_files:
            readiness["source_bundle"]["manifest_files"] = manifest_files
        return final_audit._packaged_scripts_source_bundle_failures(package_dir, readiness)


def _packaged_source_snapshot_failures(
    tamper_source: bool = False,
    omit_source: bool = False,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="source_snapshot_") as temp_dir:
        package_dir = Path(temp_dir)
        evidence_dir = package_dir / "evidence"
        readiness = _readiness()
        manifest_files = []
        for rel_path in ("industrial_ai/package_submission.py", "industrial_ai/final_audit.py"):
            data = f"# {rel_path}\nVALUE = 1\n".encode("utf-8")
            path = evidence_dir / "source_snapshot" / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (omit_source and rel_path.endswith("final_audit.py")):
                path.write_bytes(data)
            manifest_files.append({
                "path": rel_path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        if tamper_source:
            (evidence_dir / "source_snapshot" / "industrial_ai" / "package_submission.py").write_text(
                "# tampered\nVALUE = 2\n",
                encoding="utf-8",
            )
        readiness["source_bundle"]["manifest_files"] = manifest_files
        return final_audit._packaged_source_snapshot_source_bundle_failures(package_dir, readiness)


def _report_source_identity_failed_checks(
    tamper_source: bool = False,
    omit_source: bool = False,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="report_source_identity_") as temp_dir:
        package_dir = Path(temp_dir)
        evidence_dir = package_dir / "evidence"
        readiness = _readiness()
        manifest_files = []
        for rel_path in ("scripts/leonardo_full_pipeline.sh", "industrial_ai/package_submission.py"):
            data = f"# {rel_path}\nVALUE = 1\n".encode("utf-8")
            package_rel = (
                Path(rel_path)
                if rel_path.startswith("scripts/")
                else Path("source_snapshot") / rel_path
            )
            path = evidence_dir / package_rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not (omit_source and rel_path.startswith("industrial_ai/")):
                path.write_bytes(data)
            manifest_files.append({
                "path": rel_path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
        if tamper_source:
            (evidence_dir / "source_snapshot" / "industrial_ai" / "package_submission.py").write_text(
                "# tampered\nVALUE = 2\n",
                encoding="utf-8",
            )
        readiness["source_bundle"]["manifest_files"] = manifest_files
        checks, _summary = run_evidence_report._package_source_identity_checks(
            package_dir,
            readiness,
            True,
            True,
        )
        return _failed_checks(checks)


def _report_reads_package_manifest_from_zip() -> bool:
    with tempfile.TemporaryDirectory(prefix="report_package_manifest_") as temp_dir:
        package_dir = Path(temp_dir)
        zip_path = package_dir / "track1_submission.zip"
        manifest = {
            "require_evidence": True,
            "require_readiness": True,
            "run_profile": "max",
            "required_checkpoint_sizes": ["tiny", "small", "medium"],
            "required_completion_checkpoint_size": "medium",
            "source_bundle": _package(_readiness())["source_bundle"],
        }
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("package_manifest.json", json.dumps(manifest))
        payload, source = run_evidence_report._read_package_json(package_dir, Path("package_manifest.json"))
        return bool(payload) and payload.get("run_profile") == "max" and "!package_manifest.json" in source


def _report_final_audit_failed_checks(
    artifacts_dir: str = "artifacts",
    package_dir: str = "artifacts/submission_package",
    count: int = 150000,
    require_source_bundle_proof: bool = True,
    package_zip_sha256: str | None = None,
    package_sidecar_sha256: str | None = None,
    package_manifest_sha256: str | None = None,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="report_final_audit_") as temp_dir:
        base = Path(temp_dir)
        expected_artifacts_dir = base / "artifacts"
        expected_package_dir = expected_artifacts_dir / "submission_package"
        expected_package_dir.mkdir(parents=True)
        manifest_bytes = json.dumps({"run_profile": "max"}, sort_keys=True).encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        (expected_package_dir / "package_manifest.json").write_bytes(manifest_bytes)
        zip_path = expected_package_dir / "track1_submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("package_manifest.json", manifest_bytes)
        zip_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        sidecar_bytes = f"{zip_sha}  track1_submission.zip\n".encode("utf-8")
        (expected_package_dir / "track1_submission.zip.sha256").write_bytes(sidecar_bytes)
        sidecar_sha = hashlib.sha256(sidecar_bytes).hexdigest()
        payload = {
            "passed": True,
            "artifacts_dir": str(base / artifacts_dir),
            "package_dir": str(base / package_dir),
            "package_zip_sha256": package_zip_sha256 if package_zip_sha256 is not None else zip_sha,
            "package_sidecar_sha256": (
                package_sidecar_sha256
                if package_sidecar_sha256 is not None
                else sidecar_sha
            ),
            "package_manifest_sha256": (
                package_manifest_sha256
                if package_manifest_sha256 is not None
                else manifest_sha
            ),
            "required_manifest_stage": "packaged_with_submissions",
            "run_profile": "max",
            "min_generated_per_family": count,
            "max_generated_per_family": count,
            "required_checkpoint_sizes": ["tiny", "small", "medium"],
            "required_completion_checkpoint_size": "medium",
            "min_reranker_count": 240,
            "min_completion_compare_count": 240,
            "min_train_epochs": 6,
            "required_batch_size": 96,
            "required_transformer_device": "cuda",
            "require_readiness": True,
            "require_source_bundle_proof": require_source_bundle_proof,
        }
        return _failed_checks(run_evidence_report._final_audit_checks(
            payload,
            "final_audit_summary.json",
            expected_artifacts_dir,
            expected_package_dir,
            "max",
            "packaged_with_submissions",
            ["tiny", "small", "medium"],
            "medium",
            150000,
            150000,
            240,
            240,
            6,
            96,
            "cuda",
            True,
            True,
            True,
        ))


def _checkpoint_staleness_failures(
    config_override: dict[str, int] | None = None,
    batch_size: int = 96,
    required_batch_size: int = 96,
    source_bundle_sha256: str = GOOD_SHA,
    expected_source_bundle_sha256: str = GOOD_SHA,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="checkpoint_staleness_") as temp_dir:
        checkpoint_dir = Path(temp_dir)
        run_dir = checkpoint_dir / "tiny"
        run_dir.mkdir(parents=True)
        model_path = run_dir / "model.pt"
        model_path.write_bytes(b"tiny checkpoint")
        train_log_text = json.dumps([{"epoch": epoch, "loss": 0.5} for epoch in range(1, 7)])
        summary = {
            "model_size": "tiny",
            "config": config_override or validate_run.MODEL_CONFIGS["tiny"],
            "epochs": 6,
            "batch_size": batch_size,
            "device": "cuda",
            "requested_device": "cuda",
            "device_fallback": False,
            "num_sequences": 3,
            "corpus_fingerprint": "fingerprint",
            "family_counts": {"IC": 1, "IGBT": 1, "MOSFET": 1},
            "final_loss": 0.5,
            "model_sha256": hashlib.sha256(b"tiny checkpoint").hexdigest(),
            "train_log_sha256": hashlib.sha256(train_log_text.encode("utf-8")).hexdigest(),
            "source_bundle_sha256": source_bundle_sha256,
            "source_bundle_required": True,
            "source_bundle_verified": True,
        }
        (run_dir / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "train_log.json").write_text(train_log_text, encoding="utf-8")
        return validate_run._failures_from_checkpoints(
            checkpoint_dir,
            ["tiny"],
            {"IC": 1, "IGBT": 1, "MOSFET": 1},
            "fingerprint",
            "cuda",
            6,
            required_batch_size,
            expected_source_bundle_sha256,
        )


def _package_verifier_reads_manifest_from_zip() -> bool:
    with tempfile.TemporaryDirectory(prefix="verify_package_manifest_") as temp_dir:
        package_dir = Path(temp_dir)
        file_bytes = b"sequence_id,prediction\n1,A\n"
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        manifest = {
            "files": [
                {
                    "path": str(package_dir / "nextstep.csv"),
                    "sha256": file_hash,
                    "rows": 1,
                }
            ],
            "evidence": [],
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        zip_path = package_dir / "track1_submission.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("nextstep.csv", file_bytes)
            zf.writestr("package_manifest.json", manifest_bytes)
        sidecar = package_dir / "track1_submission.zip.sha256"
        sidecar.write_text(f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  track1_submission.zip\n", encoding="utf-8")
        final_payload, final_failures = final_audit._read_package_root_json(package_dir, Path("package_manifest.json"))
        return (
            not verify_package.verify_package(package_dir)
            and bool(final_payload)
            and not final_failures
            and not (package_dir / "package_manifest.json").exists()
        )


def _zip_only_checkpoint_source_failures(source_bundle_sha256: str = GOOD_SHA) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="zip_checkpoint_source_") as temp_dir:
        package_dir = Path(temp_dir)
        checkpoint_sha = "c" * 64
        train_log_text = json.dumps([{"epoch": 1, "loss": 1.0, "perplexity": 2.0}])
        train_log_sha = hashlib.sha256(train_log_text.encode("utf-8")).hexdigest()
        train_summary = {
            "model_size": "tiny",
            "config": validate_run.MODEL_CONFIGS["tiny"],
            "batch_size": 96,
            "model_sha256": checkpoint_sha,
            "train_log_sha256": train_log_sha,
            "source_bundle_sha256": source_bundle_sha256,
            "source_bundle_required": True,
            "source_bundle_verified": True,
        }
        completion_metrics = {
            "transformer_device": "cuda",
            "checkpoint_used": "checkpoints/tiny/model.pt",
            "checkpoint_sha256": checkpoint_sha,
        }
        reranker_metrics = {
            "transformer_device": "cuda",
            "selection_scope": "checkpoints",
            "best_reranker": "tiny",
            "best_checkpoint": "checkpoints/tiny/model.pt",
            "runs": [
                {
                    "reranker": "tiny",
                    "checkpoint": "checkpoints/tiny/model.pt",
                    "checkpoint_sha256": checkpoint_sha,
                    "available": True,
                    "selection_eligible": True,
                    "selection_score": 1.0,
                }
            ],
        }
        inference_summary = {
            "transformer_device": "cuda",
            "checkpoint_used": "checkpoints/tiny/model.pt",
            "checkpoint_sha256": checkpoint_sha,
            "transformer_available": True,
        }
        with zipfile.ZipFile(package_dir / "track1_submission.zip", "w") as zf:
            zf.writestr("evidence/checkpoints/tiny/train_summary.json", json.dumps(train_summary))
            zf.writestr("evidence/checkpoints/tiny/train_log.json", train_log_text)
            zf.writestr("evidence/completion_compare/metrics.json", json.dumps(completion_metrics))
            zf.writestr("evidence/reranker_compare/metrics.json", json.dumps(reranker_metrics))
            zf.writestr("evidence/reranker_compare/best_checkpoint.txt", "checkpoints/tiny/model.pt\n")
            zf.writestr("evidence/inference_summary.json", json.dumps(inference_summary))
        return final_audit._check_packaged_execution_evidence(
            package_dir,
            "cuda",
            True,
            ["tiny"],
            "tiny",
            96,
            GOOD_SHA,
        )


def _checkpoint_train_log_hash_failures(tamper_log: bool = False) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="checkpoint_log_hash_") as temp_dir:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        run_dir = checkpoint_dir / "tiny"
        run_dir.mkdir(parents=True)
        model_path = run_dir / "model.pt"
        log_path = run_dir / "train_log.json"
        model_path.write_bytes(b"model")
        log_path.write_text(json.dumps([{"epoch": 1, "loss": 1.0, "perplexity": 2.0}], indent=2), encoding="utf-8")
        summary = {
            "model_size": "tiny",
            "config": validate_run.MODEL_CONFIGS["tiny"],
            "epochs": 1,
            "batch_size": 96,
            "num_sequences": 1,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "train_log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
            "corpus_fingerprint": GOOD_SHA,
            "family_counts": {"IC": 1},
            "final_loss": 1.0,
            "device": "cuda",
            "requested_device": "cuda",
            "device_fallback": False,
        }
        (run_dir / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        if tamper_log:
            log_path.write_text(json.dumps([{"epoch": 1, "loss": 9.0, "perplexity": 9.0}], indent=2), encoding="utf-8")
        return validate_run._failures_from_checkpoints(
            checkpoint_dir,
            ["tiny"],
            {"IC": 1},
            GOOD_SHA,
            "cuda",
            1,
            96,
        )


def _report_checkpoint_artifact_hash_failures(tamper_log: bool = False) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="report_checkpoint_hash_") as temp_dir:
        package_dir = Path(temp_dir)
        evidence_dir = package_dir / "evidence" / "checkpoints" / "tiny"
        evidence_dir.mkdir(parents=True)
        log_text = json.dumps([{"epoch": 1, "loss": 1.0, "perplexity": 2.0}], indent=2)
        log_path = evidence_dir / "train_log.json"
        log_path.write_text(
            (
                json.dumps([{"epoch": 1, "loss": 9.0, "perplexity": 9.0}], indent=2)
                if tamper_log
                else log_text
            ),
            encoding="utf-8",
        )
        log_hash = hashlib.sha256(log_path.read_bytes() if not tamper_log else log_text.encode("utf-8")).hexdigest()
        summary = {
            "model_size": "tiny",
            "model_sha256": GOOD_SHA,
            "train_log_sha256": log_hash,
        }
        (evidence_dir / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return _failed_checks(run_evidence_report._checkpoint_artifact_hash_checks(
            package_dir,
            ["tiny"],
        ))


def run_selftest() -> dict[str, object]:
    readiness = _readiness()
    readiness_with_eval = json.loads(json.dumps(readiness))
    readiness_with_eval["require_eval"] = True
    readiness_with_eval["defer_eval_staging"] = False
    readiness_with_eval["eval_inputs"] = _eval_rows()
    manifest = _manifest()
    package = _package(readiness)
    failures: list[str] = []

    _assert(
        not _failed_checks(run_evidence_report._eval_staging_checks(
            _eval_staging_payload(),
            "eval_staging_manifest.json",
            readiness_with_eval,
            {"eval_inputs": _eval_rows()},
            True,
            True,
        )),
        "run_evidence_report rejected valid eval staging hash evidence",
        failures,
    )
    _assert(
        not _readiness_eval_staging_failures(),
        "leonardo_readiness rejected valid eval staging hash evidence",
        failures,
    )
    _assert(
        _readiness_eval_staging_failures(manifest_hash=BAD_SHA),
        "leonardo_readiness accepted stale eval staging manifest hash evidence",
        failures,
    )
    tampered_eval_readiness = json.loads(json.dumps(readiness_with_eval))
    tampered_eval_readiness["eval_inputs"][0]["sha256"] = BAD_SHA
    _assert(
        _failed_checks(run_evidence_report._eval_staging_checks(
            _eval_staging_payload(),
            "eval_staging_manifest.json",
            tampered_eval_readiness,
            {"eval_inputs": _eval_rows()},
            True,
            True,
        )),
        "run_evidence_report accepted eval staging readiness hash mismatch",
        failures,
    )
    _assert(
        not _final_audit_eval_staging_failures(readiness_with_eval),
        "final_audit rejected valid packaged eval staging hash evidence",
        failures,
    )
    _assert(
        _final_audit_eval_staging_failures(readiness_with_eval, preflight_hash=BAD_SHA),
        "final_audit accepted eval staging preflight hash mismatch",
        failures,
    )
    _assert(
        not _package_submission_eval_staging_failures(readiness_with_eval),
        "package_submission rejected valid eval staging hash evidence",
        failures,
    )
    tampered_package_readiness = json.loads(json.dumps(readiness_with_eval))
    tampered_package_readiness["eval_inputs"][0]["sha256"] = BAD_SHA
    _assert(
        _package_submission_eval_staging_failures(tampered_package_readiness),
        "package_submission accepted eval staging readiness hash mismatch",
        failures,
    )
    _assert(
        _package_submission_eval_staging_failures(readiness_with_eval, preflight_hash=BAD_SHA),
        "package_submission accepted eval staging preflight hash mismatch",
        failures,
    )
    _assert(
        not _returned_package_summary_failures(),
        "verify_returned_package rejected valid objective-ready evidence summary",
        failures,
    )
    _assert(
        not _returned_package_summary_failures(
            min_generated_per_family=1,
            max_generated_per_family=1,
            final_leonardo_objective_ready=False,
        ),
        "verify_returned_package rejected valid local-smoke summary that is not final-Leonardo ready",
        failures,
    )
    _assert(
        _returned_package_summary_failures(
            min_generated_per_family=1,
            max_generated_per_family=1,
            final_leonardo_objective_ready=False,
            require_final_leonardo_objective=True,
        ),
        "verify_returned_package accepted local-smoke summary when final Leonardo objective was required",
        failures,
    )
    _assert(
        _returned_package_summary_failures(objective_ready=False),
        "verify_returned_package accepted run evidence report with objective_ready=false",
        failures,
    )
    _assert(
        _returned_package_summary_failures(final_passed=False),
        "verify_returned_package accepted final_audit_summary with passed=false",
        failures,
    )
    _assert(
        _returned_package_summary_failures(sidecar_hash=BAD_SHA),
        "verify_returned_package accepted returned package summary with stale ZIP checksum sidecar",
        failures,
    )
    _assert(
        not _returned_package_summary_failures(zip_only_manifest=True),
        "verify_returned_package rejected returned package summary when package_manifest.json exists only in ZIP",
        failures,
    )
    _assert(
        not verify_returned_package._generated_count_args_failure(150000, 150000),
        "verify_returned_package rejected exact custom generated-count target",
        failures,
    )
    _assert(
        verify_returned_package._generated_count_args_failure(50000, 150000),
        "verify_returned_package accepted ranged generated-count target",
        failures,
    )
    _assert(
        _returned_package_summary_failures(stale_final_expected=True),
        "verify_returned_package accepted final_audit_summary with stale expected thresholds",
        failures,
    )
    _assert(
        _returned_package_summary_failures(stale_report_expected=True),
        "verify_returned_package accepted run_evidence_report with stale expected thresholds",
        failures,
    )
    _assert(
        not _checkpoint_train_log_hash_failures(),
        "validate_run rejected valid checkpoint train_log_sha256 evidence",
        failures,
    )
    _assert(
        _checkpoint_train_log_hash_failures(tamper_log=True),
        "validate_run accepted checkpoint train_log.json hash mismatch",
        failures,
    )
    _assert(
        not _report_checkpoint_artifact_hash_failures(),
        "run_evidence_report rejected valid packaged checkpoint train-log hash evidence",
        failures,
    )
    _assert(
        any(
            check["name"] == "tiny train log hash binding"
            for check in _report_checkpoint_artifact_hash_failures(tamper_log=True)
        ),
        "run_evidence_report accepted packaged checkpoint train-log hash mismatch",
        failures,
    )
    _assert(
        Path("eval_staging_manifest.json") in package_submission._required_evidence_paths(
            ["tiny", "small", "medium"],
            True,
            True,
            True,
        ),
        "package_submission strict evidence requirements omitted eval staging manifest",
        failures,
    )
    _assert(
        Path("eval_staging_manifest.json") not in package_submission._required_evidence_paths(
            ["tiny"],
            False,
            False,
            False,
        ),
        "package_submission local evidence requirements unexpectedly require eval staging manifest",
        failures,
    )

    _assert(
        not package_submission._manifest_source_bundle_failures(manifest, readiness, "manifest"),
        "package_submission rejected valid manifest source-bundle evidence",
        failures,
    )
    _assert(
        not final_audit._manifest_source_bundle_failures(manifest, readiness, "manifest"),
        "final_audit rejected valid manifest source-bundle evidence",
        failures,
    )
    _assert(
        not final_audit._package_source_bundle_failures(package, readiness),
        "final_audit rejected valid package source-bundle evidence",
        failures,
    )
    event_parameters = {
        "stage": "packaged_with_submissions",
        "run_profile": "max",
        "parameters": {
            "COUNT_PER_FAMILY": "150000",
            "REQUIRE_EVAL": "1",
            "REQUIRE_SOURCE_BUNDLE": "1",
            "COMPLETION_CHECKPOINT": "checkpoints/medium/model.pt",
        },
    }
    _assert(
        not final_audit._manifest_parameter_failures(
            event_parameters,
            "event",
            "max",
            150000,
            "checkpoints/medium/model.pt",
            True,
            True,
        ),
        "final_audit rejected valid run-manifest event parameters",
        failures,
    )
    _assert(
        not package_submission._manifest_parameter_failures(
            event_parameters,
            "event",
            "max",
            150000,
            "checkpoints/medium/model.pt",
            True,
            True,
        ),
        "package_submission rejected valid run-manifest event parameters",
        failures,
    )
    wrong_event_parameters = json.loads(json.dumps(event_parameters))
    wrong_event_parameters["parameters"]["COUNT_PER_FAMILY"] = "50000"
    _assert(
        final_audit._manifest_parameter_failures(
            wrong_event_parameters,
            "event",
            "max",
            150000,
            "checkpoints/medium/model.pt",
            True,
            True,
        ),
        "final_audit accepted run-manifest event parameters with wrong count",
        failures,
    )
    _assert(
        package_submission._manifest_parameter_failures(
            wrong_event_parameters,
            "event",
            "max",
            150000,
            "checkpoints/medium/model.pt",
            True,
            True,
        ),
        "package_submission accepted run-manifest event parameters with wrong count",
        failures,
    )
    _assert(
        not _failed_checks(run_evidence_report._manifest_checks(
            manifest,
            "manifest",
            readiness,
            {"passed": True},
            "final_audit",
            "packaged_with_submissions",
            "max",
            "checkpoints/medium/model.pt",
            True,
            True,
        )),
        "run_evidence_report rejected valid manifest source-bundle evidence",
        failures,
    )
    _assert(
        not _failed_checks(run_evidence_report._package_checks(
            package,
            "package",
            readiness,
            ["tiny", "small", "medium"],
            "max",
            "medium",
            0,
            240,
            240,
            6,
            "cuda",
            True,
            True,
            True,
            True,
            True,
            True,
        )),
        "run_evidence_report rejected valid package source-bundle evidence",
        failures,
    )
    reranker_checks, _reranker_summary = run_evidence_report._reranker_checks(
        _reranker_payload(),
        "reranker",
        ["tiny", "small", "medium"],
        240,
        True,
    )
    _assert(
        not _failed_checks(reranker_checks),
        "run_evidence_report rejected valid selected reranker score ordering",
        failures,
    )
    stale_reranker_checks, _stale_reranker_summary = run_evidence_report._reranker_checks(
        _reranker_payload(best_score=1.0, other_score=2.0),
        "reranker",
        ["tiny", "small", "medium"],
        240,
        True,
    )
    _assert(
        any(
            check["name"] == "selected reranker score ordering"
            for check in _failed_checks(stale_reranker_checks)
        ),
        "run_evidence_report accepted selected reranker below the best eligible score",
        failures,
    )
    _assert(
        not _infer_selected_checkpoint_failures(),
        "infer rejected a selected checkpoint whose metrics hash matches the live checkpoint file",
        failures,
    )
    _assert(
        any("checkpoint_sha256" in failure for failure in _infer_selected_checkpoint_failures(omit_checkpoint_sha=True)),
        "infer accepted selected reranker metrics without checkpoint_sha256",
        failures,
    )
    _assert(
        any("hash mismatch" in failure for failure in _infer_selected_checkpoint_failures(BAD_SHA)),
        "infer accepted a selected checkpoint whose live file hash differs from reranker metrics",
        failures,
    )
    selected_checkpoint = (Path.cwd() / "checkpoints" / "small" / "model.pt").resolve()
    chosen_checkpoint, choice_failures = _infer_checkpoint_choice(None, selected_checkpoint)
    _assert(
        not choice_failures and chosen_checkpoint == selected_checkpoint,
        "infer did not automatically use the selected reranker checkpoint when --checkpoint was omitted",
        failures,
    )
    _mismatch_checkpoint, mismatch_failures = _infer_checkpoint_choice(
        Path("checkpoints/tiny/model.pt"),
        selected_checkpoint,
    )
    _assert(
        any("does not match selected checkpoint" in failure for failure in mismatch_failures),
        "infer accepted an explicit checkpoint override that differs from the selected reranker checkpoint",
        failures,
    )
    _assert(
        not _report_selected_inference_failed_checks(),
        "run_evidence_report rejected matching selected inference checkpoint evidence",
        failures,
    )
    _assert(
        any(
            check["name"] == "inference selected checkpoint path"
            for check in _report_selected_inference_failed_checks(checkpoint_used="checkpoints/small/model.pt")
        ),
        "run_evidence_report accepted inference checkpoint path drift from selected reranker checkpoint",
        failures,
    )
    _assert(
        any(
            check["name"] == "inference selected checkpoint summary path"
            for check in _report_selected_inference_failed_checks(selected_checkpoint="checkpoints/small/model.pt")
        ),
        "run_evidence_report accepted inference selected_checkpoint summary drift",
        failures,
    )
    _assert(
        any(
            check["name"] == "inference selected checkpoint hash"
            for check in _report_selected_inference_failed_checks(checkpoint_sha=BAD_SHA)
        ),
        "run_evidence_report accepted inference checkpoint hash drift from selected reranker metrics",
        failures,
    )
    weak_package = json.loads(json.dumps(package))
    weak_package["require_selected_checkpoint"] = False
    weak_package["require_preflight_cuda"] = False
    weak_package["require_preflight_eval"] = False
    weak_package["require_generated_metadata"] = False
    weak_package["required_transformer_device"] = "cpu"
    weak_package["required_min_reranker_count"] = 12
    weak_package["required_min_completion_compare_count"] = 12
    weak_package["required_min_train_epochs"] = 1
    weak_package_failures = _failed_checks(run_evidence_report._package_checks(
        weak_package,
        "package",
        readiness,
        ["tiny", "small", "medium"],
        "max",
        "medium",
        0,
        240,
        240,
        6,
        "cuda",
        True,
        True,
        True,
        True,
        True,
        True,
    ))
    _assert(
        len(weak_package_failures) >= 8,
        "run_evidence_report accepted weak package manifest strict-mode flags",
        failures,
    )
    relaxed_package = json.loads(json.dumps(package))
    relaxed_package["require_selected_checkpoint"] = False
    relaxed_package["require_preflight_cuda"] = False
    relaxed_package["require_preflight_eval"] = False
    relaxed_package["require_generated_metadata"] = False
    relaxed_package["required_transformer_device"] = ""
    _assert(
        not _failed_checks(run_evidence_report._package_checks(
            relaxed_package,
            "package",
            readiness,
            ["tiny", "small", "medium"],
            "max",
            "medium",
            0,
            240,
            240,
            6,
            "",
            False,
            False,
            False,
            False,
            True,
            True,
        )),
        "run_evidence_report rejected relaxed package manifest mode flags",
        failures,
    )
    _assert(
        not _packaged_script_source_bundle_failures(),
        "final_audit rejected packaged scripts that match source-bundle manifest hashes",
        failures,
    )
    _assert(
        _packaged_script_source_bundle_failures(tamper_script=True),
        "final_audit accepted packaged script bytes that differ from source-bundle manifest hashes",
        failures,
    )
    _assert(
        _packaged_script_source_bundle_failures(omit_manifest_files=True),
        "final_audit accepted source-bundle readiness without manifest file hashes for scripts",
        failures,
    )
    _assert(
        not _packaged_source_snapshot_failures(),
        "final_audit rejected packaged Python source snapshots that match source-bundle manifest hashes",
        failures,
    )
    _assert(
        _packaged_source_snapshot_failures(tamper_source=True),
        "final_audit accepted packaged Python source snapshots that differ from source-bundle manifest hashes",
        failures,
    )
    _assert(
        _packaged_source_snapshot_failures(omit_source=True),
        "final_audit accepted missing packaged Python source snapshot evidence",
        failures,
    )
    _assert(
        not _report_source_identity_failed_checks(),
        "run_evidence_report rejected packaged source identity evidence matching the source bundle",
        failures,
    )
    _assert(
        _report_source_identity_failed_checks(tamper_source=True),
        "run_evidence_report accepted tampered packaged Python source snapshot evidence",
        failures,
    )
    _assert(
        _report_source_identity_failed_checks(omit_source=True),
        "run_evidence_report accepted missing packaged Python source snapshot evidence",
        failures,
    )
    _assert(
        _report_reads_package_manifest_from_zip(),
        "run_evidence_report did not read package_manifest.json directly from the ZIP",
        failures,
    )
    _assert(
        _package_verifier_reads_manifest_from_zip(),
        "verify_package/final_audit did not read package_manifest.json directly from the ZIP",
        failures,
    )
    _assert(
        not _zip_only_checkpoint_source_failures(),
        "final_audit rejected matching ZIP-only checkpoint source-bundle evidence",
        failures,
    )
    _assert(
        any("source_bundle_sha256" in failure for failure in _zip_only_checkpoint_source_failures(BAD_SHA)),
        "final_audit accepted ZIP-only checkpoint evidence with stale source-bundle SHA",
        failures,
    )
    _assert(
        not _report_final_audit_failed_checks(),
        "run_evidence_report rejected matching final_audit summary provenance",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(package_dir="artifacts/other_package"),
        "run_evidence_report accepted final_audit summary for a different package directory",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(artifacts_dir="artifacts/other_run"),
        "run_evidence_report accepted final_audit summary for a different artifacts directory",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(package_zip_sha256=BAD_SHA),
        "run_evidence_report accepted final_audit summary for a stale package ZIP hash",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(package_sidecar_sha256=BAD_SHA),
        "run_evidence_report accepted final_audit summary for a stale package sidecar hash",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(package_manifest_sha256=BAD_SHA),
        "run_evidence_report accepted final_audit summary for a stale package manifest hash",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(count=50000),
        "run_evidence_report accepted final_audit summary with stale generated-count thresholds",
        failures,
    )
    _assert(
        _report_final_audit_failed_checks(require_source_bundle_proof=False),
        "run_evidence_report accepted final_audit summary with stale source-bundle proof flag",
        failures,
    )

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["source_bundle"]["bundle_sha256"] = BAD_SHA
    _assert(
        package_submission._manifest_source_bundle_failures(tampered_manifest, readiness, "manifest"),
        "package_submission accepted a tampered manifest source-bundle hash",
        failures,
    )
    _assert(
        final_audit._manifest_source_bundle_failures(tampered_manifest, readiness, "manifest"),
        "final_audit accepted a tampered manifest source-bundle hash",
        failures,
    )
    _assert(
        _failed_checks(run_evidence_report._manifest_checks(
            tampered_manifest,
            "manifest",
            readiness,
            {"passed": True},
            "final_audit",
            "packaged_with_submissions",
            "max",
            "checkpoints/medium/model.pt",
            True,
            True,
        )),
        "run_evidence_report accepted a tampered manifest source-bundle hash",
        failures,
    )

    tampered_package = json.loads(json.dumps(package))
    tampered_package["source_bundle"]["bundle_sha256"] = BAD_SHA
    _assert(
        final_audit._package_source_bundle_failures(tampered_package, readiness),
        "final_audit accepted a tampered package source-bundle hash",
        failures,
    )
    _assert(
        _failed_checks(run_evidence_report._package_checks(
            tampered_package,
            "package",
            readiness,
            ["tiny", "small", "medium"],
            "max",
            "medium",
            0,
            240,
            240,
            6,
            "cuda",
            True,
            True,
            True,
            True,
            True,
            True,
        )),
        "run_evidence_report accepted a tampered package source-bundle hash",
        failures,
    )

    unverified_manifest = json.loads(json.dumps(manifest))
    unverified_manifest["source_bundle"]["verified"] = False
    _assert(
        package_submission._manifest_source_bundle_failures(unverified_manifest, readiness, "manifest"),
        "package_submission accepted unverified manifest source-bundle evidence",
        failures,
    )
    _assert(
        final_audit._manifest_source_bundle_failures(unverified_manifest, readiness, "manifest"),
        "final_audit accepted unverified manifest source-bundle evidence",
        failures,
    )

    unverified_package = json.loads(json.dumps(package))
    unverified_package["source_bundle"]["verified"] = False
    _assert(
        final_audit._package_source_bundle_failures(unverified_package, readiness),
        "final_audit accepted unverified package source-bundle evidence",
        failures,
    )

    valid_events = _event_text([
        "start",
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ])
    _assert(
        not _event_order_failed_checks(valid_events),
        "run_evidence_report rejected valid run-manifest event order",
        failures,
    )
    final_positions = final_audit._event_log_text_stage_positions(valid_events)
    _assert(
        final_positions["generation_prepared"]
        < final_positions["checkpoint_audited"]
        < final_positions["comparisons_complete"]
        < final_positions["packaged_with_submissions"],
        "final_audit did not parse valid run-manifest event order",
        failures,
    )
    package_positions = _package_stage_positions(valid_events)
    _assert(
        package_positions["generation_prepared"]
        < package_positions["checkpoint_audited"]
        < package_positions["comparisons_complete"]
        < package_positions["packaged_with_submissions"],
        "package_submission did not parse valid run-manifest event order",
        failures,
    )

    misordered_events = _event_text([
        "start",
        "checkpoint_audited",
        "generation_prepared",
        "comparisons_complete",
        "packaged_with_submissions",
    ])
    _assert(
        _event_order_failed_checks(misordered_events),
        "run_evidence_report accepted misordered run-manifest event stages",
        failures,
    )

    retry_events = _event_text([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ])
    _assert(
        not _event_order_failed_checks(retry_events),
        "run_evidence_report rejected valid retry run-manifest event order",
        failures,
    )

    stale_terminal_events = _event_text([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
        "generation_prepared",
    ])
    _assert(
        _event_order_failed_checks(stale_terminal_events),
        "run_evidence_report accepted stale terminal event order after retry",
        failures,
    )
    source_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ])
    _assert(
        not _event_source_failed_checks(source_events),
        "run_evidence_report rejected valid event-level source-bundle proof",
        failures,
    )
    missing_source_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ], include_source=False)
    _assert(
        _event_source_failed_checks(missing_source_events),
        "run_evidence_report accepted events without source-bundle proof",
        failures,
    )
    bad_source_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ], bundle_sha=BAD_SHA)
    _assert(
        _event_source_failed_checks(bad_source_events),
        "run_evidence_report accepted events with mismatched source-bundle hash",
        failures,
    )
    wrong_count_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ], count_per_family=50000)
    _assert(
        _event_source_failed_checks(wrong_count_events),
        "run_evidence_report accepted events with COUNT_PER_FAMILY not matching readiness",
        failures,
    )
    missing_eval_flag_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ], require_eval="0")
    _assert(
        _event_source_failed_checks(missing_eval_flag_events),
        "run_evidence_report accepted events without required eval mode",
        failures,
    )
    wrong_completion_checkpoint_events = _event_text_with_source([
        "generation_prepared",
        "checkpoint_audited",
        "comparisons_complete",
        "packaged_with_submissions",
    ], completion_checkpoint="checkpoints/tiny/model.pt")
    _assert(
        _event_source_failed_checks(wrong_completion_checkpoint_events),
        "run_evidence_report accepted events with wrong completion checkpoint",
        failures,
    )
    _assert(
        not _packaged_readiness_failures(True, True),
        "final_audit rejected valid packaged source-bundle readiness proof",
        failures,
    )
    _assert(
        any("required eval export" in failure for failure in _packaged_readiness_failures(True, True, False)),
        "final_audit accepted packaged readiness launch commands without required eval export",
        failures,
    )
    _assert(
        any(
            "split generation source-bundle export" in failure
            or "split training source-bundle export" in failure
            for failure in _packaged_readiness_failures(True, True, True, False)
        ),
        "final_audit accepted packaged readiness without split source-bundle exports",
        failures,
    )
    _assert(
        any("recorded full-pipeline launch command" in failure for failure in _packaged_readiness_failures(
            True,
            True,
            True,
            True,
            False,
        )),
        "final_audit accepted packaged readiness launch script without the exact recorded full-pipeline command",
        failures,
    )
    _assert(
        not _package_submission_launch_failures(True),
        "package_submission rejected valid launch commands with required eval export",
        failures,
    )
    _assert(
        any("required eval export" in failure for failure in _package_submission_launch_failures(False)),
        "package_submission accepted launch commands without required eval export",
        failures,
    )
    _assert(
        any(
            "recorded full-pipeline launch command" in failure
            for failure in _package_submission_launch_failures(True, False)
        ),
        "package_submission accepted launch script without exact recorded full-pipeline command",
        failures,
    )
    _assert(
        any("no-source-bundle proof flag" in failure for failure in _package_submission_launch_failures(True, True, False)),
        "package_submission accepted no-source readiness verification commands without explicit disable flags",
        failures,
    )
    _assert(
        not _package_submission_source_launch_failures(True),
        "package_submission rejected valid launch commands with split source-bundle exports",
        failures,
    )
    _assert(
        any("split generation source-bundle export" in failure for failure in _package_submission_source_launch_failures(False)),
        "package_submission accepted launch commands without split source-bundle exports",
        failures,
    )
    _assert(
        any(
            "recorded full-pipeline launch command" in failure
            or "recorded dependency-safe split launch command" in failure
            for failure in _package_submission_source_launch_failures(True, False)
        ),
        "package_submission accepted launch script without exact recorded readiness launch commands",
        failures,
    )
    _assert(
        not _handoff_launch_failures(True),
        "leonardo_handoff rejected valid launch commands with required eval export",
        failures,
    )
    _assert(
        any("eval export" in failure for failure in _handoff_launch_failures(False)),
        "leonardo_handoff accepted launch commands without required eval export",
        failures,
    )
    _assert(
        not _handoff_strict_readiness_failures(True, False),
        "leonardo_handoff rejected valid strict non-deferred readiness handoff command",
        failures,
    )
    _assert(
        _handoff_strict_readiness_failures(False, False),
        "leonardo_handoff accepted strict readiness handoff command without required eval",
        failures,
    )
    _assert(
        _handoff_strict_readiness_failures(True, True),
        "leonardo_handoff accepted strict readiness handoff command that still defers eval staging",
        failures,
    )
    _assert(
        not _handoff_checklist_missing_needles(),
        "leonardo_handoff checklist omitted required upload, launch, or verification commands",
        failures,
    )
    _assert(
        not _transfer_packet_failures(),
        "leonardo_handoff transfer packet omitted upload files, evidence, manifest, or checksum proof",
        failures,
    )
    _assert(
        not _return_packet_failures(),
        "leonardo_return_packet omitted returned package files, summaries, manifest, or checksum proof",
        failures,
    )
    _assert(
        not _readiness_report_failed_checks(True),
        "run_evidence_report rejected valid readiness launch commands with required eval export",
        failures,
    )
    _assert(
        _readiness_report_failed_checks(False),
        "run_evidence_report accepted readiness launch commands without required eval export",
        failures,
    )
    _assert(
        _readiness_report_failed_checks(True, False),
        "run_evidence_report accepted readiness launch commands without split source-bundle exports",
        failures,
    )
    _assert(
        _readiness_report_failed_checks(True, True, False),
        "run_evidence_report accepted launch script without exact recorded readiness launch commands",
        failures,
    )
    _assert(
        not _readiness_report_no_source_failed_checks(),
        "run_evidence_report rejected no-source readiness commands with explicit disable flags",
        failures,
    )
    _assert(
        _readiness_report_no_source_failed_checks(False),
        "run_evidence_report accepted no-source readiness commands without explicit disable flags",
        failures,
    )
    _assert(
        any("did not require source-bundle proof" in failure for failure in _packaged_readiness_failures(False, True)),
        "final_audit accepted packaged readiness without required source-bundle proof",
        failures,
    )
    _assert(
        not any(
            "did not require source-bundle proof" in failure
            for failure in _packaged_readiness_failures(False, False)
        ),
        "final_audit required source-bundle proof after it was explicitly disabled",
        failures,
    )
    _assert(
        any(
            "no-source-bundle proof flag" in failure
            for failure in _packaged_readiness_failures(False, False, include_no_source_proof_flag=False)
        ),
        "final_audit accepted no-source readiness verification commands without explicit disable flags",
        failures,
    )
    _assert(
        any("batch-size proof" in failure for failure in _packaged_readiness_failures(True, True, include_batch_proof=False)),
        "final_audit accepted readiness verification commands without batch-size proof",
        failures,
    )
    _assert(
        not _checkpoint_staleness_failures(),
        "checkpoint staleness gate rejected a matching train summary",
        failures,
    )
    stale_config = dict(validate_run.MODEL_CONFIGS["tiny"])
    stale_config["num_layers"] = 99
    _assert(
        any("config does not match" in failure for failure in _checkpoint_staleness_failures(stale_config)),
        "checkpoint staleness gate accepted a checkpoint with stale model config",
        failures,
    )
    _assert(
        any("batch_size" in failure for failure in _checkpoint_staleness_failures(batch_size=32)),
        "checkpoint staleness gate accepted a checkpoint with stale batch size",
        failures,
    )
    _assert(
        any("source_bundle_sha256" in failure for failure in _checkpoint_staleness_failures(source_bundle_sha256=BAD_SHA)),
        "checkpoint staleness gate accepted a checkpoint from a stale source bundle",
        failures,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "cases": {
            "valid_manifest_and_package": True,
            "tampered_manifest_hash_rejected": True,
            "tampered_package_hash_rejected": True,
            "unverified_manifest_rejected": True,
            "unverified_package_rejected": True,
            "valid_event_order_accepted": True,
            "misordered_event_order_rejected": True,
            "retry_event_order_accepted": True,
            "stale_terminal_event_order_rejected": True,
            "packaged_readiness_source_bundle_required": True,
            "packaged_readiness_source_bundle_disable_respected": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-test source-bundle proof gates with tampered evidence.")
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "source_bundle_proof_selftest.json")
    args = parser.parse_args()
    payload = run_selftest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.out}")
    if not payload["passed"]:
        print("Source-bundle proof self-test failed:")
        for failure in payload["failures"]:
            print(f"- {failure}")
        raise SystemExit(2)
    print("Source-bundle proof self-test passed")


if __name__ == "__main__":
    main()
