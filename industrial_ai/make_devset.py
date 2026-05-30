from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import join_steps, write_rows
from .official import load_generator
from .paths import DEFAULT_DATA_DIR, PROJECT_ROOT


RULES = [
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
]


def _insert_after_prefix(steps: list[str], step: str) -> list[str]:
    out = list(steps)
    out.insert(min(3, len(out)), step)
    return out


def _move_before(steps: list[str], moving: str, anchor: str) -> list[str] | None:
    if moving not in steps or anchor not in steps:
        return None
    out = list(steps)
    out.pop(out.index(moving))
    out.insert(out.index(anchor), moving)
    return out


def inject_violation(steps: list[str], rule: str, rng: random.Random) -> list[str]:
    if rule == "RULE_DEP_NO_CLEAN":
        return _insert_after_prefix(steps, "DEPOSIT METAL 1")
    if rule == "RULE_METAL_ETCH_NO_LITHO":
        return _insert_after_prefix(steps, "METAL ETCH")
    if rule == "RULE_ETCH_NO_MASK":
        return _insert_after_prefix(steps, "OXIDE ETCH")
    if rule == "RULE_LITHO_LEVEL_SKIP":
        out = list(steps)
        for idx, step in enumerate(out):
            if step == "ALIGN MASK LEVEL 2":
                out[idx] = "ALIGN MASK LEVEL 4"
                return out
        return _insert_after_prefix(out, "ALIGN MASK LEVEL 3")
    if rule == "RULE_IMPLANT_NO_MASK":
        return _insert_after_prefix(steps, "IMPLANT WELL")
    if rule == "RULE_CMP_NO_DEP":
        return _insert_after_prefix(steps, "CMP METAL")
    if rule == "RULE_PAD_OPEN_BEFORE_DEP":
        moved = _move_before(steps, "OPEN PAD WINDOW", "DEPOSIT PASSIVATION")
        return moved or _insert_after_prefix(steps, "OPEN PAD WINDOW")
    if rule == "RULE_TEST_BEFORE_PASSIVATION":
        for test_step in ("PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST", "LEAKAGE TEST"):
            moved = _move_before(steps, test_step, "CURE PASSIVATION")
            if moved:
                return moved
        return _insert_after_prefix(steps, "LEAKAGE TEST")
    if rule == "RULE_SHIP_BEFORE_TEST":
        moved = _move_before(steps, "SHIP LOT", "WAFER SORT TEST")
        return moved or _insert_after_prefix(steps, "SHIP LOT")
    if rule == "RULE_BACKSIDE_BEFORE_PASSIVATION":
        moved = _move_before(steps, "DEPOSIT BACKSIDE METAL", "CURE PASSIVATION")
        return moved or _insert_after_prefix(steps, "DEPOSIT BACKSIDE METAL")
    return _insert_after_prefix(steps, "SHIP LOT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic local dev eval files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "dev")
    parser.add_argument("--valid-per-family", type=int, default=40)
    parser.add_argument("--anomaly-valid-per-family", type=int, default=40)
    parser.add_argument("--anomaly-invalid-per-family", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generator = load_generator(args.data_dir)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    valid_rows: list[dict[str, object]] = []
    next_truth: list[dict[str, object]] = []
    completion_truth: list[dict[str, object]] = []

    for family in ("mosfet", "igbt", "ic"):
        sequences = generator.generate_dataset(family, args.valid_per_family, seed=args.seed + len(family), validate=True)
        for seq_idx, steps in enumerate(sequences):
            family_label = family.upper()
            for frac in (0.6, 0.8):
                cut = max(1, min(len(steps) - 1, int(len(steps) * frac)))
                example_id = f"{family}_{seq_idx:04d}_{int(frac * 100)}"
                valid_rows.append({
                    "EXAMPLE_ID": example_id,
                    "FAMILY": family_label,
                    "COMPLETION_FRACTION": frac,
                    "PARTIAL_SEQUENCE": join_steps(steps[:cut]),
                })
                next_truth.append({"EXAMPLE_ID": example_id, "NEXT_STEP": steps[cut]})
                completion_truth.append({"EXAMPLE_ID": example_id, "REMAINING_SEQUENCE": join_steps(steps[cut:])})

    anomaly_rows: list[dict[str, object]] = []
    anomaly_truth: list[dict[str, object]] = []
    for family in ("mosfet", "igbt", "ic"):
        family_label = family.upper()
        valid_sequences = generator.generate_dataset(
            family,
            args.anomaly_valid_per_family,
            seed=args.seed + 101 + len(family),
            validate=True,
        )
        for seq_idx, steps in enumerate(valid_sequences):
            example_id = f"{family}_valid_{seq_idx:04d}"
            anomaly_rows.append({"EXAMPLE_ID": example_id, "FAMILY": family_label, "SEQUENCE": join_steps(steps)})
            anomaly_truth.append({"EXAMPLE_ID": example_id, "IS_VALID": 1, "RULE": ""})

        invalid_bases = generator.generate_dataset(
            family,
            args.anomaly_invalid_per_family,
            seed=args.seed + 202 + len(family),
            validate=True,
        )
        for seq_idx, steps in enumerate(invalid_bases):
            requested_rule = RULES[seq_idx % len(RULES)]
            mutated = inject_violation(list(steps), requested_rule, rng)
            violations = generator.validate_sequence(mutated)
            actual_rule = violations[0].rule if violations else requested_rule
            example_id = f"{family}_invalid_{seq_idx:04d}"
            anomaly_rows.append({"EXAMPLE_ID": example_id, "FAMILY": family_label, "SEQUENCE": join_steps(mutated)})
            anomaly_truth.append({"EXAMPLE_ID": example_id, "IS_VALID": 0, "RULE": actual_rule})

    rng.shuffle(anomaly_rows)
    write_rows(args.out_dir / "eval_input_valid.csv", ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"], valid_rows)
    write_rows(args.out_dir / "nextstep_truth.csv", ["EXAMPLE_ID", "NEXT_STEP"], next_truth)
    write_rows(args.out_dir / "completion_truth.csv", ["EXAMPLE_ID", "REMAINING_SEQUENCE"], completion_truth)
    write_rows(args.out_dir / "eval_input_anomaly.csv", ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], anomaly_rows)
    write_rows(args.out_dir / "anomaly_truth.csv", ["EXAMPLE_ID", "IS_VALID", "RULE"], anomaly_truth)

    print(f"Wrote dev files to {args.out_dir}")


if __name__ == "__main__":
    main()

