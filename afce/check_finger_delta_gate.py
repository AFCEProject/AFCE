"""Audit the reviewed 8-GPU trainable-decoder GT-AUX resume gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from afce.check_multihost import compare_checkpoints, compare_runtime
from afce.prepare import atomic_json


def training_rows(root):
    return [
        row
        for line in (root / "exact_metrics.jsonl").read_text().splitlines()
        if (row := json.loads(line))["event"] == "train"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--decoder-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.root / "checkpoints/afce_official"
    continuous, interrupted = base / "continuous", base / "interrupted"
    parameters = compare_checkpoints(continuous / "3", interrupted / "3")
    runtime = compare_runtime(continuous / "3", interrupted / "3")
    left, right = training_rows(continuous), training_rows(interrupted)
    keys = (
        "step", "batch_indices", "batch_sha256", "loss", "effect_loss",
        "action_loss", "gtaux_loss", "finger_delta_loss", "finger_delta_loss_raw", "auxiliary_loss_raw", "ramp", "grad_norm", "param_norm",
        "decoder_train_enabled", "single_decoder_grad_norm",
        "bimanual_decoder_grad_norm", "single_samples", "bimanual_samples",
    )
    trace_equal = [{k: row[k] for k in keys} for row in left] == [
        {k: row[k] for k in keys} for row in right
    ]
    contract = json.loads((continuous / "training_contract.json").read_text())
    model = contract["model"]
    contract_ok = (
        contract["seed"] == 42
        and contract["batch_size"] == 32
        and contract["steps"] == 60_000
        and contract["optimizer"] == {
            "b1": 0.9, "b2": 0.95, "eps": 1e-8,
            "weight_decay": 1e-10, "clip_gradient_norm": 1.0,
        }
        and model["joint_objective"] == "finger_delta_joint_gtaux_fixed"
        and model["decoder_warmup_steps"] == 0
        and model["alignment_target_mode"] == "ground_truth"
        and model["alignment_weight"] == 0.25628781345139295
        and model["finger_delta_weight"] == .05
        and model["finger_delta_scale_floor"] == .1
        and model["joint_decoder_lr_scale"] == .25
        and model["alignment_start"] == 0
        and model["alignment_ramp_steps"] == 10_000
        and model["alignment_huber_delta"] == 1.0
    )
    finite = all(
        all(np.isfinite(row[name]) for name in (
            "loss", "effect_loss", "action_loss", "auxiliary_loss_raw", "grad_norm"
        ))
        for row in left
    )
    arithmetic = all(
        math.isclose(row["loss"], row["effect_loss"] + row["action_loss"], rel_tol=1e-6, abs_tol=1e-7)
        and math.isclose(row["action_loss"], row["ramp"] * (0.25628781345139295 * row["auxiliary_loss_raw"] + .05 * row["finger_delta_loss_raw"]), rel_tol=1e-6, abs_tol=1e-7)
        and row["decoder_train_enabled"] == 1
        for row in left
    )
    decoder_gradients = all(
        any(row[name] > 0 for row in left if row[samples] > 0 and row["ramp"] > 0)
        for name, samples in (
            ("single_decoder_grad_norm", "single_samples"),
            ("bimanual_decoder_grad_norm", "bimanual_samples"),
        )
    )
    decoder_audit = json.loads(args.decoder_audit.read_text())
    runtime_keys = (
        "updates_equal", "loader_state_equal", "jax_rng_equal",
        "partition_complete",
    )
    passed = (
        all(item["bitwise_equal"] for item in parameters.values())
        and all(runtime[key] for key in runtime_keys)
        and trace_equal
        and [row["step"] for row in left] == [1, 2, 3, 4]
        and finite
        and arithmetic
        and decoder_gradients
        and decoder_audit.get("passed") is True
        and contract_ok
        and runtime["all_host_rng_equal"]
    )
    report = {
        "complete": passed,
        "passed": passed,
        "strict_bitwise_passed": passed and runtime["all_host_rng_equal"],
        "host_rng_difference_is_blocking": True,
        "experiment": "finger-Δ single jointly trained decoder with fixed finger-delta",
        "continuous_vs_2_plus_2": parameters,
        "runtime": runtime,
        "trace_equal": trace_equal,
        "steps": [row["step"] for row in left],
        "finite": finite,
        "loss_arithmetic": arithmetic,
        "both_decoders_receive_gradients": decoder_gradients,
        "decoder_checkpoint_audit": decoder_audit,
        "contract_ok": contract_ok,
        "contract": contract,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise AssertionError("Trainable GT-AUX gate failed")


if __name__ == "__main__":
    main()
