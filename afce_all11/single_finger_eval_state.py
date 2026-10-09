"""Prepare, resume and summarize the single seed-0 trainable GT-AUX evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TASKS = (
    "bimanual_assembly", "bimanual_hanoi", "bimanual_microwave_cook",
    "bimanual_photograph", "bimanual_unlock_ipad", "click_mouse",
    "fold_glasses", "hammer_nail", "pick_bucket", "pinch_tongs", "water_plant",
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_contract(root, effect_cache, decoder_init):
    complete = json.loads((root / "complete.json").read_text())
    latest = json.loads((root / "latest_complete.json").read_text())
    assert complete["complete"] is True
    assert complete["optimizer_updates"] == 60_000 and complete["checkpoint_index"] == 59_999
    assert latest["complete"] and latest["optimizer_updates"] == 60_000
    checkpoint = root / "59999"
    for name in ("params", "train_state", "assets"):
        assert (checkpoint / name).is_dir(), name
    metadata = json.loads((checkpoint / "assets/resume_metadata.json").read_text())
    assert metadata["optimizer_updates"] == metadata["next_global_batch"] == 60_000
    runtime_sha = file_sha(checkpoint / "assets/training_runtime.pkl")
    assert runtime_sha == (checkpoint / "assets/training_runtime.sha256").read_text().strip()
    contract = json.loads((root / "training_contract.json").read_text())
    model = contract["model"]
    assert contract["steps"] == 60_000 and contract["batch_size"] == 32 and contract["seed"] == 42
    assert model.get("observation_horizon", 1) == 1
    assert model["joint_objective"] == "c01_single_joint_gtaux_finger_fixed_v1"
    assert model["decoder_warmup_steps"] == 0 and model["alignment_weight"] == 0.25628781345139295
    assert model["finger_delta_weight"] == .05 and model["finger_delta_scale_floor"] == .1
    assert model["joint_decoder_lr_scale"] == .25
    assert model["alignment_start"] == 0 and model["alignment_ramp_steps"] == 10_000
    manifest = json.loads((effect_cache / "manifest.json").read_text())
    receipt = json.loads(decoder_init.with_suffix(".json").read_text())
    assert manifest["complete"] and receipt["codec_sha256"] == manifest["checkpoint_sha256"]
    return {
        "checkpoint": str(checkpoint.resolve()), "runtime_sha256": runtime_sha,
        "effect_checkpoint_sha256": manifest["checkpoint_sha256"],
        "decoder_init_sha256": file_sha(decoder_init), "seed": 0,
        "nodes": 2, "gpus": 8, "inference_decoder": "single jointly trained decoder",
        "episodes_per_task": 50, "tasks": list(TASKS), "num_denoise_steps": 10,
        "method": "C01 E + calibrated GT-AUX + fixed finger-delta + single jointly trained decoder",
    }


def progress(task_dir, task):
    path = task_dir / "progress.json"
    if not path.exists():
        if list(task_dir.glob("success_rate_*_50.txt")):
            raise ValueError("Result marker exists without progress: " + task)
        return {"completed": 0, "success": 0, "rng_offset": 0}
    record = json.loads(path.read_text())
    assert record["version"] == 1 and record["env_name"] == task and record["seed"] == 0
    # The single-frame DexJoCo evaluator predates this progress field. Its
    # checkpoint contract fixes the horizon at one; reject any explicit mismatch.
    assert record["episodes"] == 50 and record.get("observation_horizon", 1) == 1
    count = record["completed_episodes"]
    assert 0 <= count <= 50 and len(record["episode_results"]) == count
    assert sum(record["episode_results"]) == record["num_success"]
    assert (task_dir / record["state_file"]).is_file()
    markers = list(task_dir.glob("success_rate_*_50.txt"))
    if markers:
        assert count == 50 and len(markers) == 1
        assert markers[0].name == f"success_rate_{record['num_success']}_50.txt"
    return {"completed": count, "success": record["num_success"],
            "rng_offset": record["inference_requests"]}


def summarize(root):
    tasks = {task: progress(root / task, task) for task in TASKS}
    result = {
        "complete": all(row["completed"] == 50 and len(list((root/task).glob("success_rate_*_50.txt"))) == 1 for task,row in tasks.items()),
        "seed": 0, "episodes_per_task": 50, "tasks": tasks,
        "completed_episodes": sum(row["completed"] for row in tasks.values()),
        "success": sum(row["success"] for row in tasks.values()), "target_episodes": 550,
    }
    atomic_json(root / "status.json", result)
    if result["complete"]:
        result["success_rate"] = result["success"] / 550
        atomic_json(root / "complete.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "summary", "offset"), required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--effect-cache", type=Path)
    parser.add_argument("--decoder-init", type=Path)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    if args.mode == "prepare":
        value = checkpoint_contract(args.training_root, args.effect_cache, args.decoder_init)
        destination = args.eval_root / "evaluation_contract.json"
        if destination.exists():
            assert json.loads(destination.read_text()) == value
        atomic_json(destination, value)
        print("FINAL_CHECKPOINT_VERIFIED")
    elif args.mode == "offset":
        row = progress(args.eval_root / args.task, args.task)
        print("COMPLETE" if row["completed"] == 50 else row["rng_offset"])
    else:
        print(json.dumps(summarize(args.eval_root)), flush=True)


if __name__ == "__main__":
    main()
