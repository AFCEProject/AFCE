"""Serve a task-balanced DexJoCo π0.5 checkpoint built by the local launcher.

The eleven-task configs are assembled dynamically by
``dexjoco_multitask.py`` and therefore are not present in OpenPI's
static config registry.  This entry point rebuilds the matching config before
loading a flat or structured checkpoint for evaluation.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import socket
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro

from dexjoco_multitask import Args as TrainArgs
from dexjoco_multitask import build_config as build_balanced_config
from dexjoco_official_sampling import build_config as build_official_sampling_config
from openpi.models import model as model_module
from openpi.policies import policy as policy_module
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint_dir: pathlib.Path
    data_root: pathlib.Path
    init_params_path: pathlib.Path
    checkpoint_base_dir: pathlib.Path
    assets_base_dir: pathlib.Path
    exp_name: str
    port: int = 18_000
    structured_hand_state: bool = False
    sampling_balance: Literal["task", "proportional"] = "task"


def _flatten_paths(tree: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    leaves: dict[tuple[str, ...], Any] = {}
    for key, value in tree.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            leaves.update(_flatten_paths(value, path))
        else:
            leaves[path] = value
    return leaves


def _load_jax_model(config, checkpoint_dir: pathlib.Path):
    params = model_module.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    shaped_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    graphdef, state = nnx.split(shaped_model)
    expected = state.to_pure_dict()
    expected_leaves = _flatten_paths(expected)
    restored_leaves = _flatten_paths(params)

    allowed_missing: set[tuple[str, ...]] = set()
    if config.model.structured_hand_state:
        # NNX represents use_bias=False as a ``bias=None`` state leaf, whereas
        # Orbax omits empty leaves. These are the only expected omissions.
        allowed_missing = {
            ("structured_hand_encoder", "joint_message_proj", "bias"),
            ("structured_hand_encoder", "palm_context_proj", "bias"),
        }
    missing = set(expected_leaves) - set(restored_leaves)
    unexpected_missing = missing - allowed_missing
    if unexpected_missing:
        formatted = ["/".join(path) for path in sorted(unexpected_missing)]
        raise ValueError(f"Checkpoint is missing required model leaves: {formatted}")

    for path, value in restored_leaves.items():
        if path not in expected_leaves:
            continue
        expected_value = expected_leaves[path]
        if hasattr(expected_value, "shape") and expected_value.shape != value.shape:
            raise ValueError(
                f"Checkpoint shape mismatch at {'/'.join(path)}: "
                f"expected {expected_value.shape}, got {value.shape}"
            )
        cursor = expected
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value

    state.replace_by_pure_dict(expected)
    return nnx.merge(graphdef, state)


def _load_policy(config, checkpoint_dir: pathlib.Path) -> policy_module.Policy:
    model = _load_jax_model(config, checkpoint_dir)

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load checkpoint normalization statistics.")
    norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
    )


def main(args: Args) -> None:
    build_config = (
        build_official_sampling_config
        if args.sampling_balance == "proportional"
        else build_balanced_config
    )
    config, _ = build_config(
        TrainArgs(
            operation="inspect",
            data_root=args.data_root,
            init_params_path=args.init_params_path,
            exp_name=args.exp_name,
            checkpoint_base_dir=args.checkpoint_base_dir,
            assets_base_dir=args.assets_base_dir,
            structured_hand_state=args.structured_hand_state,
        )
    )
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    for required in (
        checkpoint_dir / "params",
        checkpoint_dir / "assets",
    ):
        if not required.is_dir():
            raise FileNotFoundError(f"Incomplete checkpoint: missing {required}")

    policy = _load_policy(config, checkpoint_dir)
    hostname = socket.gethostname()
    logging.info(
        "Serving %s checkpoint %s on %s:%d",
        "structured" if args.structured_hand_state else "flat",
        checkpoint_dir,
        hostname,
        args.port,
    )
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
