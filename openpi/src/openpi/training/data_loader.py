from bisect import bisect_right
from collections.abc import Iterator, Sequence
import itertools
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RenamedKeyDataset(Dataset[dict]):
    """Lazily rename one top-level sample key while retaining the original value object."""

    def __init__(self, dataset: Dataset[dict], source_key: str, target_key: str):
        if not source_key or not target_key:
            raise ValueError("Dataset key names must be non-empty.")
        self._dataset = dataset
        self._source_key = source_key
        self._target_key = target_key

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._dataset[index]
        if self._source_key == self._target_key:
            return sample
        if self._source_key not in sample:
            raise KeyError(f"Source dataset sample does not contain {self._source_key!r}.")
        if self._target_key in sample:
            raise KeyError(f"Cannot rename {self._source_key!r} to existing sample key {self._target_key!r}.")

        renamed = sample.copy()
        renamed[self._target_key] = renamed.pop(self._source_key)
        return renamed

    def __len__(self) -> int:
        return len(self._dataset)


class CanonicalizedDexJoCoDataset(Dataset[dict]):
    """Lazily standardize one DexJoCo task to the official multi-task schema."""

    def __init__(
        self,
        dataset: Dataset[dict],
        source: _config.LeRobotDatasetSource,
        *,
        target_state_dim: int,
        target_action_dim: int,
        structured_hand_state: bool = False,
    ):
        if not source.wrist_left_image_key or not source.wrist_right_image_key:
            raise ValueError("DexJoCo canonicalization requires both wrist image keys.")
        self._dataset = dataset
        self._image_key_map = {
            _config.MULTI_SOURCE_BASE_IMAGE_KEY: source.base_image_key,
            _config.MULTI_SOURCE_LEFT_WRIST_IMAGE_KEY: source.wrist_left_image_key,
            _config.MULTI_SOURCE_RIGHT_WRIST_IMAGE_KEY: source.wrist_right_image_key,
        }
        self._target_state_dim = target_state_dim
        self._target_action_dim = target_action_dim
        self._structured_hand_state = structured_hand_state

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._dataset[index]
        canonical = sample.copy()
        for target_key, source_key in self._image_key_map.items():
            if source_key not in sample:
                raise KeyError(f"Source dataset sample does not contain {source_key!r}.")
            canonical[target_key] = sample[source_key]

        if "observation.state" not in sample or "action" not in sample:
            raise KeyError("DexJoCo sample must contain 'observation.state' and 'action'.")
        state = np.asarray(sample["observation.state"])
        if self._structured_hand_state:
            if self._target_state_dim != 46:
                raise ValueError(
                    "Structured DexJoCo hand state requires the canonical 46-D target layout, "
                    f"got {self._target_state_dim}."
                )
            if state.shape[-1] == 23:
                # Single-arm input: [right_tcp7, right_joints16]. Put values in
                # the same semantic slots used by bimanual observations.
                canonical["observation.state"] = np.concatenate(
                    [
                        state[..., :7],
                        np.zeros((*state.shape[:-1], 7), dtype=state.dtype),
                        state[..., 7:23],
                        np.zeros((*state.shape[:-1], 16), dtype=state.dtype),
                    ],
                    axis=-1,
                )
                canonical["observation.hand_presence"] = np.asarray([True, False], dtype=np.bool_)
            elif state.shape[-1] == 46:
                canonical["observation.state"] = state
                canonical["observation.hand_presence"] = np.asarray([True, True], dtype=np.bool_)
            else:
                raise ValueError(
                    "Structured DexJoCo hand state expects a 23-D single-arm or 46-D bimanual state, "
                    f"got shape {state.shape}."
                )
        else:
            canonical["observation.state"] = _transforms.pad_to_dim(
                state, self._target_state_dim, axis=-1
            )
        canonical["action"] = _transforms.pad_to_dim(
            np.asarray(sample["action"]), self._target_action_dim, axis=-1
        )
        return canonical

    def __len__(self) -> int:
        return len(self._dataset)


class MultiSourceDataset(Dataset[T_co]):
    """A zero-copy index view over multiple independently stored datasets."""

    def __init__(self, datasets: Sequence[Dataset[T_co]], *, balance: _config.DatasetBalance):
        if not datasets:
            raise ValueError("MultiSourceDataset requires at least one dataset.")
        if balance not in ("proportional", "task"):
            raise ValueError(f"Unsupported dataset balance mode: {balance!r}")

        self._datasets = tuple(datasets)
        self._lengths = tuple(len(dataset) for dataset in self._datasets)
        if any(length <= 0 for length in self._lengths):
            raise ValueError(f"MultiSourceDataset does not support empty sources: {self._lengths}")
        self._balance = balance
        self._cumulative_lengths = tuple(itertools.accumulate(self._lengths))
        self._max_source_length = max(self._lengths)
        self._length = (
            self._max_source_length * len(self._datasets) if balance == "task" else self._cumulative_lengths[-1]
        )

    def resolve_index(self, index: SupportsIndex) -> tuple[int, int]:
        """Resolve a virtual index to ``(source_index, source_local_index)``."""
        resolved = index.__index__()
        if resolved < 0:
            resolved += self._length
        if resolved < 0 or resolved >= self._length:
            raise IndexError(f"Dataset index {resolved} is out of range for length {self._length}.")

        if self._balance == "task":
            source_index = resolved % len(self._datasets)
            balance_round = resolved // len(self._datasets)
            source_length = self._lengths[source_index]
            return source_index, balance_round * source_length // self._max_source_length

        source_index = bisect_right(self._cumulative_lengths, resolved)
        previous_end = 0 if source_index == 0 else self._cumulative_lengths[source_index - 1]
        return source_index, resolved - previous_end

    def __getitem__(self, index: SupportsIndex) -> T_co:
        source_index, local_index = self.resolve_index(index)
        return self._datasets[source_index][local_index]

    def __len__(self) -> int:
        return self._length


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _create_lerobot_dataset(
    *,
    repo_id: str,
    root,
    action_sequence_keys: Sequence[str],
    action_horizon: int,
    prompt_from_task: bool,
    video_backend: str | None,
) -> Dataset[dict]:
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    video_backend_kwargs = {} if video_backend is None else {"video_backend": video_backend}
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_sequence_keys},
        **video_backend_kwargs,
    )

    if prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if data_config.sources:
        if data_config.root is not None:
            raise ValueError("DataConfig.root must be None when DataConfig.sources is set.")
        datasets = []
        for source in data_config.sources:
            dataset = _create_lerobot_dataset(
                repo_id=source.repo_id,
                root=source.root,
                action_sequence_keys=data_config.action_sequence_keys,
                action_horizon=action_horizon,
                prompt_from_task=data_config.prompt_from_task,
                video_backend=data_config.video_backend,
            )
            if data_config.afce_cache_root is not None:
                from pathlib import Path
                from afce_all11.pi_bridge import EffectCacheDataset
                cache_dataset = EffectCacheDataset
                if data_config.afce_joint_decoder:
                    from afce_all11.pi_bridge_joint import JointEffectCacheDataset
                    cache_dataset = JointEffectCacheDataset
                dataset = cache_dataset(dataset, Path(data_config.afce_cache_root) / Path(source.root).name)
            if data_config.source_target_state_dim is not None or data_config.source_target_action_dim is not None:
                if data_config.source_target_state_dim is None or data_config.source_target_action_dim is None:
                    raise ValueError("Both heterogeneous source target dimensions must be set together.")
                dataset = CanonicalizedDexJoCoDataset(
                    dataset,
                    source,
                    target_state_dim=data_config.source_target_state_dim,
                    target_action_dim=data_config.source_target_action_dim,
                    structured_hand_state=data_config.structured_hand_state,
                )
            else:
                dataset = RenamedKeyDataset(
                    dataset,
                    source_key=source.base_image_key,
                    target_key=_config.MULTI_SOURCE_BASE_IMAGE_KEY,
                )
            datasets.append(dataset)
        return MultiSourceDataset(datasets, balance=data_config.balance)

    if data_config.balance != "proportional":
        raise ValueError("Task balancing requires DataConfig.sources.")
    dataset = _create_lerobot_dataset(
        repo_id=repo_id,
        root=data_config.root,
        action_sequence_keys=data_config.action_sequence_keys,
        action_horizon=action_horizon,
        prompt_from_task=data_config.prompt_from_task,
        video_backend=data_config.video_backend,
    )
    return _maybe_wrap_effect_cache(dataset, data_config)


def _maybe_wrap_effect_cache(dataset: Dataset, data_config: _config.DataConfig) -> Dataset:
    cache_dir = getattr(data_config, "effect_cache_dir", None)
    if cache_dir is None:
        return dataset
    import sys
    from pathlib import Path

    dexjoco_root = Path(__file__).resolve().parents[4]
    if str(dexjoco_root) not in sys.path:
        sys.path.insert(0, str(dexjoco_root))
    from effect_vla.data.effect_cache_dataset import EffectCacheDataset, EffectCacheStore

    store = EffectCacheStore(Path(cache_dir).parent, Path(cache_dir).name)
    logging.info(f"Wrapping dataset with Effect cache at {cache_dir}")
    return EffectCacheDataset(dataset, store, required=True)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
