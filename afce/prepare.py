"""Timestamp-checked low-dimensional data and sparse frozen-DINO evidence.

No interpolation of visual truth. Missing robot segmentation is reported, never
represented as a successfully measured all-background mask. Safe to resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

TASKS = ('click_mouse', 'fold_glasses', 'hammer_nail', 'pick_bucket', 'pinch_tongs',
         'water_plant', 'bimanual_assembly', 'bimanual_hanoi',
         'bimanual_microwave_cook', 'bimanual_photograph', 'bimanual_unlock_ipad')


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temp.replace(path)


def camera(task):
    return ('observation.images.ego' if task.startswith('bimanual_') else
            'observation.images.ego_right' if task == 'click_mouse' else 'observation.images.front')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dino', type=Path, required=True)
    p.add_argument('--reuse', type=Path, action='append', default=[])
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--task', action='append')
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.stride < 1:
        raise ValueError('stride must be positive')
    import pandas as pd
    import av
    import torch
    from effect_vla.effect.dino_extractor import FrozenDINOv3
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no silent CPU fallback')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else [str(x) for x in v] if k == 'reuse' else v
              for k, v in vars(args).items()}
    config_path = args.output / 'prepare_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Resume configuration differs; use a different output directory')
    atomic_json(config_path, config)
    manifest = {'schema': 'afce30_sparse_v1', 'tasks': {}, 'complete': False,
                'missing_robot_masks': [], 'stride': args.stride}
    backbone = None
    started = time.monotonic()
    for task in args.task or TASKS:
        root, cam = args.data / task, camera(task)
        info = json.loads((root/'meta/info.json').read_text())
        if info['total_episodes'] != 100 or info['fps'] != 30:
            raise ValueError(f'Unexpected official dataset metadata for {task}')
        rows = pd.concat([pd.read_parquet(f) for f in sorted((root/'meta/episodes').rglob('*.parquet'))])
        rows = rows.sort_values('episode_index').to_dict('records')
        low = pd.concat([pd.read_parquet(f, columns=['episode_index','frame_index','timestamp',
                            'observation.state','action']) for f in sorted((root/'data').rglob('*.parquet'))])
        rows = rows[:args.max_episodes] if args.max_episodes else rows
        manifest['tasks'][task] = []
        video_handle, current_video, cursor = None, None, 0
        for row in rows:
            ep, length = int(row['episode_index']), int(row['length'])
            dest = args.output/task/f'episode_{ep:06d}'
            if (dest/'ready.json').exists():
                rec = json.loads((dest/'ready.json').read_text())
                for file in ('actions.npy','states.npy','timestamps.npy','frames.npy','features.npy'):
                    if not (dest/file).exists():
                        raise RuntimeError(f'Incomplete committed episode: {dest}')
                manifest['tasks'][task].append(rec)
                if rec['mask_source'] == 'unavailable':
                    manifest['missing_robot_masks'].append([task,ep])
                continue
            dest.mkdir(parents=True, exist_ok=True)
            sample = low[low.episode_index == ep].sort_values('frame_index')
            frames = sample.frame_index.to_numpy(dtype=np.int64)
            ts = sample.timestamp.to_numpy(dtype=np.float64)
            if len(sample) != length or not np.array_equal(frames,np.arange(length)):
                raise ValueError(f'Frame index mismatch: {task}/{ep}')
            if not np.allclose(ts, np.arange(length)/30., atol=2e-5, rtol=0):
                raise ValueError(f'Recorded timestamp mismatch: {task}/{ep}')
            ad, sd = (44,46) if task.startswith('bimanual_') else (22,23)
            actions, states = np.stack(sample.action), np.stack(sample['observation.state'])
            if actions.shape != (length,ad) or states.shape != (length,sd):
                raise ValueError(f'Action/state schema mismatch: {task}/{ep}')
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError(f'Nonfinite low-dimensional data: {task}/{ep}')
            selected = np.unique(np.r_[np.arange(0,length,args.stride),length-1]).astype(np.int64)
            prefix = f'videos/{cam}/'
            video = root/'videos'/cam/f"chunk-{int(row[prefix+'chunk_index']):03d}"/f"file-{int(row[prefix+'file_index']):03d}.mp4"
            start = round(float(row[prefix+'from_timestamp'])*30)
            end = round(float(row[prefix+'to_timestamp'])*30)
            if end-start != length:
                raise ValueError(f'Episode video offset mismatch: {task}/{ep}')
            # Decode only selected genuine frames. Absolute PTS checks catch
            # both wrong episode offsets and dropped/duplicated video frames.
            if current_video != video:
                if video_handle is not None:
                    video_handle.close()
                video_handle = av.open(str(video))
                stream = video_handle.streams.video[0]
                stream.thread_type = 'AUTO'
                stream.codec_context.thread_count = 4
                if abs(float(stream.average_rate)-30.) > .01:
                    raise ValueError(f'Unexpected video FPS: {video}')
                iterator, cursor, current_video = video_handle.decode(video=0), 0, video
            if cursor > start:
                raise ValueError('Overlapping episode video offsets')
            images = []
            wanted = set((selected+start).tolist())
            while cursor < end:
                frame = next(iterator)
                if frame.pts is None or abs(float(frame.pts*frame.time_base)*30-cursor) > .1:
                    raise ValueError(f'Video PTS mismatch: {video}, frame {cursor}')
                if cursor in wanted:
                    images.append(frame.to_ndarray(format='rgb24'))
                cursor += 1
            if len(images) != len(selected):
                raise ValueError('Sparse RGB evidence is incomplete')
            if backbone is None:
                backbone = FrozenDINOv3(args.dino,device=args.device)
            # Recompute sparse descriptors from verified RGB even when legacy
            # caches are available; legacy caches contain no timestamp provenance.
            features = backbone.encode_numpy(np.stack(images), batch_size=32).astype(np.float16)
            if not np.isfinite(features).all() or features.shape != (len(selected),196,768):
                raise ValueError('DINO shape or numerical failure')
            mask_source = 'unavailable'
            for reuse in args.reuse:
                maskfile = reuse/task/f'episode_{ep:06d}'/'robot_mask.npz'
                if maskfile.exists():
                    with np.load(maskfile,allow_pickle=False) as z:
                        mask = z['mask']
                        if mask.shape == (length,196) and np.isfinite(mask).all() and np.any(mask > 0):
                            np.save(dest/'robot_mask.npy',mask[selected].astype(np.float32))
                            mask_source = str(maskfile)
                            break
            for name,value in [('actions',actions.astype(np.float32)),('states',states.astype(np.float32)),
                               ('timestamps',ts),('frames',selected),('features',features)]:
                temp = dest/f'{name}.tmp.npy'
                np.save(temp,value)
                temp.replace(dest/f'{name}.npy')
            rec = {'episode':ep,'frames':length,'anchors':len(selected),'action_dim':ad,
                   'mask_source':mask_source,'camera':cam,'video':str(video),
                   'video_start_frame':start,'timestamp_max_error':float(np.max(np.abs(ts-np.arange(length)/30))),
                   'feature_sha256':hashlib.sha256(features.tobytes()).hexdigest()}
            atomic_json(dest/'ready.json',rec)
            manifest['tasks'][task].append(rec)
            if mask_source == 'unavailable':
                manifest['missing_robot_masks'].append([task,ep])
            atomic_json(args.output/'manifest.json',manifest)
            print(json.dumps({'event':'prepared','task':task,'episode':ep,'frames':length,
                              'anchors':len(selected),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
        if video_handle is not None:
            video_handle.close()
    manifest['complete'] = set(manifest['tasks']) == set(TASKS) and all(len(x)==100 for x in manifest['tasks'].values())
    manifest['total_frames'] = sum(r['frames'] for rows in manifest['tasks'].values() for r in rows)
    atomic_json(args.output/'manifest.json',manifest)
    print(json.dumps({'event':'prepare_done','complete':manifest['complete'],'frames':manifest['total_frames']}),flush=True)


if __name__ == '__main__':
    main()
