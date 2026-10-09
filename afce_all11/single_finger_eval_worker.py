"""Eight GPU workers use per-task process locks and episode-boundary resume."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from afce_all11.single_finger_eval_state import progress, atomic_json

CODE=Path(os.environ.get('AFCE_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
RUNTIME=Path(os.environ.get('AFCE_RUNTIME', str(CODE/'runtime'))).resolve()
QUERY=Path(os.environ.get('AFCE_QUERY_ROOT', str(RUNTIME/'query_c01'))).resolve()
EXPERIMENT=Path(os.environ.get('AFCE_EXPERIMENT_ROOT', str(RUNTIME/'query_c01_single_finger'))).resolve()
EVAL=Path(os.environ.get('AFCE_EVAL_ROOT', str(RUNTIME/'evaluations'/'c01_single_finger_seed0'))).resolve()
NAME='c01_single_finger_seed42_60000'
CHECKPOINT=EXPERIMENT/'pi'/'checkpoints'/'afce_all11_official'/NAME/'59999'
BASE=Path(os.environ.get(
    'AFCE_PI05_BASE_PARAMS',
    str(CODE/'checkpoints'/'pi05_base_action_dim_44'/'params'),
)).resolve()
# First eight: five long tasks and three historically short tasks.
# Last three are claimed by whichever GPU actually becomes free first.
TASK_ORDER=('bimanual_assembly','bimanual_hanoi','pinch_tongs','fold_glasses',
            'bimanual_microwave_cook','hammer_nail','bimanual_unlock_ipad','pick_bucket',
            'bimanual_photograph','click_mouse','water_plant')
HISTORICAL_REQUESTS=dict(zip(TASK_ORDER,(10431,8707,6796,5518,5360,1885,2847,3453,5107,5072,4144)))


def stop_process(server):
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait(timeout=10)


def run_task(task, deadline):
    worker=int(os.environ['SLURM_PROCID']); gpu=int(os.environ['SLURM_LOCALID'])
    output=EVAL/task; output.mkdir(parents=True,exist_ok=True)
    prior=progress(output,task); offset=prior['rng_offset']
    env=dict(os.environ, MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
             XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.65',
             TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    port=24500+gpu; job=os.environ['SLURM_JOB_ID']
    print(json.dumps(dict(event='eval_start',task=task,worker=worker,host=socket.gethostname(),gpu=gpu,
                          offset=offset,completed=prior['completed'],utc=time.time())),flush=True)
    cmd=[sys.executable,'-u','-m','afce_all11.serve_single_finger_pi',
         '--checkpoint',str(CHECKPOINT),'--data',str(RUNTIME/'datasets_hf/dexjoco_lerobot_datasets'),
         '--base',str(BASE),'--output',str(EXPERIMENT/'pi'),'--name',NAME,'--task',task,
         '--effect-cache',str(QUERY/'effect_cache'),'--joint-decoder-init',str(QUERY/'joint_decoder_init.npz'),
         '--single-finger','--rng-offset',str(offset),'--port',str(port)]
    with (output/f'server.{job}.log').open('a') as log:
        server=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=CODE/'openpi')
        try:
            ready=False
            started=time.time()
            while time.time()<min(started+480,deadline-30):
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited {server.returncode}; see {log.name}')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=1): pass
                    ready=True;break
                except OSError:
                    time.sleep(2)
            if not ready:
                if time.time() >= deadline-30:
                    return False
                raise TimeoutError('Policy server readiness timed out: '+log.name)
            with (output/f'client.{job}.log').open('a') as clientlog:
                subprocess.run([sys.executable,'-u','-m','dexjoco_openpi_client.cli.evaluate',
                    '--config',str(CODE/'configs/rand_obj'/f'{task}.yaml'),'--seed','0',
                    '--host','127.0.0.1','--port',str(port),'--output',str(output),'--episodes','50',
                    '--pad-state-dim46','--resume','--expected-rng-offset',str(offset),
                    '--deadline-unix',str(deadline)],check=True,env=env,cwd=CODE/'openpi',
                    stdout=clientlog,stderr=subprocess.STDOUT)
        finally:
            stop_process(server)
    row=progress(output,task)
    print(json.dumps(dict(event='eval_saved',task=task,worker=worker,utc=time.time(),**row)),flush=True)
    return row['completed']==50


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix',type=int,required=True)
    args=parser.parse_args()
    EVAL.mkdir(parents=True,exist_ok=True)
    while time.time()+180<args.deadline_unix:
        selected=None
        for task in TASK_ORDER:
            folder=EVAL/task;folder.mkdir(parents=True,exist_ok=True)
            lock=(folder/'worker.lock').open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close();continue
            row=progress(folder,task)
            if row['completed']==50 and list(folder.glob('success_rate_*_50.txt')):
                lock.close();continue
            selected=(task,lock);break
        if selected is None:
            return 0
        task,lock=selected
        try:
            if not run_task(task,args.deadline_unix):
                return 0  # Parent returns 75 after collecting valid partial results.
        except Exception as exc:
            atomic_json(EVAL/task/f'failure.{os.environ["SLURM_JOB_ID"]}.json',
                dict(task=task,utc=time.time(),error=repr(exc),job=os.environ['SLURM_JOB_ID']))
            raise
        finally:
            lock.close()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
