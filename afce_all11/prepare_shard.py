"""Prepare one remaining task and atomically publish finished episodes.

The original all-task coordinator owns the final manifest. Existing destinations
are never overwritten, so an episode already being processed stays with it.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from afce_all11.prepare import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--dino',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--staging',type=Path,required=True)
    p.add_argument('--task',required=True);args=p.parse_args()
    cmd=[sys.executable,'-u','-m','afce_all11.prepare','--data',str(args.data),'--dino',str(args.dino),
         '--output',str(args.staging),'--task',args.task]
    child=subprocess.Popen(cmd)
    published=0
    def publish():
        nonlocal published
        for ready in sorted((args.staging/args.task).glob('episode_*/ready.json')):
            source=ready.parent;target=args.destination/args.task/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try: source.rename(target)
                except FileExistsError: continue
                published+=1
                print(json.dumps({'event':'published','task':args.task,'episode':target.name,'published':published}),flush=True)
    while child.poll() is None:
        publish();time.sleep(1)
    publish()
    atomic_json(args.staging/'shard_done.json',{'returncode':child.returncode,'published':published,'task':args.task})
    if child.returncode: raise SystemExit(child.returncode)

if __name__=='__main__':main()
