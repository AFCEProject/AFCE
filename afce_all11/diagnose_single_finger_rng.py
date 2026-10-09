import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from afce_all11.check_multihost import equal_nested

root = Path(
    os.environ.get(
        "AFCE_RESUME_ROOT",
        str(
            Path(os.environ.get("AFCE_RUNTIME", "runtime"))
            / "query_c01_single_finger"
            / "gates"
            / "resume"
            / "checkpoints"
            / "afce_all11_official"
        ),
    )
).resolve()
runs = {
    name: pickle.loads((root / name / "assets" / "training_runtime.pkl").read_bytes())
    for name in ("continuous/3", "interrupted/1", "interrupted/3")
}
for rank in range(2):
    print("RANK", rank)
    for name, d in runs.items():
        r = d["host_rng_by_process"][rank]
        print(name, {k: hashlib.sha256(pickle.dumps(v)).hexdigest()[:12] for k, v in r.items()})
        base = random.Random(42 + rank)
        offset = None
        for i in range(10000):
            if base.getstate() == r["python"]:
                offset = i
                break
            base.random()
        print("python_random_draw_offset", offset, "py_state_index", r["python"][1][-1])
    for k in runs["continuous/3"]["host_rng_by_process"][rank]:
        print(
            "field",
            k,
            "equal",
            equal_nested(
                runs["continuous/3"]["host_rng_by_process"][rank][k],
                runs["interrupted/3"]["host_rng_by_process"][rank][k],
            ),
        )
