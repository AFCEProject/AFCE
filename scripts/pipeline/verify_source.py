"""Verify this release against the recorded E and policy training sources."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "experiments/finger_delta/reference/source_manifest.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    aggregate = hashlib.sha256()
    for name in manifest["e_source_files_in_hash_order"]:
        path = root / name
        if not path.is_file():
            continue
        aggregate.update(name.encode())
        aggregate.update(path.read_bytes())
    if aggregate.hexdigest() != manifest["e_source_sha256"]:
        failures.append("E training aggregate hash")
    print(json.dumps({"passed": not failures, "checked_files": len(manifest["files"]),
                      "e_source_sha256": aggregate.hexdigest(), "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
