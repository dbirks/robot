#!/usr/bin/env python3
"""Fetch the dances library into the local HF cache. A TRAVEL STEP, not a runtime step.

The sibling of fetch_emotions.py, for the same reason it exists: `dance` reads with
`local_files_only=True` (shell/motion/dances.py), and the s2s unit runs with HF_HUB_OFFLINE=1, so
anything the robot needs must already be on this disk before the car is loaded. At the venue the
tool either has the files or says so and keeps talking - a network round trip inside a 12 s tool
timeout is how a missing gesture becomes a retry storm.

Run this at home, on the box, before you travel. It is the only place in the repo that hits
huggingface.co for the dance pack on purpose, and it prints the revision it pinned so
docs/pins.yaml can be updated from a real value instead of a remembered one.

    uv run python scripts/fetch_dances.py            # fetch + verify all nineteen
    uv run python scripts/fetch_dances.py --check    # verify only, never network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shell.motion import dances  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the cache; do not download")
    args = ap.parse_args(argv)

    if not args.check:
        from huggingface_hub import hf_hub_download  # the one deliberate network call

        print(f"fetching {len(dances.DANCE_FILES)} dance files from {dances.REPO_ID}...")
        for name, filename in dances.DANCE_FILES.items():
            try:
                path = hf_hub_download(dances.REPO_ID, filename, repo_type=dances.REPO_TYPE)
                print(f"  {name:22} {filename:28} {Path(path).stat().st_size:>9} B")
            except Exception as e:
                print(f"  {name:22} {filename:28} FAILED {type(e).__name__}: {e}", file=sys.stderr)

    st = dances.status()
    print(
        f"\nresolved offline: {len(st['cached'])}/{len(dances.DANCE_NAMES)}  "
        f"revision {st['revision']}  {st['bytes']} B cached"
    )
    for name in dances.DANCE_NAMES:
        if name in st["cached"]:
            try:
                anim = dances.load(name)
                seconds = sum(kf.duration for kf in anim.keyframes)
                print(f"  ok   {name:22} {len(anim.keyframes):2} keyframes  {seconds:5.2f}s")
            except Exception as e:
                print(f"  BAD  {name:22} {type(e).__name__}: {e}", file=sys.stderr)
        else:
            print(f"  MISS {name:22} {st['missing']}", file=sys.stderr)
    if len(st["cached"]) < len(dances.DANCE_NAMES):
        print(
            "\nNot every dance is available offline. Run this without --check on a networked "
            "box before travel; the booth will not fetch it for you.",
            file=sys.stderr,
        )
        return 1
    print(f"\ndocs/pins.yaml -> models.dances.revision should read: {st['revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
