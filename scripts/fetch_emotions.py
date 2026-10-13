#!/usr/bin/env python3
"""Fetch the gesture library into the local HF cache. A TRAVEL STEP, not a runtime step.

`play_emotion` reads with `local_files_only=True`, so at the booth it either has the files or it
says so and keeps talking (shell/motion/emotes.py explains why it must not reach for the network
inside a tool call: a 12 s timeout on an etag request over filtered venue egress, answered with a
message that tells the model the word it used is invalid, is a retry storm waiting to happen).

Run this at home, on the box, before you travel. It is the only place in the repo that hits
huggingface.co on purpose, and it prints the revision it pinned so docs/pins.yaml can be updated
from a real value instead of a remembered one.

    uv run python scripts/fetch_emotions.py            # fetch + verify all ten
    uv run python scripts/fetch_emotions.py --check    # verify only, never network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shell.motion import emotes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the cache; do not download")
    args = ap.parse_args(argv)

    if not args.check:
        from huggingface_hub import hf_hub_download  # the one deliberate network call

        print(f"fetching {len(emotes.EMOTE_FILES)} gesture files from {emotes.REPO_ID}...")
        for name, filename in emotes.EMOTE_FILES.items():
            try:
                path = hf_hub_download(emotes.REPO_ID, filename, repo_type=emotes.REPO_TYPE)
                print(f"  {name:11} {filename:20} {Path(path).stat().st_size:>9} B")
            except Exception as e:
                print(f"  {name:11} {filename:20} FAILED {type(e).__name__}: {e}", file=sys.stderr)

    st = emotes.status()
    print(
        f"\nresolved offline: {len(st['cached'])}/{len(emotes.EMOTE_NAMES)}  "
        f"revision {st['revision']}  {st['bytes']} B cached"
    )
    for name in emotes.EMOTE_NAMES:
        if name in st["cached"]:
            try:
                anim = emotes.load(name)
                seconds = sum(kf.duration for kf in anim.keyframes)
                print(f"  ok   {name:11} {len(anim.keyframes):2} keyframes  {seconds:5.2f}s")
            except Exception as e:
                print(f"  BAD  {name:11} {type(e).__name__}: {e}", file=sys.stderr)
        else:
            print(f"  MISS {name:11} {st['missing']}", file=sys.stderr)
    if len(st["cached"]) < len(emotes.EMOTE_NAMES):
        print(
            "\nNot every gesture is available offline. Run this without --check on a networked "
            "box before travel; the booth will not fetch it for you.",
            file=sys.stderr,
        )
        return 1
    print(f"\ndocs/pins.yaml -> models.emotions.revision should read: {st['revision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
