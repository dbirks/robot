"""Vision for the shell: one detector, geometric presence, no memory of faces.

Detection only. Nothing in here recognises, embeds, persists or compares two
visits (docs/environments.md B1 is the recognition lane, and it is deliberately
not landed); the greeter's "same person" is a bbox in a frame, not a signature.
"""

from .gaze import Gate, area_share, azimuth_deg, estimate_orientation, looking_at_me
from .greeter import GROUP_CUE, SOLO_CUE, GreetCue, Greeter, Track, greet_hook
from .tracker import Face, FaceTracker

__all__ = [
    "Face",
    "FaceTracker",
    "GROUP_CUE",
    "Gate",
    "Greeter",
    "GreetCue",
    "SOLO_CUE",
    "Track",
    "greet_hook",
    "area_share",
    "azimuth_deg",
    "estimate_orientation",
    "looking_at_me",
]
