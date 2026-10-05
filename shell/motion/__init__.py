"""Head/antenna motion: one owner of set_target (see owner.py)."""

from .doa import DoaTracker, make_doa_source
from .owner import Animation, Command, Keyframe, MotionOwner, nod_animation, shake_animation

__all__ = [
    "Animation",
    "Command",
    "DoaTracker",
    "Keyframe",
    "MotionOwner",
    "make_doa_source",
    "nod_animation",
    "shake_animation",
]
