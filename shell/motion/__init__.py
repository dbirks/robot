"""Head/antenna motion: one owner of set_target (see owner.py)."""

from .owner import Animation, Command, Keyframe, MotionOwner, nod_animation, shake_animation

__all__ = ["Animation", "Command", "Keyframe", "MotionOwner", "nod_animation", "shake_animation"]
