"""Pure pose math for the motion owner (no SDK, no hardware).

Poses are 6-vectors [x, y, z, roll, pitch, yaw] in metres / radians, the same
convention as reachy_mini.utils.create_head_pose (extrinsic "xyz", i.e.
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)). Layers compose by addition in this
space, which is exact for the base yaw/pitch plus small additive offsets the
loop deals in, and never produces the non-orthonormal matrices the legacy
4x4 lerp did. Interpolators ported from app/movement_manager.py.
"""

from __future__ import annotations

import math

import numpy as np

LN2 = 0.693147


def smooth_step(t: float) -> float:
    """Hermite smooth step t*t*(3-2t), clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def minimum_jerk(t: float) -> float:
    """10t^3 - 15t^4 + 6t^5: zero velocity and acceleration at both ends."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def spring_update(pos, vel, target, halflife: float, dt: float):
    """Critically damped spring: smooth accel/decel, zero overshoot.

    Works on floats or numpy arrays. `halflife` is the time to close half
    the remaining distance (from rest)."""
    y = (4.0 * LN2) / (halflife + 1e-5) / 2.0
    j0 = pos - target
    j1 = vel + j0 * y
    eydt = math.exp(-y * dt)
    return eydt * (j0 + j1 * dt) + target, eydt * (vel - j1 * y * dt)


def exp_alpha(dt: float, tau: float) -> float:
    """Frame-rate independent smoothing factor for x += alpha * (target - x)."""
    return 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0


def pose_matrix(p) -> np.ndarray:
    x, y, z, roll, pitch, yaw = (float(v) for v in p)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    m[:3, 3] = (x, y, z)
    return m


def pose6_from_matrix(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    r = m[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    return np.array([m[0, 3], m[1, 3], m[2, 3], roll, pitch, yaw])


def pose6(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, *, degrees: bool = True) -> np.ndarray:
    k = math.pi / 180.0 if degrees else 1.0
    return np.array([x, y, z, roll * k, pitch * k, yaw * k], dtype=np.float64)
