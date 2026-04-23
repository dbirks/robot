import base64
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import cv2
import numpy as np
from openai import OpenAI
from reachy_mini.utils import create_head_pose

from .robot_state import RobotConnection

log = logging.getLogger(__name__)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "look_left",
            "description": "Turn the robot's head to look left",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_right",
            "description": "Turn the robot's head to look right",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_center",
            "description": "Return the robot's head to center position",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nod",
            "description": "Make the robot nod its head (yes gesture)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shake_head",
            "description": "Make the robot shake its head (no gesture)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_snapshot",
            "description": "Capture a photo from the robot's camera and save it",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_status",
            "description": "Get the current status of the robot (connected, position)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_scene",
            "description": "Look through the camera and describe what you see. Use this when someone asks what you see, or to comment on your surroundings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Optional specific question about the scene (e.g. 'what color is their hat?')",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save an important fact to long-term memory (e.g. a person's name, preference, or something you learned)",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember",
                    }
                },
                "required": ["fact"],
            },
        },
    },
]

MOTION_DURATION = 0.8
NOD_DURATION = 0.3
NOD_ANGLE = 15
LOOK_ANGLE = 30


def make_handlers(robot: RobotConnection, agent=None) -> dict[str, Callable]:
    def _require_robot():
        if not robot.connected or robot.mini is None:
            return {"ok": False, "error": "Robot not connected"}
        return None

    def look_left(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=create_head_pose(yaw=LOOK_ANGLE, degrees=True), duration=MOTION_DURATION)
            return {"ok": True, "action": "look_left"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def look_right(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=create_head_pose(yaw=-LOOK_ANGLE, degrees=True), duration=MOTION_DURATION)
            return {"ok": True, "action": "look_right"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def look_center(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=create_head_pose(), duration=MOTION_DURATION)
            return {"ok": True, "action": "look_center"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def nod(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=create_head_pose(pitch=-NOD_ANGLE, degrees=True), duration=NOD_DURATION)
            robot.mini.goto_target(head=create_head_pose(pitch=NOD_ANGLE, degrees=True), duration=NOD_DURATION)
            robot.mini.goto_target(head=create_head_pose(), duration=NOD_DURATION)
            return {"ok": True, "action": "nod"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def shake_head(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=create_head_pose(yaw=20, degrees=True), duration=NOD_DURATION)
            robot.mini.goto_target(head=create_head_pose(yaw=-20, degrees=True), duration=NOD_DURATION)
            robot.mini.goto_target(head=create_head_pose(yaw=20, degrees=True), duration=NOD_DURATION)
            robot.mini.goto_target(head=create_head_pose(), duration=NOD_DURATION)
            return {"ok": True, "action": "shake_head"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def take_snapshot(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            frame = robot.mini.media.get_frame()
            if frame is None:
                return {"ok": False, "error": "No frame available from camera"}
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = f"logs/snapshot_{ts}.jpg"
            cv2.imwrite(path, frame)
            return {"ok": True, "path": path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_robot_status(**_kwargs: Any) -> dict:
        return {
            "connected": robot.connected,
            "host": robot.config.reachy_host,
        }

    def get_time(**_kwargs: Any) -> dict:
        return {"time": datetime.now(tz=timezone.utc).isoformat()}

    def describe_scene(question: str = "", **_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            frame = robot.mini.media.get_frame()
            if frame is None:
                return {"ok": False, "error": "No frame available from camera"}
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode()

            prompt = question if question else "Describe what you see briefly in 1-2 sentences."
            client = OpenAI(base_url=agent.client.base_url, api_key=agent.client.api_key)
            resp = client.chat.completions.create(
                model=agent.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            description = resp.choices[0].message.content or ""
            log.info("Vision: %s", description)
            return {"ok": True, "description": description}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def remember(fact: str = "", **_kwargs: Any) -> dict:
        if not fact:
            return {"ok": False, "error": "No fact provided"}
        if agent is None:
            return {"ok": False, "error": "Agent not available"}
        try:
            agent.save_memory(fact)
            return {"ok": True, "saved": fact}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {
        "look_left": look_left,
        "look_right": look_right,
        "look_center": look_center,
        "nod": nod,
        "shake_head": shake_head,
        "take_snapshot": take_snapshot,
        "describe_scene": describe_scene,
        "get_robot_status": get_robot_status,
        "get_time": get_time,
        "remember": remember,
    }
