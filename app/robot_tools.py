import base64
import json
import logging
import random
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import sounddevice as sd
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
            "description": "Describe the general scene through the camera (objects, colors, environment). This is SLOW (~10s). Do NOT use for identifying people — use identify_face instead, which is instant.",
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
    {
        "type": "function",
        "function": {
            "name": "learn_face",
            "description": "Learn and remember the face of the person currently in front of the camera. Use when someone tells you their name and you want to recognize them later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The person's name to associate with their face",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_face",
            "description": "Identify all faces currently visible in the camera. Returns a list of all detected faces with their names (if recognized). Use when you want to see who is in the room.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_face",
            "description": "Remove a previously learned face from memory. Use when asked to forget someone or to clean up incorrect face entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person whose face to forget",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use when the user asks a question you don't know the answer to, or asks you to look something up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_conversation",
            "description": "Reset the conversation and start fresh. Use when the user says things like 'start over', 'new conversation', 'forget what we talked about', or 'reset'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Adjust the robot's speaking volume. Use when asked to be louder, quieter, or set a specific volume level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "Volume level: 'low', 'medium', 'high', or a number 1-10",
                    }
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_to_sleep",
            "description": "Put the robot to sleep. ONLY use this when the user directly and explicitly tells you to go to sleep or shut down. Never call this just because someone mentions sleep or says goodbye.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "peekaboo",
            "description": "Play peekaboo! Hide the robot's head down, then pop up. Great for playing with kids.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SLEEP_HEAD_POSE = np.array(
    [
        [0.911, 0.004, 0.413, -0.021],
        [-0.004, 1.0, -0.001, 0.001],
        [-0.413, -0.001, 0.911, -0.044],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
SLEEP_ANTENNAS = [-3.05, 3.05]

MOTION_DURATION = 0.8
NOD_DURATION = 0.3
NOD_ANGLE = 15
LOOK_ANGLE = 30


def make_handlers(
    robot: RobotConnection,
    agent=None,
    face_tracker=None,
    sleep_event: threading.Event | None = None,
    movement=None,
    wobbler=None,
) -> dict[str, Callable]:
    def _require_robot():
        if not robot.connected or robot.mini is None:
            return {"ok": False, "error": "Robot not connected"}
        return None

    def _play_sdk_sound(name: str):
        import importlib.resources

        sound_path = Path(importlib.resources.files("reachy_mini") / "assets" / name)
        if not sound_path.exists():
            log.warning("Sound not found: %s", name)
            return
        with wave.open(str(sound_path)) as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sr = wf.getframerate()
            ch = wf.getnchannels()
        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]
        from .playback import _output_device, _device_sr, _resample

        if _device_sr and sr != _device_sr:
            audio = _resample(audio, sr, _device_sr)
            sr = _device_sr
        sd.play(audio, samplerate=sr, device=_output_device)
        sd.wait()

    def _grab_camera_frame():
        if face_tracker is not None:
            frame = face_tracker.grab_frame()
            if frame is not None:
                return frame
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return frame
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
        try:
            frame = _grab_camera_frame()
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
        try:
            frame = _grab_camera_frame()
            if frame is None:
                return {"ok": False, "error": "No frame available from camera"}
            h, w = frame.shape[:2]
            if w > 320:
                scale = 320 / w
                frame = cv2.resize(frame, (320, int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
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

    def learn_face(name: str = "", **_kwargs: Any) -> dict:
        if not name:
            return {"ok": False, "error": "No name provided"}
        if face_tracker is None:
            return {"ok": False, "error": "Face tracker not available"}
        try:
            frame = _grab_camera_frame()
            if frame is None:
                return {"ok": False, "error": "No frame available from camera"}
            faces = face_tracker.detect(frame)
            if not faces:
                return {"ok": False, "error": "No face detected in frame"}
            face_tracker.register_face(name, faces[0]["embedding"])
            return {"ok": True, "name": name, "known_faces": list(face_tracker.known_faces.keys())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def identify_face(**_kwargs: Any) -> dict:
        if face_tracker is None:
            return {"ok": False, "error": "Face tracker not available"}
        try:
            frame = _grab_camera_frame()
            if frame is None:
                return {"ok": False, "error": "No frame available from camera"}
            faces = face_tracker.detect(frame)
            if not faces:
                return {"ok": False, "error": "No face detected in frame"}
            results = []
            for face in faces:
                name = face_tracker.identify(face["embedding"])
                results.append({"name": name, "recognized": name is not None})
            return {"ok": True, "faces": results, "count": len(results)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def forget_face(name: str = "", **_kwargs: Any) -> dict:
        if not name:
            return {"ok": False, "error": "No name provided"}
        if face_tracker is None:
            return {"ok": False, "error": "Face tracker not available"}
        try:
            if name not in face_tracker.known_faces:
                return {
                    "ok": False,
                    "error": f"No face named '{name}' found",
                    "known_faces": list(face_tracker.known_faces.keys()),
                }
            del face_tracker.known_faces[name]
            face_tracker._save_faces()
            log.info("Forgot face: %s", name)
            return {"ok": True, "forgotten": name, "known_faces": list(face_tracker.known_faces.keys())}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_volume(level: str = "medium", **_kwargs: Any) -> dict:
        try:
            from .playback import set_volume_boost
            level_map = {"low": 0.5, "quiet": 0.5, "medium": 1.0, "normal": 1.0, "high": 1.8, "loud": 1.8}
            if level.lower() in level_map:
                boost = level_map[level.lower()]
            else:
                try:
                    num = float(level)
                    boost = max(0.1, min(3.0, num / 5.0))
                except ValueError:
                    return {"ok": False, "error": f"Unknown volume level: {level}"}
            set_volume_boost(boost)
            log.info("Volume set to %.1f (level=%s)", boost, level)
            return {"ok": True, "volume": boost, "level": level}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def reset_conversation(**_kwargs: Any) -> dict:
        if agent is None:
            return {"ok": False, "error": "Agent not available"}
        try:
            agent._pending_reset = True
            return {"ok": True, "action": "conversation_reset", "message": "Say a brief greeting to start fresh."}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def web_search(query: str = "", **_kwargs: Any) -> dict:
        if not query:
            return {"ok": False, "error": "No query provided"}
        try:
            from duckduckgo_search import DDGS

            results = DDGS().text(query, max_results=3)
            snippets = [{"title": r["title"], "body": r["body"]} for r in results]
            return {"ok": True, "results": snippets}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def peekaboo(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        try:
            robot.mini.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS, duration=1.0)
            time.sleep(1.0)
            hide_time = random.uniform(1.5, 4.0)
            time.sleep(hide_time)
            robot.mini.goto_target(
                head=create_head_pose(pitch=-10, degrees=True),
                antennas=[0.5, 0.5],
                duration=0.2,
            )
            _play_sdk_sound("wake_up.wav")
            time.sleep(0.8)
            robot.mini.goto_target(head=create_head_pose(), antennas=[-0.1745, 0.1745], duration=0.8)
            return {"ok": True, "action": "peekaboo"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def go_to_sleep(**_kwargs: Any) -> dict:
        if err := _require_robot():
            return err
        if sleep_event is None:
            return {"ok": False, "error": "Sleep not supported"}
        try:
            # Stop movement systems BEFORE the sleep animation so they
            # don't fight the goto_target with their 60Hz control loop.
            if face_tracker:
                face_tracker.stop_tracking()
            if wobbler:
                wobbler.stop()
            if movement:
                movement.stop()
            robot.mini.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS, duration=2)
            sleep_event.set()
            log.info("Robot going to sleep")
            return {"ok": True, "action": "sleeping"}
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
        "learn_face": learn_face,
        "identify_face": identify_face,
        "forget_face": forget_face,
        "set_volume": set_volume,
        "reset_conversation": reset_conversation,
        "web_search": web_search,
        "peekaboo": peekaboo,
        "go_to_sleep": go_to_sleep,
    }
