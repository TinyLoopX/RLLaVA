#!/usr/bin/env python3
"""
OSWorld subprocess worker script.

This script runs in a separate process (potentially in a different conda env)
and communicates with the parent RLLAVA process via JSON over stdin/stdout.

Protocol:
  Parent sends JSON commands via stdin (one per line):
    {"cmd": "reset", "task_config": {...}, "env_kwargs": {...}}
    {"cmd": "step", "action": "pyautogui.click(100, 200)"}
    {"cmd": "evaluate"}
    {"cmd": "close"}

  Worker replies with JSON via stdout (one per line):
    {"status": "ok", "obs": {...}, "reward": 0.0, "done": false, "info": {...}}
    {"status": "error", "message": "..."}

  All logging goes to stderr to avoid polluting the JSON protocol.
"""

import sys
import os
import json
import base64
import logging
import traceback
import io
import time

# All logging to stderr so stdout stays clean for JSON protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[OSWorld-Worker %(levelname)s] %(message)s",
)
logger = logging.getLogger("osworld_worker")


def encode_screenshot(screenshot_bytes) -> str:
    """Encode screenshot bytes to base64 string."""
    if screenshot_bytes is None:
        return ""
    if isinstance(screenshot_bytes, bytes):
        return base64.b64encode(screenshot_bytes).decode("utf-8")
    # If it's a PIL Image
    if hasattr(screenshot_bytes, "save"):
        buf = io.BytesIO()
        screenshot_bytes.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return ""


def obs_to_dict(obs: dict) -> dict:
    """Convert OSWorld observation to JSON-serializable dict."""
    result = {}
    if obs is None:
        return result

    # screenshot: bytes -> base64
    screenshot = obs.get("screenshot")
    if screenshot is not None:
        result["screenshot_b64"] = encode_screenshot(screenshot)

    # text fields
    for key in ("instruction", "accessibility_tree", "terminal"):
        val = obs.get(key)
        if val is not None:
            result[key] = str(val)

    return result


def send_response(data: dict):
    """Send JSON response to parent via stdout."""
    line = json.dumps(data, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def send_ok(**kwargs):
    resp = {"status": "ok"}
    resp.update(kwargs)
    send_response(resp)


def send_error(message: str):
    send_response({"status": "error", "message": message})


def main():
    env = None

    logger.info("OSWorld worker started, waiting for commands...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd_data = json.loads(line)
        except json.JSONDecodeError as e:
            send_error(f"Invalid JSON: {e}")
            continue

        cmd = cmd_data.get("cmd")

        try:
            if cmd == "reset":
                # Create or reuse DesktopEnv
                env_kwargs = cmd_data.get("env_kwargs", {})
                task_config = cmd_data.get("task_config")

                if env is None:
                    logger.info("Creating DesktopEnv with kwargs: %s", env_kwargs)
                    from desktop_env.desktop_env import DesktopEnv
                    env = DesktopEnv(**env_kwargs)

                logger.info("Resetting environment with task_config...")
                obs = env.reset(task_config=task_config)

                # Wait a bit for environment readiness
                wait_time = cmd_data.get("wait_after_reset", 5)
                if wait_time > 0:
                    logger.info("Waiting %ds for environment readiness...", wait_time)
                    time.sleep(wait_time)
                    # Get fresh observation after waiting
                    obs = env._get_obs()

                send_ok(obs=obs_to_dict(obs))

            elif cmd == "step":
                if env is None:
                    send_error("Environment not initialized. Call reset first.")
                    continue

                action = cmd_data.get("action")
                sleep_after = cmd_data.get("sleep_after_execution", 3)

                logger.info("Executing action: %s", str(action)[:200])
                obs, reward, done, info = env.step(action, sleep_after)

                # Make info JSON-serializable
                safe_info = {}
                if isinstance(info, dict):
                    for k, v in info.items():
                        try:
                            json.dumps(v)
                            safe_info[k] = v
                        except (TypeError, ValueError):
                            safe_info[k] = str(v)

                send_ok(
                    obs=obs_to_dict(obs),
                    reward=float(reward) if reward is not None else 0.0,
                    done=bool(done),
                    info=safe_info,
                )

            elif cmd == "evaluate":
                if env is None:
                    send_error("Environment not initialized. Call reset first.")
                    continue

                logger.info("Running evaluation...")
                # Wait a bit before evaluation to let environment settle
                wait_time = cmd_data.get("wait_before_eval", 5)
                if wait_time > 0:
                    time.sleep(wait_time)

                result = env.evaluate()
                logger.info("Evaluation result: %s", result)
                send_ok(result=float(result) if result is not None else 0.0)

            elif cmd == "close":
                logger.info("Closing environment...")
                if env is not None:
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning("Error closing env: %s", e)
                    env = None
                send_ok()
                break  # Exit the worker loop

            elif cmd == "ping":
                send_ok(message="pong")

            else:
                send_error(f"Unknown command: {cmd}")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Exception handling cmd '%s': %s\n%s", cmd, e, tb)
            send_error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
