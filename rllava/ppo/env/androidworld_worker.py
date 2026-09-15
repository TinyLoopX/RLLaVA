#!/usr/bin/env python3
"""
AndroidWorld subprocess worker script.

This script runs in a separate process (potentially in a different conda env)
and communicates with the parent RLLAVA process via JSON over stdin/stdout.

Protocol:
  Parent sends JSON commands via stdin (one per line):
    {"cmd": "reset", "env_kwargs": {...}, "task_name": "...", "task_params": {...}}
    {"cmd": "step", "action": {"action_type": "click", "x": 100, "y": 200}}
    {"cmd": "evaluate"}
    {"cmd": "close"}

  Worker replies with JSON via stdout (one per line):
    {"status": "ok", "obs": {...}, "reward": 0.0, "done": false, "info": {...}}
    {"status": "error", "message": "..."}

  All logging goes to stderr to avoid polluting the JSON protocol.

AndroidWorld-specific notes:
  - State.pixels is a numpy array (H, W, 3), encoded as base64 PNG
  - ui_elements are serialized to JSON-safe dicts
  - Tasks are loaded from the AndroidWorld registry by task_name + params
  - Actions are JSONAction dicts (action_type, x, y, text, index, etc.)
"""

import sys
import os
import json
import base64
import logging
import traceback
import io
import time
import dataclasses

import numpy as np

# All logging to stderr so stdout stays clean for JSON protocol
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[AndroidWorld-Worker %(levelname)s] %(message)s",
)
logger = logging.getLogger("androidworld_worker")


def encode_pixels(pixels) -> str:
    """Encode numpy pixel array to base64 PNG string."""
    if pixels is None:
        return ""
    try:
        from PIL import Image
        if isinstance(pixels, np.ndarray):
            img = Image.fromarray(pixels.astype(np.uint8))
        elif hasattr(pixels, "save"):
            img = pixels
        else:
            return ""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to encode pixels: %s", e)
        return ""


def ui_element_to_dict(elem) -> dict:
    """Convert a UIElement to a JSON-serializable dict."""
    if elem is None:
        return {}
    if dataclasses.is_dataclass(elem):
        result = {}
        for field in dataclasses.fields(elem):
            val = getattr(elem, field.name, None)
            if val is None:
                continue
            # Handle nested dataclasses (e.g., BoundingBox)
            if dataclasses.is_dataclass(val):
                val = dataclasses.asdict(val)
            # Skip non-serializable types
            try:
                json.dumps(val)
                result[field.name] = val
            except (TypeError, ValueError):
                result[field.name] = str(val)
        return result
    elif isinstance(elem, dict):
        return elem
    else:
        return {"repr": str(elem)}


def state_to_dict(state) -> dict:
    """Convert AndroidWorld State to JSON-serializable dict."""
    result = {}
    if state is None:
        return result

    # pixels: numpy array -> base64 PNG
    pixels = getattr(state, "pixels", None)
    if pixels is not None:
        result["screenshot_b64"] = encode_pixels(pixels)

    # ui_elements: list[UIElement] -> list[dict]
    ui_elements = getattr(state, "ui_elements", None)
    if ui_elements is not None:
        result["ui_elements"] = [ui_element_to_dict(e) for e in ui_elements]

    # forest: protobuf -> skip (too heavy, not needed for policy)
    # We only pass ui_elements which is the processed version

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


def create_env(env_kwargs: dict):
    """Create AndroidWorld environment from kwargs."""
    console_port = env_kwargs.get("console_port", 5554)
    adb_path = env_kwargs.get("adb_path", "~/Android/Sdk/platform-tools/adb")
    grpc_port = env_kwargs.get("grpc_port", 8554)

    # Try env_launcher first (handles emulator setup, datetime freeze, etc.)
    use_launcher = env_kwargs.get("use_launcher", True)
    if use_launcher:
        try:
            from android_world.env import env_launcher
            env = env_launcher.load_and_setup_env(
                console_port=console_port,
                emulator_setup=env_kwargs.get("emulator_setup", False),
                freeze_datetime=env_kwargs.get("freeze_datetime", True),
                adb_path=adb_path,
                grpc_port=grpc_port,
            )
            logger.info("Created env via env_launcher (port=%d)", console_port)
            return env
        except Exception as e:
            logger.warning("env_launcher failed, falling back to manual: %s", e)

    # Fallback: manual controller creation
    from android_world.env import android_world_controller
    from android_world.env import interface as aw_interface
    controller = android_world_controller.get_controller(
        console_port=console_port,
        adb_path=adb_path,
        grpc_port=grpc_port,
    )
    env = aw_interface.AsyncAndroidEnv(controller)
    logger.info("Created env via manual controller (port=%d)", console_port)
    return env


def load_task(task_name: str, task_params: dict = None):
    """Load a task from the AndroidWorld registry.
    
    Args:
        task_name: Task class name in the registry (e.g., "ContactsAddContact")
        task_params: Parameters for the task. If None, generates random params.
        
    Returns:
        TaskEval instance
    """
    from android_world.registry import TaskRegistry

    registry = TaskRegistry()

    # Search across all families for the task
    task_cls = None
    for family in [
        TaskRegistry.ANDROID_WORLD_FAMILY,
        TaskRegistry.ANDROID_FAMILY,
        TaskRegistry.MINIWOB_FAMILY,
        TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
    ]:
        try:
            family_registry = registry.get_registry(family)
            if task_name in family_registry:
                task_cls = family_registry[task_name]
                break
        except Exception:
            continue

    if task_cls is None:
        raise ValueError(f"Task '{task_name}' not found in any registry family")

    if task_params is None:
        task_params = task_cls.generate_random_params()

    task = task_cls(task_params)
    logger.info("Loaded task '%s' with goal: %s", task_name, task.goal)
    return task


def execute_action(env, action_dict: dict):
    """Execute an action on the AndroidWorld environment.
    
    Args:
        env: AsyncAndroidEnv instance
        action_dict: Dict with action_type and parameters
    """
    from android_world.env import json_action

    # Build JSONAction from dict
    action = json_action.JSONAction(
        action_type=action_dict.get("action_type"),
        index=action_dict.get("index"),
        x=action_dict.get("x"),
        y=action_dict.get("y"),
        text=action_dict.get("text"),
        direction=action_dict.get("direction"),
        app_name=action_dict.get("app_name"),
        goal_status=action_dict.get("goal_status"),
        keycode=action_dict.get("keycode"),
        clear_text=action_dict.get("clear_text"),
    )
    env.execute_action(action)


def main():
    env = None
    task = None

    logger.info("AndroidWorld worker started, waiting for commands...")

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
                env_kwargs = cmd_data.get("env_kwargs", {})
                task_name = cmd_data.get("task_name")
                task_params = cmd_data.get("task_params")
                task_goal = cmd_data.get("task_goal")

                # Create environment if not exists
                if env is None:
                    logger.info("Creating AndroidWorld env with kwargs: %s", env_kwargs)
                    env = create_env(env_kwargs)

                # Load task if task_name provided
                task = None
                if task_name:
                    task = load_task(task_name, task_params)
                    task.initialize_task(env)
                    goal = task.goal
                elif task_goal:
                    goal = task_goal
                else:
                    goal = ""

                # Reset environment
                logger.info("Resetting environment...")
                state = env.reset(go_home=True)

                # Wait for environment readiness
                wait_time = cmd_data.get("wait_after_reset", 3)
                if wait_time > 0:
                    logger.info("Waiting %ds for environment readiness...", wait_time)
                    time.sleep(wait_time)
                    state = env.get_state(wait_to_stabilize=True)

                obs = state_to_dict(state)
                obs["goal"] = goal

                send_ok(obs=obs)

            elif cmd == "step":
                if env is None:
                    send_error("Environment not initialized. Call reset first.")
                    continue

                action_dict = cmd_data.get("action")
                if not isinstance(action_dict, dict):
                    send_error(f"Action must be a dict, got {type(action_dict).__name__}")
                    continue

                action_type = action_dict.get("action_type", "")
                logger.info("Executing action: %s", action_type)

                # Execute action
                execute_action(env, action_dict)

                # Wait for UI to stabilize
                wait_stable = cmd_data.get("wait_to_stabilize", True)
                state = env.get_state(wait_to_stabilize=wait_stable)

                obs = state_to_dict(state)

                # Check if done
                done = False
                info = {"action_type": action_type}

                if action_type == "status":
                    done = True
                    info["goal_status"] = action_dict.get("goal_status")

                send_ok(obs=obs, reward=0.0, done=done, info=info)

            elif cmd == "evaluate":
                if env is None:
                    send_error("Environment not initialized. Call reset first.")
                    continue

                if task is None:
                    logger.warning("No task loaded, cannot evaluate. Returning 0.0")
                    send_ok(result=0.0)
                    continue

                logger.info("Running evaluation...")
                wait_time = cmd_data.get("wait_before_eval", 3)
                if wait_time > 0:
                    time.sleep(wait_time)

                try:
                    result = task.is_successful(env)
                    result = float(result) if result is not None else 0.0
                except Exception as e:
                    logger.error("Evaluation failed: %s", e)
                    result = 0.0

                logger.info("Evaluation result: %s", result)
                send_ok(result=result)

            elif cmd == "teardown":
                # Tear down task (cleanup task state on device)
                if task is not None and env is not None:
                    try:
                        task.tear_down(env)
                        logger.info("Task torn down successfully")
                    except Exception as e:
                        logger.warning("Error tearing down task: %s", e)
                    task = None
                send_ok()

            elif cmd == "close":
                logger.info("Closing environment...")
                if task is not None and env is not None:
                    try:
                        task.tear_down(env)
                    except Exception as e:
                        logger.warning("Error tearing down task: %s", e)
                if env is not None:
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning("Error closing env: %s", e)
                    env = None
                task = None
                send_ok()
                break  # Exit the worker loop

            elif cmd == "ping":
                send_ok(message="pong")

            elif cmd == "list_tasks":
                # Utility: list available task names
                try:
                    from android_world.registry import TaskRegistry
                    registry = TaskRegistry()
                    all_tasks = {}
                    for family in [
                        TaskRegistry.ANDROID_WORLD_FAMILY,
                        TaskRegistry.ANDROID_FAMILY,
                        TaskRegistry.MINIWOB_FAMILY,
                        TaskRegistry.INFORMATION_RETRIEVAL_FAMILY,
                    ]:
                        try:
                            family_registry = registry.get_registry(family)
                            all_tasks[family] = list(family_registry.keys())
                        except Exception:
                            continue
                    send_ok(tasks=all_tasks)
                except Exception as e:
                    send_error(f"Failed to list tasks: {e}")

            else:
                send_error(f"Unknown command: {cmd}")

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Exception handling cmd '%s': %s\n%s", cmd, e, tb)
            send_error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
