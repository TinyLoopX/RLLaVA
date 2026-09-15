"""
AndroidWorld Subprocess Environment for RLLAVA.

Uses a subprocess to run AndroidWorld's AsyncAndroidEnv in an isolated Python
environment, communicating via JSON over stdin/stdout pipes. This avoids
dependency conflicts between RLLAVA (torch, transformers, etc.) and
AndroidWorld (android_env, dm_env, protobuf, etc.).

Architecture:
    ┌───────────────────────────────┐     JSON/stdin/stdout     ┌─────────────────────────────────┐
    │  RLLAVA Main Process          │  ←───────────────────→   │  androidworld_worker.py          │
    │  (AndroidWorldSubprocessEnv)  │  reset/step/evaluate     │  (AsyncAndroidEnv + ADB + Task)  │
    │  BaseEnv interface            │                           │  Runs in AndroidWorld conda env  │
    └───────────────────────────────┘                           └─────────────────────────────────┘
                                                                           ↕ ADB
                                                                ┌─────────────────────────────────┐
                                                                │  Android Emulator / Device       │
                                                                └─────────────────────────────────┘

Usage in config YAML:
    class_name: AndroidWorldSubprocessEnv
    config:
      android_world_dir: /path/to/android_world-main
      python_executable: /path/to/androidworld_conda/bin/python
      console_port: 5554
      adb_path: ~/Android/Sdk/platform-tools/adb
      max_steps: 30
"""

import os
import re
import io
import json
import base64
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import BaseEnv

logger = logging.getLogger(__name__)


class AndroidWorldSubprocessEnv(BaseEnv):
    """AndroidWorld environment running in a subprocess, implementing RLLAVA's BaseEnv.

    The subprocess runs androidworld_worker.py in AndroidWorld's Python
    environment, which manages the actual AsyncAndroidEnv (emulator connection,
    action execution, screenshot capture, task evaluation).

    This class handles:
    - Subprocess lifecycle management (spawn, communicate, kill)
    - JSON protocol marshalling (serialize actions, deserialize observations)
    - Screenshot decoding (base64 PNG -> PIL.Image)
    - Action extraction from model output text
    - Task management (loading from registry, evaluation)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # --- Subprocess config ---
        self.android_world_dir = config.get("android_world_dir", "")
        self.python_executable = config.get("python_executable", "python")
        self.worker_script = config.get(
            "worker_script",
            os.path.join(os.path.dirname(__file__), "androidworld_worker.py"),
        )
        self.startup_timeout = config.get("startup_timeout", 120)
        self.command_timeout = config.get("command_timeout", 120)

        # --- AndroidWorld env kwargs (passed to subprocess worker) ---
        self.console_port = config.get("console_port", 5554)
        self.adb_path = config.get("adb_path", "~/Android/Sdk/platform-tools/adb")
        self.grpc_port = config.get("grpc_port", 8554)
        self.use_launcher = config.get("use_launcher", True)
        self.emulator_setup = config.get("emulator_setup", False)
        self.freeze_datetime = config.get("freeze_datetime", True)
        self.wait_after_reset = config.get("wait_after_reset", 3)
        self.wait_before_eval = config.get("wait_before_eval", 3)

        # --- Episode config ---
        self.max_steps = config.get("max_steps", 30)
        self.step_count = 0
        self.current_goal = ""

        # --- Subprocess handle ---
        self._proc: Optional[subprocess.Popen] = None

    # ================================================================
    # Subprocess Management
    # ================================================================

    def _start_worker(self):
        """Start the subprocess worker if not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return  # Already running

        env = os.environ.copy()
        # Add AndroidWorld directory to PYTHONPATH
        if self.android_world_dir:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = self.android_world_dir + (":" + existing if existing else "")

        cmd = [self.python_executable, self.worker_script]
        logger.info("Starting AndroidWorld worker: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.android_world_dir or None,
            env=env,
            bufsize=0,
        )

        # Verify worker is alive with a ping
        try:
            resp = self._send_command({"cmd": "ping"}, timeout=self.startup_timeout)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Worker ping failed: {resp}")
            logger.info("AndroidWorld worker started successfully (pid=%d)", self._proc.pid)
        except Exception as e:
            self._kill_worker()
            raise RuntimeError(f"Failed to start AndroidWorld worker: {e}") from e

    def _kill_worker(self):
        """Kill the subprocess worker."""
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass
        # Drain stderr for debugging
        try:
            stderr = self._proc.stderr.read()
            if stderr:
                logger.debug("Worker stderr:\n%s", stderr.decode("utf-8", errors="replace")[-2000:])
        except Exception:
            pass
        self._proc = None

    def _send_command(self, cmd_data: dict, timeout: float = None) -> dict:
        """Send a command to the worker and read the response.

        Args:
            cmd_data: Command dict to send as JSON
            timeout: Timeout in seconds (None = use default)

        Returns:
            Response dict from worker

        Raises:
            RuntimeError: If worker dies, times out, or returns error
        """
        if timeout is None:
            timeout = self.command_timeout

        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Worker process is not running")

        # Send command
        line = json.dumps(cmd_data, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"Failed to send command to worker: {e}") from e

        # Read response with timeout
        import select
        start_time = time.time()
        response_line = b""

        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                raise RuntimeError(
                    f"Worker timed out after {timeout}s for command: {cmd_data.get('cmd')}"
                )

            # Check if worker is still alive
            if self._proc.poll() is not None:
                stderr = ""
                try:
                    stderr = self._proc.stderr.read().decode("utf-8", errors="replace")[-1000:]
                except Exception:
                    pass
                raise RuntimeError(
                    f"Worker process died (exit={self._proc.returncode}). stderr: {stderr}"
                )

            # Use select for timeout-aware reading
            try:
                ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 1.0))
            except (ValueError, OSError):
                raise RuntimeError("Worker stdout closed unexpectedly")

            if ready:
                chunk = self._proc.stdout.read1(65536) if hasattr(self._proc.stdout, 'read1') else self._proc.stdout.readline()
                if not chunk:
                    raise RuntimeError("Worker stdout returned empty (process likely dead)")
                response_line += chunk
                if b"\n" in response_line:
                    break

        # Parse response
        line_str = response_line.split(b"\n")[0].decode("utf-8", errors="replace")
        try:
            return json.loads(line_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from worker: {e}\nRaw: {line_str[:500]}") from e

    # ================================================================
    # Observation Processing
    # ================================================================

    @staticmethod
    def _decode_screenshot(obs_dict: dict) -> Optional[Image.Image]:
        """Decode base64 screenshot from worker response to PIL Image."""
        b64 = obs_dict.get("screenshot_b64", "")
        if not b64:
            return None
        try:
            raw = base64.b64decode(b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            logger.warning("Failed to decode screenshot: %s", e)
            return None

    def _process_obs(self, obs_dict: dict) -> Dict[str, Any]:
        """Convert worker observation dict to RLLAVA-compatible format.

        Returns:
            Dict with:
                - "screenshot": PIL.Image or None
                - "ui_elements": list of dicts or None
                - "goal": str
        """
        screenshot = self._decode_screenshot(obs_dict)
        return {
            "screenshot": screenshot,
            "ui_elements": obs_dict.get("ui_elements"),
            "goal": obs_dict.get("goal", self.current_goal),
        }

    # ================================================================
    # BaseEnv Interface
    # ================================================================

    def reset(self, task_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset environment with task configuration.

        Starts the subprocess worker if needed, then sends reset command.

        Args:
            task_config: Task configuration dict. Supports:
                - {"task_name": "ContactsAddContact", "task_params": {...}}
                  Load task from AndroidWorld registry
                - {"task_name": "...", "task_goal": "Add a contact named John"}
                  Load task + override goal text
                - {"task_goal": "Do something on the phone"}
                  Free-form goal without registry task (no evaluation)

        Returns:
            obs: Dict with "screenshot" (PIL.Image), "ui_elements", "goal"
        """
        self.step_count = 0
        self.current_goal = ""

        # Start worker if not running
        self._start_worker()

        # Build env kwargs
        env_kwargs = {
            "console_port": self.console_port,
            "adb_path": self.adb_path,
            "grpc_port": self.grpc_port,
            "use_launcher": self.use_launcher,
            "emulator_setup": self.emulator_setup,
            "freeze_datetime": self.freeze_datetime,
        }

        # Extract task info
        task_name = None
        task_params = None
        task_goal = None
        if task_config:
            task_name = task_config.get("task_name")
            task_params = task_config.get("task_params")
            task_goal = task_config.get("task_goal") or task_config.get("instruction")

        cmd = {
            "cmd": "reset",
            "env_kwargs": env_kwargs,
            "task_name": task_name,
            "task_params": task_params,
            "task_goal": task_goal,
            "wait_after_reset": self.wait_after_reset,
        }

        resp = self._send_command(cmd, timeout=self.startup_timeout)
        if resp.get("status") != "ok":
            raise RuntimeError(f"Reset failed: {resp.get('message', resp)}")

        obs = self._process_obs(resp.get("obs", {}))
        self.current_goal = obs.get("goal", "")
        return obs

    def extract_action(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract action dict from model-generated text.

        Supports multiple formats commonly used by mobile GUI agents:
        1. <action>{"action_type": "click", "x": 100, "y": 200}</action>
        2. {"action_type": "click", ...} direct JSON
        3. <action type="click" x="100" y="200"/>  XML-style
        4. Special actions: DONE, FAIL, WAIT, status(...)

        Args:
            content: Model response text

        Returns:
            Action dict or None if no action found
        """
        if not content:
            return None

        # Pattern 1: <action>JSON</action> tags
        m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if m:
            inner = m.group(1).strip()
            try:
                parsed = json.loads(inner)
                if isinstance(parsed, dict) and "action_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Pattern 2: Direct JSON object with action_type
        m = re.search(r'\{[^{}]*"action_type"\s*:\s*"[^"]*"[^{}]*\}', content)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Pattern 3: XML-style <action type="..." .../> or <action ...>...</action>
        m = re.search(
            r'<action\s+type="(\w+)"([^>]*)(?:>(.*?)</action>|/>)',
            content,
            re.DOTALL,
        )
        if m:
            action = {"action_type": m.group(1)}
            attrs = m.group(2)
            for attr_match in re.findall(r'(\w+)="([^"]*)"', attrs):
                key, val = attr_match
                if key in ("x", "y", "index"):
                    try:
                        action[key] = int(val)
                    except ValueError:
                        action[key] = val
                else:
                    action[key] = val
            if m.group(3):
                action["text"] = m.group(3).strip()
            return action

        # Pattern 4: ```json ... ``` code block
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
                if isinstance(parsed, dict) and "action_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Pattern 5: Special terminal actions
        content_upper = content.upper().strip()
        if "STATUS" in content_upper:
            # Try to extract goal_status
            m_status = re.search(
                r'(?:status|goal_status)\s*[:=]\s*["\']?(\w+)',
                content,
                re.IGNORECASE,
            )
            goal_status = m_status.group(1) if m_status else "complete"
            return {"action_type": "status", "goal_status": goal_status}

        for action_word in ("DONE", "FAIL"):
            if action_word in content_upper:
                return {"action_type": "status", "goal_status": action_word.lower()}

        if "WAIT" in content_upper:
            return {"action_type": "wait"}

        return None

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Execute an action in the AndroidWorld environment.

        Sends the action to the subprocess worker which executes it via
        ADB commands on the Android emulator/device.

        Args:
            action: Action dict with action_type and parameters,
                    or a string that will be parsed

        Returns:
            obs: Dict with "screenshot" (PIL.Image), "ui_elements", etc.
            reward: Step reward (typically 0.0, final reward from evaluate())
            done: Whether episode is finished
            info: Additional information dict
        """
        self.step_count += 1

        if self._proc is None or self._proc.poll() is not None:
            return (
                {"screenshot": None, "ui_elements": None, "error": "Worker not running"},
                0.0,
                True,
                {"error": "worker_dead"},
            )

        # Convert string action to dict if needed
        if isinstance(action, str):
            parsed = self.extract_action(action)
            if parsed is None:
                return (
                    {"screenshot": None, "ui_elements": None, "error": f"Cannot parse action: {action}"},
                    0.0,
                    False,
                    {"error": "parse_failed", "raw_action": action, "step": self.step_count},
                )
            action = parsed

        # Handle terminal status action locally for quick response
        action_type = action.get("action_type", "")

        cmd = {
            "cmd": "step",
            "action": action,
            "wait_to_stabilize": True,
        }

        try:
            resp = self._send_command(cmd, timeout=self.command_timeout)
        except RuntimeError as e:
            logger.error("Step failed: %s", e)
            return (
                {"screenshot": None, "error": str(e)},
                0.0,
                True,
                {"error": str(e), "step": self.step_count},
            )

        if resp.get("status") != "ok":
            error_msg = resp.get("message", "Unknown error")
            logger.warning("Step returned error: %s", error_msg)
            return (
                {"screenshot": None, "error": error_msg},
                0.0,
                False,
                {"error": error_msg, "step": self.step_count},
            )

        obs = self._process_obs(resp.get("obs", {}))
        reward = resp.get("reward", 0.0)
        done = resp.get("done", False)
        info = resp.get("info", {})
        info["step"] = self.step_count

        # Check max steps
        if self.step_count >= self.max_steps:
            done = True
            info["max_steps_reached"] = True

        return obs, reward, done, info

    def evaluate(self) -> float:
        """Evaluate task completion using AndroidWorld's task evaluator.

        Calls task.is_successful(env) in the subprocess worker.

        Returns:
            score: 1.0 if task successful, 0.0 otherwise
        """
        if self._proc is None or self._proc.poll() is not None:
            logger.warning("Cannot evaluate: worker not running")
            return 0.0

        try:
            resp = self._send_command(
                {"cmd": "evaluate", "wait_before_eval": self.wait_before_eval},
                timeout=self.command_timeout,
            )
        except RuntimeError as e:
            logger.error("Evaluate failed: %s", e)
            return 0.0

        if resp.get("status") != "ok":
            logger.warning("Evaluate returned error: %s", resp.get("message"))
            return 0.0

        return float(resp.get("result", 0.0))

    def close(self):
        """Close the environment and terminate the subprocess worker.

        Sends close command to let the worker gracefully tear down the task
        and disconnect from the emulator, then kills the subprocess.
        """
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send_command({"cmd": "close"}, timeout=60)
            except Exception as e:
                logger.warning("Error sending close to worker: %s", e)
        self._kill_worker()
        logger.info("AndroidWorldSubprocessEnv closed.")

    def __del__(self):
        """Ensure worker is cleaned up on garbage collection."""
        try:
            self.close()
        except Exception:
            pass
