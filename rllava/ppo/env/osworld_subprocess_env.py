"""
OSWorld Subprocess Environment for RLLAVA.

Uses a subprocess to run OSWorld's DesktopEnv in an isolated Python environment,
communicating via JSON over stdin/stdout pipes. This avoids dependency conflicts
between RLLAVA (torch, transformers, etc.) and OSWorld (gymnasium, desktop_env, etc.).

Architecture:
    ┌──────────────────────┐          JSON/stdin/stdout          ┌──────────────────────────┐
    │  RLLAVA Main Process │  ←──────────────────────────────→  │  osworld_worker.py        │
    │  (OSWorldSubprocEnv) │      reset/step/evaluate/close     │  (DesktopEnv + VM mgmt)   │
    │  BaseEnv interface   │                                     │  Runs in OSWorld conda env│
    └──────────────────────┘                                     └──────────────────────────┘

Usage in config YAML:
    class_name: OSWorldSubprocessEnv
    config:
      osworld_dir: /path/to/OSWorld-main
      python_executable: /path/to/osworld_conda/bin/python
      provider_name: volcengine
      action_space: pyautogui
      max_steps: 30
      screen_size: [1920, 1080]
"""

import os
import re
import io
import json
import base64
import logging
import subprocess
import time
import glob
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base import BaseEnv

logger = logging.getLogger(__name__)


class OSWorldSubprocessEnv(BaseEnv):
    """OSWorld environment running in a subprocess, implementing RLLAVA's BaseEnv.
    
    The subprocess runs osworld_worker.py in OSWorld's Python environment,
    which manages the actual DesktopEnv (VM creation, action execution,
    screenshot capture, evaluation).
    
    This class handles:
    - Subprocess lifecycle management (spawn, communicate, kill)
    - JSON protocol marshalling (serialize actions, deserialize observations)
    - Screenshot decoding (base64 -> PIL.Image)
    - Action extraction from model output text
    - Robust error handling and timeout management
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # --- Subprocess config ---
        raw_osworld_dir = config.get("osworld_dir", "")
        self.osworld_dir = os.path.abspath(raw_osworld_dir) if raw_osworld_dir else ""
        self.python_executable = config.get("python_executable", "python")
        self.env_name = config.get("env_name", "osworld")
        self.worker_script = os.path.abspath(config.get(
            "worker_script",
            os.path.join(os.path.dirname(__file__), "osworld_worker.py"),
        ))
        self.extra_ld_library_paths = config.get("extra_ld_library_paths", [])
        # Prefer direct interpreter launch in concurrent setting; fallback to conda run.
        # If use_conda_run is explicitly configured, respect it.
        use_conda_run = config.get("use_conda_run", None)
        if use_conda_run is None:
            self.use_conda_run = not os.path.isabs(self.python_executable)
        else:
            self.use_conda_run = bool(use_conda_run)
        self.startup_timeout = config.get("startup_timeout", 120)
        self.command_timeout = config.get("command_timeout", 300)
        # Reset resilience: retry failed reset by restarting worker.
        self.reset_max_retries = int(config.get("reset_max_retries", 3))
        self.reset_retry_backoff = float(config.get("reset_retry_backoff", 5.0))
        self.restart_worker_on_reset_failure = bool(config.get("restart_worker_on_reset_failure", True))

        # --- DesktopEnv kwargs (passed to subprocess worker) ---
        self.provider_name = config.get("provider_name", "vmware")
        self.region = config.get("region", None)
        self.path_to_vm = config.get("path_to_vm", None)
        self.snapshot_name = config.get("snapshot_name", "init_state")
        self.action_space = config.get("action_space", "pyautogui")
        self.screen_size = config.get("screen_size", [1920, 1080])
        self.headless = config.get("headless", True)
        self.require_a11y_tree = config.get("require_a11y_tree", False)
        self.require_terminal = config.get("require_terminal", False)
        self.os_type = config.get("os_type", "Ubuntu")
        self.wait_after_reset = config.get("wait_after_reset", 5)
        self.sleep_after_execution = config.get("sleep_after_execution", 3)
        self.wait_before_eval = config.get("wait_before_eval", 5)

        # --- Episode config ---
        self.max_steps = config.get("max_steps", 30)
        self.step_count = 0
        self.task_config = None

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
        # Add OSWorld directory to PYTHONPATH so imports work
        if self.osworld_dir:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = self.osworld_dir + (":" + existing if existing else "")
        # Some osworld envs ship torch+cu124. Ensure nvjitlink/cusparse libs are visible
        # to avoid ImportError like "__nvJitLinkComplete_12_4".
        ld_paths: List[str] = []
        if os.path.isabs(self.python_executable):
            env_prefix = os.path.dirname(os.path.dirname(self.python_executable))
            nvidia_dirs = glob.glob(
                os.path.join(env_prefix, "lib", "python*", "site-packages", "nvidia")
            )
            for nvidia_dir in nvidia_dirs:
                for rel in ("nvjitlink/lib", "cusparse/lib"):
                    path = os.path.join(nvidia_dir, rel)
                    if os.path.isdir(path):
                        ld_paths.append(path)
        for p in self.extra_ld_library_paths:
            if isinstance(p, str) and p and os.path.isdir(p):
                ld_paths.append(p)
        if ld_paths:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing_ld] if existing_ld else []))

        if self.use_conda_run:
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.env_name,
                self.python_executable,
                self.worker_script,
            ]
        else:
            cmd = [self.python_executable, self.worker_script]
        logger.info("Starting OSWorld worker: %s", " ".join(cmd))

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.osworld_dir or None,
            env=env,
            bufsize=0,  # Unbuffered for responsive communication
        )

        # Verify worker is alive with a ping
        try:
            resp = self._send_command({"cmd": "ping"}, timeout=self.startup_timeout)
            if resp.get("status") != "ok":
                raise RuntimeError(f"Worker ping failed: {resp}")
            logger.info("OSWorld worker started successfully (pid=%d)", self._proc.pid)
        except Exception as e:
            self._kill_worker()
            raise RuntimeError(f"Failed to start OSWorld worker: {e}") from e

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
        # Drain any remaining stderr for debugging
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
        response_buffer = b""
        last_bad_line = None

        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                detail = f"\nLast non-JSON line: {last_bad_line}" if last_bad_line else ""
                raise RuntimeError(
                    f"Worker timed out after {timeout}s for command: {cmd_data.get('cmd')}{detail}"
                )

            # Check if worker is still alive
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read().decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(
                    f"Worker process died (exit={self._proc.returncode}). stderr: {stderr}"
                )

            # Use select for timeout-aware reading
            try:
                ready, _, _ = select.select([self._proc.stdout], [], [], min(remaining, 1.0))
            except (ValueError, OSError):
                # stdout was closed
                raise RuntimeError("Worker stdout closed unexpectedly")

            if ready:
                chunk = self._proc.stdout.read1(65536) if hasattr(self._proc.stdout, 'read1') else self._proc.stdout.readline()
                if not chunk:
                    if self._proc.poll() is not None:
                        stderr = self._proc.stderr.read().decode("utf-8", errors="replace")[-1000:]
                        raise RuntimeError(
                            f"Worker stdout returned empty and process exited (exit={self._proc.returncode}). "
                            f"stderr: {stderr}"
                        )
                    raise RuntimeError("Worker stdout returned empty (process likely dead)")
                response_buffer += chunk

                while b"\n" in response_buffer:
                    raw_line, response_buffer = response_buffer.split(b"\n", 1)
                    line_str = raw_line.decode("utf-8", errors="replace").strip()
                    if not line_str:
                        continue

                    try:
                        resp = json.loads(line_str)
                    except json.JSONDecodeError:
                        last_bad_line = line_str[:500]
                        logger.warning(
                            "Skipping non-JSON worker stdout line for cmd '%s': %s",
                            cmd_data.get("cmd"),
                            last_bad_line,
                        )
                        continue

                    if isinstance(resp, dict):
                        return resp

                    last_bad_line = line_str[:500]
                    logger.warning(
                        "Skipping non-dict worker stdout line for cmd '%s': %s",
                        cmd_data.get("cmd"),
                        last_bad_line,
                    )

    # ================================================================
    # Observation Processing
    # ================================================================

    @staticmethod
    def _decode_screenshot(obs_dict: dict) -> Optional[Image.Image]:
        """Decode base64 screenshot from worker response to PIL Image."""
        b64 = obs_dict.get("screenshot_b64", "")
        if not b64:
            return None
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _process_obs(self, obs_dict: dict) -> Dict[str, Any]:
        """Convert worker observation dict to RLLAVA-compatible format.
        
        Returns:
            Dict with:
                - "screenshot": PIL.Image or None
                - "instruction": str
                - "accessibility_tree": str or None
        """
        screenshot = self._decode_screenshot(obs_dict)
        return {
            "screenshot": screenshot,
            "instruction": obs_dict.get("instruction", ""),
            "accessibility_tree": obs_dict.get("accessibility_tree"),
        }

    # ================================================================
    # BaseEnv Interface
    # ================================================================

    def reset(self, task_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset environment with task configuration.
        
        Starts the subprocess worker if needed, then sends reset command
        with DesktopEnv kwargs and task configuration.
        
        Args:
            task_config: OSWorld task configuration dict containing:
                - id: task identifier
                - instruction: natural language instruction
                - config: setup/evaluator configuration
                
        Returns:
            obs: Dict with "screenshot" (PIL.Image), "instruction" (str), etc.
        """
        self.task_config = task_config
        self.step_count = 0

        # Build DesktopEnv kwargs
        env_kwargs = {
            "provider_name": self.provider_name,
            "action_space": self.action_space,
            "screen_size": list(self.screen_size),
            "headless": self.headless,
            "require_a11y_tree": self.require_a11y_tree,
            "require_terminal": self.require_terminal,
            "os_type": self.os_type,
        }
        if self.region is not None:
            env_kwargs["region"] = self.region
        if self.path_to_vm is not None:
            env_kwargs["path_to_vm"] = self.path_to_vm
        if self.snapshot_name:
            env_kwargs["snapshot_name"] = self.snapshot_name

        cmd = {
            "cmd": "reset",
            "task_config": task_config,
            "env_kwargs": env_kwargs,
            "wait_after_reset": self.wait_after_reset,
        }

        last_error: Optional[str] = None
        for attempt in range(1, max(self.reset_max_retries, 1) + 1):
            try:
                # Start worker if not running
                self._start_worker()

                resp = self._send_command(cmd, timeout=self.startup_timeout)
                if resp.get("status") != "ok":
                    raise RuntimeError(f"Reset failed: {resp.get('message', resp)}")
                return self._process_obs(resp.get("obs", {}))
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "OSWorld reset attempt %d/%d failed: %s",
                    attempt,
                    self.reset_max_retries,
                    last_error,
                )
                if self.restart_worker_on_reset_failure:
                    self._kill_worker()
                if attempt < self.reset_max_retries:
                    # Linear backoff to reduce startup collision pressure.
                    time.sleep(self.reset_retry_backoff * attempt)

        raise RuntimeError(
            f"Reset failed after {self.reset_max_retries} attempts: {last_error}"
        )

    def extract_action(self, content: str) -> Optional[str]:
        """Extract action from model-generated text.
        
        Supports multiple formats commonly used by GUI agents:
        1. <action>pyautogui.click(100, 200)</action>
        2. ```action\\nclick(100, 200)\\n``` (or python code block)
        3. Direct pyautogui command
        4. Simple GUI action syntax: click/type/scroll/hotkey/wait/done
        5. Special actions: DONE, FAIL, WAIT
        
        Args:
            content: Model response text
            
        Returns:
            Action string or None if no action found
        """
        if not content:
            return None
        content = content.strip()

        # Pattern 1: <action>...</action> tags
        m = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if m:
            action = self._normalize_action(m.group(1))
            if action is not None:
                return action

        # Pattern 2: code block (```action / ```python / ```)
        m = re.search(r"```(?:python|action)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if m:
            code = m.group(1).strip()
            action = self._normalize_action(code)
            if action is not None:
                return action

        # Pattern 3: Direct pyautogui command (possibly multiple lines)
        lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("pyautogui.") or stripped.startswith("import pyautogui"):
                lines.append(stripped)
        if lines:
            return "\n".join(lines)

        # Pattern 4: Single-line simple action syntax
        for line in content.split("\n"):
            action = self._normalize_action(line)
            if action is not None:
                return action

        # Pattern 5: Special terminal actions
        for action in ("DONE", "FAIL", "WAIT"):
            if action in content.upper():
                return action

        return None

    def _normalize_action(self, action: str) -> Optional[str]:
        """Normalize action string into pyautogui/special action format."""
        if not action:
            return None
        action = action.strip().rstrip(";")
        if not action:
            return None

        if action.startswith("pyautogui.") or action.startswith("import pyautogui"):
            return action

        converted = self._convert_simple_action(action)
        if converted is not None:
            return converted

        if "\n" in action:
            normalized_lines = []
            for line in action.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("pyautogui.") or line.startswith("import pyautogui"):
                    normalized_lines.append(line)
                    continue
                converted_line = self._convert_simple_action(line)
                if converted_line is None:
                    return None
                normalized_lines.append(converted_line)
            if normalized_lines:
                return "\n".join(normalized_lines)
        return None

    def _convert_simple_action(self, action: str) -> Optional[str]:
        """Convert click/type/scroll/hotkey/wait/done syntax to executable action."""
        action = re.sub(r"^action\s*:\s*", "", action.strip(), flags=re.IGNORECASE)
        if not action:
            return None

        upper = action.upper()
        if upper in ("DONE", "FAIL", "WAIT"):
            return upper
        if re.fullmatch(r"done\s*\(\s*\)", action, flags=re.IGNORECASE):
            return "DONE"
        if re.fullmatch(r"fail\s*\(\s*\)", action, flags=re.IGNORECASE):
            return "FAIL"
        if re.fullmatch(r"wait\s*\(\s*\)", action, flags=re.IGNORECASE):
            return "WAIT"

        m = re.fullmatch(r"wait\s*\(\s*([0-9]*\.?[0-9]+)\s*\)", action, flags=re.IGNORECASE)
        if m:
            seconds = float(m.group(1))
            return f"pyautogui.sleep({seconds})"

        m = re.fullmatch(
            r"click\s*\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
            action,
            flags=re.IGNORECASE,
        )
        if m:
            x = int(float(m.group(1)))
            y = int(float(m.group(2)))
            return f"pyautogui.click({x}, {y})"

        m = re.fullmatch(r"type\s*\(\s*(.*?)\s*\)", action, flags=re.IGNORECASE)
        if m:
            text_arg = m.group(1).strip()
            if not text_arg:
                return None
            if not (
                (text_arg.startswith("'") and text_arg.endswith("'"))
                or (text_arg.startswith('"') and text_arg.endswith('"'))
            ):
                text_arg = repr(text_arg)
            return f"pyautogui.write({text_arg})"

        m = re.fullmatch(r"scroll\s*\(\s*(.*?)\s*\)", action, flags=re.IGNORECASE)
        if m:
            direction = m.group(1).strip().strip("'\"").lower()
            if direction == "up":
                return "pyautogui.scroll(500)"
            if direction == "down":
                return "pyautogui.scroll(-500)"
            if direction == "left":
                return "pyautogui.hscroll(-500)"
            if direction == "right":
                return "pyautogui.hscroll(500)"
            try:
                amount = int(float(direction))
                return f"pyautogui.scroll({amount})"
            except ValueError:
                return None

        m = re.fullmatch(r"hotkey\s*\(\s*(.*?)\s*\)", action, flags=re.IGNORECASE)
        if m:
            args = m.group(1).strip()
            if not args:
                return None
            keys = []
            for key in args.split(","):
                normalized_key = key.strip().strip("'\"")
                if normalized_key:
                    keys.append(normalized_key.lower())
            if not keys:
                return None
            quoted_keys = ", ".join(repr(key) for key in keys)
            return f"pyautogui.hotkey({quoted_keys})"

        return None

    def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Execute an action in the OSWorld environment.
        
        Sends the action to the subprocess worker which executes it via
        the VM's PythonController (HTTP POST to pyautogui on the VM).
        
        Args:
            action: Action string (pyautogui command) or special action
            
        Returns:
            obs: Dict with "screenshot" (PIL.Image), etc.
            reward: Step reward (typically 0.0, final reward from evaluate())
            done: Whether episode is finished
            info: Additional information dict
        """
        self.step_count += 1

        # Handle special actions locally
        if isinstance(action, str) and action.upper() in ("DONE", "FAIL"):
            return (
                {"screenshot": None, "instruction": ""},
                0.0,
                True,
                {"action": action.upper(), "step": self.step_count},
            )

        cmd = {
            "cmd": "step",
            "action": action if isinstance(action, str) else str(action),
            "sleep_after_execution": self.sleep_after_execution,
        }

        resp = self._send_command(cmd, timeout=self.command_timeout)

        if resp.get("status") != "ok":
            raise RuntimeError(f"Step failed: {resp.get('message', resp)}")

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
        """Evaluate task completion using OSWorld's built-in evaluator.
        
        Sends evaluate command to the subprocess worker, which calls
        DesktopEnv.evaluate() to run the task's metric functions.
        
        Returns:
            score: Float between 0.0 and 1.0 indicating task completion
        """
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Cannot evaluate: worker not running")

        resp = self._send_command(
            {"cmd": "evaluate", "wait_before_eval": self.wait_before_eval},
            timeout=self.command_timeout,
        )

        if resp.get("status") != "ok":
            raise RuntimeError(f"Evaluate failed: {resp.get('message', resp)}")

        return float(resp.get("result", 0.0))

    def close(self):
        """Close the environment and terminate the subprocess worker.
        
        Sends a close command to let the worker gracefully shut down
        the VM, then kills the subprocess.
        """
        if self._proc is not None and self._proc.poll() is None:
            self._send_command({"cmd": "close"}, timeout=60)
        self._kill_worker()
        logger.info("OSWorldSubprocessEnv closed.")
