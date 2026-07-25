# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.util
import logging
import os
import sys

from omegaconf import OmegaConf
from .base import BaseEnv

logger = logging.getLogger(__name__)


def _get_class(cls_name: str):
    """Load a class by dotted module path **or** ``file_path:ClassName`` syntax.

    Supported formats:
        - ``rllava.ppo.env.deepeyes_env.DeepEyesEnv``  (standard module path)
        - ``./examples/tasks/deepeyes/deepeyes_env.py:DeepEyesEnv``  (file path)
    """
    if ":" in cls_name:
        file_path, class_name = cls_name.rsplit(":", 1)
    elif os.path.sep in cls_name or cls_name.endswith(".py"):
        file_path, class_name = cls_name.rsplit(".", 1) if "." in cls_name else (cls_name, "")
        # Won't work without explicit class name; fall through to module import
        if not class_name or file_path.endswith(".py"):
            raise ValueError(
                f"File-path class_name must use 'path/to/file.py:ClassName' format, got '{cls_name}'"
            )
    else:
        file_path = None
        class_name = cls_name

    if file_path and os.path.exists(file_path):
        mod_name = f"_env_module_{os.path.basename(file_path).replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    # Standard dotted module path
    module_name, class_name = cls_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def initialize_env_from_config(env_config_file: str) -> BaseEnv:
    """Create an env instance from a YAML configuration file.

    Supported config formats
    ------------------------
    **Simple (recommended)** – a single env::

        env:
          class_name: "./examples/tasks/deepeyes/deepeyes_env.py:DeepEyesEnv"
          config:
            min_crop_side: 28

    ``class_name`` accepts either a dotted Python module path or a
    ``file_path:ClassName`` pair (useful for task-local env files that live
    outside of installed packages).

    **Legacy interaction-based** – kept for backward compatibility::

        interaction:
          - class_name: "module.ClassName"
            config: { ... }

    Returns:
        A ``BaseEnv`` (or subclass) instance.
    """
    cfg = OmegaConf.load(env_config_file)

    # ---- simple format: top-level "env" key ----
    if "env" in cfg:
        env_cfg = cfg["env"]
        cls = _get_class(env_cfg["class_name"])
        config = OmegaConf.to_container(env_cfg.get("config", {}), resolve=True)
        env = cls(config=config)
        logger.info("Initialized env '%s'", env_cfg["class_name"])
        return env

    # ---- legacy interaction format ----
    if "interaction" in cfg:
        interaction_map = {}
        for item in cfg["interaction"]:
            cls_name = item["class_name"]
            cls = _get_class(cls_name)
            config = OmegaConf.to_container(item.get("config", {}), resolve=True)
            name = item.get("name") or cls_name.split(".")[-1].lower()
            config["name"] = name
            if name in interaction_map:
                raise ValueError(f"Duplicate interaction name '{name}'.")
            interaction_map[name] = cls(config=config)
            logger.info("Initialized interaction '%s' (%s)", name, cls_name)
        if interaction_map:
            return next(iter(interaction_map.values()))

    raise ValueError(
        f"Env config '{env_config_file}' must contain an 'env' or 'interaction' key."
    )