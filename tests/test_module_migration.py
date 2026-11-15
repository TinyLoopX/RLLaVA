"""Exercise relocated configuration and dynamic loaders without GPU dependencies."""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_generator import module

ROOT = Path(__file__).resolve().parents[1]


def load_config_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace('.', '/') + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class ConfigurationMigrationTests(unittest.TestCase):
    def setUp(self):
        dist = module('torch.distributed')
        self.modules = patch.dict(sys.modules, {
            'torch': module('torch', distributed=dist),
            'torch.distributed': dist,
            'omegaconf': module('omegaconf', OmegaConf=object, DictConfig=dict),
            'rllava.utils': module('rllava.utils', pkg_version=Mock()),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ('rllava.data.config', 'rllava.utils.config', 'rllava.model.config',
                     'rllava.distribute.resource_manager', 'rllava.engine.inference.config',
                     'rllava.train.config', 'rllava.roles.config', 'rllava.rollout.config'):
            load_config_module(name)
        self.config_cls = load_config_module('rllava.controller.config').RLConfig

    def test_composition_preserves_config_schema_and_derived_sampling_settings(self):
        config = self.config_cls()
        config.data.max_prompt_length = 128
        config.data.max_response_length = 64
        config.algorithm.adv_estimator = 'rloo'
        config.rollout.seed = 'none'
        config.deep_post_init()
        self.assertEqual(set(config.to_dict()), {'data', 'trainer', 'algorithm', 'actor', 'critic', 'ref', 'reward', 'rollout'})
        self.assertEqual(config.rollout.prompt_length, 128)
        self.assertEqual(config.rollout.response_length, 64)
        self.assertEqual(config.rollout.adv_estimator, 'rloo')
        self.assertIsNone(config.rollout.seed)
        self.assertEqual(config.actor.policy_loss.loss_mode, 'vanilla')
        config.rollout.agent = 'gui'
        self.assertEqual(self.config_cls().rollout.agent, 'single_turn')

    def test_sampling_probability_guards_survive_config_split(self):
        config = self.config_cls()
        config.algorithm.prefix_strategy = 'hint'
        with self.assertRaisesRegex(ValueError, 'prefix/demo'):
            config.post_init()
        config.algorithm.prefix_strategy = 'none'
        config.actor.tis_imp_ratio_cap = 2
        config.actor.policy_loss.loss_mode = 'luffy'
        with self.assertRaisesRegex(ValueError, 'vanilla'):
            config.post_init()


class DynamicLoaderMigrationTests(unittest.TestCase):
    def setUp(self):
        self.omega = NS(load=Mock(), to_container=lambda value, **kw: value)
        self.modules = patch.dict(sys.modules, {'omegaconf': module('omegaconf', OmegaConf=self.omega)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_builtin_envs_load_at_new_paths_and_resolve_worker_scripts(self):
        registry = importlib.import_module('rllava.rollout.envs.env_registry')
        for name, cls_name in [('osworld', 'OSWorldSubprocessEnv'), ('androidworld', 'AndroidWorldSubprocessEnv')]:
            with self.subTest(name=name):
                self.omega.load.return_value = {'env': {
                    'class_name': f'rllava.rollout.envs.{name}_subprocess_env.{cls_name}',
                    'config': {'max_steps': 7},
                }}
                env = registry.initialize_env_from_config('unused.yaml')
                self.assertEqual(env.max_steps, 7)
                self.assertIsNone(env._proc)
                self.assertEqual(Path(env.worker_script), ROOT / f'rllava/rollout/envs/{name}_worker.py')
                self.assertTrue(Path(env.worker_script).is_file())

    def test_tool_classes_resolve_without_running_external_services(self):
        registry = importlib.import_module('rllava.rollout.tools.tool_registry')
        base = importlib.import_module('rllava.rollout.tools.base').BaseTool
        for name, cls_name in [('visual_crop_tool', 'VisualCropTool'),
                              ('web_search_tool', 'WebSearchTool'),
                              ('code_execute_tool', 'CodeExecuteTool')]:
            with self.subTest(name=name):
                cls = registry.get_tool_class(f'rllava.rollout.tools.{name}.{cls_name}')
                self.assertTrue(issubclass(cls, base))


if __name__ == '__main__':
    unittest.main()
