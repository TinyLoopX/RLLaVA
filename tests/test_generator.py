"""CPU control-flow tests using fake engines; no training dependencies required.

Run with: python -m unittest discover -s tests -p 'test_generator.py' -v
"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from rllava.roles import Generator


ROOT = Path(__file__).resolve().parents[1]


class Batch:
    def __init__(self, size=1):
        self.size = size
        self.meta_info = {"temperature": 0.7, "n": 2}
        self.batch = {"responses": object(), "rollout_log_probs": object()}

    def __len__(self):
        return self.size

    def to(self, device):
        self.device = device
        return self


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def load_rollout():
    """Load real rollout orchestration with only its external imports stubbed."""
    dist = module("torch.distributed")
    dependencies = {
        "torch": module("torch", distributed=dist),
        "torch.distributed": dist,
        "numpy": module("numpy"),
        "tensordict": module("tensordict", TensorDict=object),
        "rllava.data.protocol": module("rllava.data.protocol", DataProto=Batch),
        "rllava.distribute.dist_utils": module(
            "rllava.distribute.dist_utils",
            CoordinatorError=RuntimeError,
            broadcast_object=lambda value, **kwargs: value,
        ),
        "rllava.rollout.batching": module(
            "rllava.rollout.batching", RolloutBatching=lambda pad: None,
        ),
    }
    spec = importlib.util.spec_from_file_location(
        "rllava.rollout.rollout_manager", ROOT / "rllava/rollout/rollout_manager.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(loaded)
    return loaded.RolloutManager


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(name="fake", n=2)
        self.tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
        self.generator = Generator(self.config, self.tokenizer)

    def test_initialize_and_weight_lifecycle(self):
        engine = Mock(collective_generate=True)
        engine_class = Mock(return_value=engine)
        factory = Mock(return_value=engine_class)
        generation_config = SimpleNamespace(eos_token_id=[2, 3], pad_token_id=4)
        from_pretrained = Mock(return_value=generation_config)
        dependencies = {
            "transformers": module("transformers", GenerationConfig=SimpleNamespace(from_pretrained=from_pretrained)),
            "rllava.engine": module("rllava.engine", EngineFactory=factory),
            "rllava.utils.model_utils": module("rllava.utils.model_utils", print_gpu_memory_usage=Mock()),
        }
        layout = object()
        # Cleanup is safe even if engine initialization has not happened.
        self.generator.close()
        with patch.dict(sys.modules, dependencies):
            self.generator.initialize("model-path", layout)
        from_pretrained.assert_called_once_with("model-path")
        factory.assert_called_once_with("fake")
        engine_class.assert_called_once_with(
            model_name_or_path="model-path", config=self.config,
            tokenizer=self.tokenizer, processor=None, layout=layout,
        )
        self.assertTrue(self.generator.collective_generate)
        policy = object()
        self.generator.load(policy)
        self.generator.offload()
        self.generator.close()
        engine.load.assert_called_once_with(policy)
        engine.offload.assert_called_once_with()
        engine.close.assert_called_once_with()
        prompts = Batch()
        self.generator.prepare_prompts(prompts)
        self.assertEqual(prompts.meta_info, {
            "temperature": 0.7, "n": 2, "eos_token_id": [2, 3], "pad_token_id": 4,
        })

    def test_tokenizer_fallback_preserves_sampling_options(self):
        prompts = Batch()
        self.assertIs(self.generator.prepare_prompts(prompts), prompts)
        self.assertEqual(prompts.meta_info, {
            "temperature": 0.7, "n": 2, "eos_token_id": 1, "pad_token_id": 0,
        })

    def make_rollout(self):
        rollout = load_rollout()(self.config, Mock(), self.tokenizer, None)
        rollout.resources = SimpleNamespace(
            roles={"rollout": SimpleNamespace()}, rank=0,
            coordinator_rank=0, world_size=1, control_group=None,
        )
        rollout.dispatcher = SimpleNamespace(
            scatter=Mock(side_effect=lambda batch, role: batch),
            gather=Mock(side_effect=lambda batch, role: batch),
        )
        output = Batch()
        rollout.generator.engine = Mock(collective_generate=False)
        rollout.generator.engine.prepare.side_effect = lambda batch: batch
        rollout.generator.engine.generate.return_value = output
        return rollout, output

    def test_rollout_preserves_output_and_metadata_through_generator(self):
        rollout, output = self.make_rollout()
        prompts = Batch()
        result = rollout.run(lambda: rollout.generate_sequences(prompts))
        self.assertIs(result, output)
        self.assertIs(result.batch["rollout_log_probs"], output.batch["rollout_log_probs"])
        self.assertEqual(result.device, "cpu")
        self.assertEqual(prompts.meta_info["eos_token_id"], 1)
        rollout.generator.engine.prepare.assert_called_once_with(prompts)
        rollout.generator.engine.generate.assert_called_once_with(prompts)
        rollout.dispatcher.gather.assert_called_once_with(output, "rollout")

    def test_empty_shard_does_not_enter_backend(self):
        rollout, _ = self.make_rollout()
        self.assertEqual(rollout._generate_round(Batch(0)), (None, None))
        rollout.generator.engine.prepare.assert_not_called()
        rollout.generator.engine.generate.assert_not_called()

    def test_prepare_failure_prevents_generate_and_gather(self):
        rollout, _ = self.make_rollout()
        rollout.generator.engine.prepare.side_effect = ValueError("bad input")
        with self.assertRaisesRegex(RuntimeError, "rank 0: ValueError: bad input"):
            rollout.generate_sequences(Batch())
        rollout.generator.engine.generate.assert_not_called()
        rollout.dispatcher.gather.assert_not_called()

    def test_generation_failure_broadcasts_abort(self):
        rollout, _ = self.make_rollout()
        rollout.generator.engine.generate.side_effect = ValueError("backend failed")
        rollout._broadcast = Mock()
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            rollout.run(lambda: rollout.generate_sequences(Batch()))
        self.assertEqual(rollout._broadcast.call_args.args[0][0], "abort")
        rollout.dispatcher.gather.assert_not_called()

    def test_public_packages_do_not_eagerly_import_training_roles(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import rllava.controller, rllava.train, rllava.rollout; "
             "from rllava.roles import Generator; "
             "assert 'rllava.roles.actor' not in sys.modules; "
             "assert 'rllava.train.train' not in sys.modules; assert Generator"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
