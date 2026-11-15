"""Manager control-flow tests with NumPy-backed stand-ins for tensor containers."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from test_generator import module
from rllava.rollout.agents.base import SingleTurnAgent, AgentAction


class Tensor(np.ndarray):
    @classmethod
    def make(cls, value):
        return np.asarray(value).view(cls)

    def size(self, dim):
        return self.shape[dim]

    def dim(self):
        return self.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def unsqueeze(self, dim):
        return np.expand_dims(self, axis=dim)

    def clone(self):
        return self.copy()

    def view(self, *shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], type):
            return super().view(shape[0], **kwargs)
        return self.reshape(*shape)

    @property
    def device(self):
        return 'cpu'


class TensorMap(dict):
    def __init__(self, source, batch_size):
        super().__init__(source)
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size[0]

    def to_dict(self):
        return dict(self)


class Proto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None, size=None):
        size = len(next(iter(batch.values()))) if size is None else size
        self.batch = TensorMap(batch, (size,))
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __len__(self):
        return self.batch.batch_size[0]

    def __getitem__(self, key):
        indices = np.arange(len(self))[key]
        return Proto({k: v[indices] for k, v in self.batch.items()},
                     {k: v[indices] for k, v in self.non_tensor_batch.items()},
                     dict(self.meta_info), len(indices))

    def repeat(self, repeat_times, interleave):
        return self[np.repeat(np.arange(len(self)), repeat_times)]

    def pop(self, batch_keys, non_tensor_batch_keys=(), meta_info_keys=()):
        return Proto({k: self.batch.pop(k) for k in batch_keys if k in self.batch},
                     {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys if k in self.non_tensor_batch},
                     {k: self.meta_info.pop(k) for k in meta_info_keys if k in self.meta_info}, len(self))

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        self.meta_info.update(other.meta_info)
        return self

    @staticmethod
    def concat(batches):
        return Proto({k: np.concatenate([b.batch[k] for b in batches]).view(Tensor) for k in batches[0].batch},
                     {k: np.concatenate([b.non_tensor_batch[k] for b in batches]) for k in batches[0].non_tensor_batch},
                     dict(batches[0].meta_info), sum(map(len, batches)))

    @staticmethod
    def from_single_dict(value):
        return value


ROOT = Path(__file__).resolve().parents[1]


def load_manager():
    dist = module('torch.distributed')
    torch = module('torch', distributed=dist, float32=np.float32,
                   tensor=lambda value, **kw: Tensor.make(value),
                   full=lambda shape, value, **kw: Tensor.make(np.full(shape, value)),
                   cat=lambda tensors, dim: Tensor.make(np.concatenate(tensors, axis=dim)))
    dependencies = {
        'torch': torch, 'torch.distributed': dist,
        'tensordict': module('tensordict', TensorDict=TensorMap),
        'rllava.data.protocol': module('rllava.data.protocol', DataProto=Proto),
        'rllava.distribute.dist_utils': module('rllava.distribute.dist_utils',
            CoordinatorError=RuntimeError, broadcast_object=lambda value, **kw: value),
    }
    spec = importlib.util.spec_from_file_location('rllava.rollout.rollout_manager', ROOT/'rllava/rollout/rollout_manager.py')
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        # Force a fresh batching module so fake imports never escape this context.
        sys.modules.pop('rllava.rollout.batching', None)
        spec.loader.exec_module(loaded)
    return loaded.RolloutManager


def prompts(size=2):
    data = Tensor.make(np.ones((size, 2), dtype=int))
    raw = np.empty(size, dtype=object)
    raw[:] = [[1, 1] for _ in range(size)]
    return Proto({'input_ids': data, 'attention_mask': data.copy(), 'position_ids': data.copy()},
                 {'raw_prompt_ids': raw, 'task': np.arange(size)},
                 {'min_pixels': None, 'max_pixels': None, 'video_fps': 2.0})


class CountingAgent(SingleTurnAgent):
    multi_turn = True
    instances = []

    def start(self, data):
        super().start(data)
        self.turn = 0
        self.instances.append(self)

    def accept_output(self, output, env, tools):
        super().accept_output(output, env, tools)
        self.turn += 1
        return AgentAction('finish' if self.turn > self.data.non_tensor_batch['task'][0] else 'feedback')


class ManagerTests(unittest.TestCase):
    def make_manager(self, agent_cls=SingleTurnAgent):
        config = SimpleNamespace(name='fake', n=2, max_turns=3, env_config_path=None,
            tool_config_path=None, val_override_config={'n': 1, 'temperature': 0},
            min_pixels=None, max_pixels=None, video_fps=2.0, adv_estimator='grpo')
        reward = Mock()
        reward.compute_rewards.side_effect = lambda data: (
            Tensor.make(np.ones((len(data), 1))), {'overall': [1.0] * len(data)})
        manager = load_manager()(config, reward, SimpleNamespace(pad_token_id=0), None, agent_cls=agent_cls)
        self.sizes = []

        def generate(data):
            self.sizes.append(len(data))
            self.assertEqual(data.meta_info['n'], 1)
            result = copy.deepcopy(data)
            result.batch['prompts'] = data.batch['input_ids'].copy()
            result.batch['responses'] = Tensor.make(np.ones((len(data), 1), dtype=int))
            result.batch['response_mask'] = result.batch['responses'].copy()
            result.batch['rollout_log_probs'] = Tensor.make(np.full((len(data), 1), -0.5))
            result.batch['input_ids'] = Tensor.make(np.concatenate([data.batch['input_ids'], result.batch['responses']], axis=-1))
            result.batch['attention_mask'] = Tensor.make(np.ones_like(result.batch['input_ids']))
            result.batch['position_ids'] = result.batch['attention_mask'].copy()
            return result
        manager.generate_sequences = Mock(side_effect=generate)
        manager.run = lambda collect: collect()
        return manager

    def test_training_and_validation_repeat_counts_and_metadata(self):
        manager = self.make_manager()
        result = manager.generate_one_batch(prompts(), global_step=7)
        self.assertEqual(len(result), 4)
        np.testing.assert_array_equal(result.non_tensor_batch['task'], [0, 0, 1, 1])
        self.assertEqual(len(set(result.non_tensor_batch['uid'])), 2)
        self.assertEqual(result.meta_info['global_step'], 7)
        manager.reward.compute_rewards.assert_called_once()
        result = manager.generate_one_batch(prompts(), val=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.meta_info['temperature'], 0)
        manager.reward.compute_rewards.assert_called_once()  # evaluation scores separately

    def test_active_workers_shrink_and_output_order_is_stable(self):
        CountingAgent.instances = []
        manager = self.make_manager(CountingAgent)
        result = manager.generate_one_batch(prompts())
        self.assertEqual(self.sizes, [4, 2])
        self.assertEqual([a.turn for a in CountingAgent.instances], [1, 1, 2, 2])
        self.assertEqual(len({id(a) for a in CountingAgent.instances}), 4)
        np.testing.assert_array_equal(result.non_tensor_batch['task'], [0, 0, 1, 1])

    def test_filter_keeps_whole_groups_and_collect_refills(self):
        manager = self.make_manager()
        manager.configure_sampling(SimpleNamespace(online_filtering=True, filter_key='overall',
                                   filter_low=0.1, filter_high=0.9), 1, 3)
        first = Proto({'reward_overall': Tensor.make([0., 0.]), 'attention_mask': Tensor.make([[1], [1]])},
                      {'uid': np.array(['a', 'a'])})
        second = Proto({'reward_overall': Tensor.make([0., 1.]), 'attention_mask': Tensor.make([[1], [1]])},
                       {'uid': np.array(['b', 'b'])})
        manager.generate_one_batch = Mock(side_effect=lambda data, **kw: manager.filter(data))
        result = manager.collect(iter([first, second]), global_step=4)
        self.assertEqual(manager.generate_one_batch.call_count, 2)
        self.assertEqual(result.non_tensor_batch['uid'].tolist(), ['b', 'b'])
        self.assertEqual(result.meta_info['global_token_num'], [1, 1])

    def test_refill_limit_raises(self):
        manager = self.make_manager()
        manager.configure_sampling(SimpleNamespace(online_filtering=False), 2, 1)
        manager.generate_one_batch = Mock(return_value=prompts(1))
        with self.assertRaisesRegex(RuntimeError, '1/4 responses after 1 attempts'):
            manager.collect(iter([prompts()]))

    def test_generation_failure_closes_all_active_environments(self):
        manager = self.make_manager(CountingAgent)
        manager.config.env_config_path = 'fake.yaml'
        envs = [Mock(), Mock()]
        initialize = Mock(side_effect=envs)
        manager.generate_sequences.side_effect = RuntimeError('generation failed')
        with patch.dict(sys.modules, {'rllava.rollout.envs': module('rllava.rollout.envs', initialize_env_from_config=initialize)}):
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                manager.run_workers(prompts())
        for env in envs:
            env.close.assert_called_once()

    def test_partial_reset_failure_closes_previously_started_workers(self):
        manager = self.make_manager(CountingAgent)
        manager.config.env_config_path = 'fake.yaml'
        envs = [Mock(), Mock()]
        envs[1].reset.side_effect = ValueError('second reset failed')
        initialize = Mock(side_effect=envs)
        with patch.dict(sys.modules, {'rllava.rollout.envs': module('rllava.rollout.envs', initialize_env_from_config=initialize)}):
            with self.assertRaisesRegex(ValueError, 'second reset failed'):
                manager.run_workers(prompts())
        manager.generate_sequences.assert_not_called()
        for env in envs:
            env.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
