"""Experience semantics with NumPy stand-ins; optional real TensorDict smoke test.

The control-flow suite runs without the GPU training stack. The final test uses
real CPU torch/TensorDict/DataProto when torch and tensordict are installed.
"""
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from test_generator import module
from test_rollout_manager import Proto, Tensor

ROOT = Path(__file__).resolve().parents[1]


class Values(Tensor):
    def bool(self):
        return self.astype(bool)

    def any(self, dim=None, **kwargs):
        return np.asarray(self).any(axis=dim, **kwargs)


class ExperienceProto(Proto):
    @classmethod
    def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):
        return cls(tensors, {k: np.asarray(v, dtype=object) for k, v in (non_tensors or {}).items()}, meta_info)

    @staticmethod
    def concat(samples):
        return ExperienceProto.from_dict(
            {k: Values.make(np.concatenate([s.batch[k] for s in samples])) for k in samples[0].batch},
            {k: np.concatenate([s.non_tensor_batch[k] for s in samples]) for k in samples[0].non_tensor_batch},
            samples[0].meta_info,
        )


def load_experience(real=False):
    if real:
        import torch
        # Load the real DataProto without importing unrelated model/dataset
        # packages through their eager __init__ modules.
        dependencies = {
            'rllava.utils.device': module('rllava.utils.device', get_torch_device=lambda: torch.cpu),
            'rllava.utils.py_functional': module('rllava.utils.py_functional', union_two_dict=lambda a, b: a | b),
        }
        with patch.dict(sys.modules, dependencies):
            spec = importlib.util.spec_from_file_location('rllava.data.protocol', ROOT / 'rllava/data/protocol.py')
            source = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = source
            spec.loader.exec_module(source)
            proto = source.DataProto
    else:
        torch = module('torch', Tensor=Values, equal=np.array_equal, isfinite=np.isfinite,
                       tensor=lambda value, **kw: Values.make(np.asarray(value, dtype=kw.get('dtype'))),
                       cat=lambda tensors, dim: Values.make(np.concatenate(tensors, axis=dim)),
                       full=lambda shape, value, **kw: Values.make(np.full(shape, value, dtype=kw.get('dtype'))),
                       **{name: getattr(np, name) for name in ('int8', 'int16', 'int32', 'int64', 'uint8')})
        proto = ExperienceProto
    dependencies = {'torch': torch, 'rllava.data.protocol': module('rllava.data.protocol', DataProto=proto)}
    with patch.dict(sys.modules, dependencies):
        for name in list(sys.modules):
            if name.startswith('rllava.experience.'):
                del sys.modules[name]
        package = importlib.import_module('rllava.experience')
        api = {name: getattr(package, name) for name in package.__all__}
    return SimpleNamespace(**api, DataProto=proto, tensor=torch.tensor)


def object_row(value):
    result = np.empty(1, dtype=object)
    result[0] = value
    return result


def sample(api, uid='a', index=0, prompt=2, version=0, mrope=False):
    t = api.tensor
    prompts = list(range(10, 10 + prompt))
    positions = list(range(prompt + 3))
    if mrope:
        positions = [positions, [v + 20 for v in positions], [v + 40 for v in positions]]
    tensors = {
        'prompts': t([prompts]), 'responses': t([[30, 31, 0]]),
        'input_ids': t([prompts + [30, 31, 0]]),
        'attention_mask': t([[1] * prompt + [1, 1, 0]]),
        'position_ids': t([positions]),
        'rollout_log_probs': t([[-0.2, -0.3, 0.]]),
        'token_level_scores': t([[0., 2., 0.]]),
        'rollout_temperature': t([0.8]), 'policy_version': t([version]), 'turn_count': t([index + 1]),
    }
    for key in ('response_mask', 'action_mask', 'loss_mask', 'rollout_logprob_mask'):
        tensors[key] = t([[1, 1, 0]])
    non_tensors = {key: object_row(value) for key, value in {
        'uid': uid, 'trajectory_id': f'{uid}:traj:{index}', 'sample_id': f'{uid}:sample:{index}',
        'sample_source': 'policy', 'termination_reason': 'eos',
        'multi_modal_data': {'image': [Image.new('RGB', (2, 2), 'red')], 'nested': t([5])},
        'task': {'ground_truth': ['answer']}, 'multi_modal_inputs': {'cached': t([999])},
    }.items()}
    # Deliberately exercise expanded ndarray rows with different prompt widths.
    non_tensors['raw_prompt_ids'] = np.asarray([prompts], dtype=object)
    return api.DataProto.from_dict(tensors=tensors, non_tensors=non_tensors,
        meta_info={'schema_version': 1, 'experience_mode': 'last_response',
                   'min_pixels': None, 'max_pixels': None, 'video_fps': 2., 'global_token_num': [999]})


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.api = load_experience()
        self.maker = self.api.ExperienceMaker(pad_token_id=0)
        self.queue = self.api.SampleQueue(self.api.ExperienceConfig(max_groups=3))

    def groups(self, *uids, version=0):
        return self.maker.build_groups([sample(self.api, uid, i, prompt=2 + i, version=version)
                                       for uid in uids for i in range(2)], expected_trajectories=2)

    def assertArray(self, actual, expected):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_config_and_empty_contract(self):
        self.assertEqual(self.maker.build_groups([], expected_trajectories=2), [])
        self.assertEqual(self.queue.pop_groups(1, policy_version=0), [])
        self.queue.put_groups([])
        for config in ({'max_groups': 0}, {'max_groups': True}, {'experience_mode': 'per_turn'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.api.ExperienceConfig(**config)
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.maker.make_experience([])
        with self.assertRaises(ValueError):
            self.maker.build_groups([], expected_trajectories=0)

    def test_grouping_preserves_order_and_original_lengths(self):
        records = [sample(self.api, 'b', 0, 2), sample(self.api, 'a', 0, 5),
                   sample(self.api, 'b', 1, 3), sample(self.api, 'a', 1, 7)]
        groups = self.maker.build_groups(records, expected_trajectories=2)
        self.assertEqual([g.group_id for g in groups], ['b', 'a'])
        self.assertEqual([[s.batch['prompts'].shape[1] for s in g.samples] for g in groups], [[2, 3], [5, 7]])
        self.queue.put_groups(groups)
        returned = self.queue.pop_groups(2, policy_version=0)
        self.assertEqual([[s.batch['prompts'].shape[1] for s in g.samples] for g in returned], [[2, 3], [5, 7]])

    def test_handoff_keeps_sample_tensor_and_nested_object_references(self):
        records = [sample(self.api, index=i) for i in range(2)]
        group = self.maker.build_groups(records, expected_trajectories=2)[0]
        self.assertIs(group.samples[0], records[0])
        self.assertNotIn('multi_modal_inputs', records[0].non_tensor_batch)
        self.queue.put_groups([group])
        returned = self.queue.pop_groups(1, policy_version=0)[0]
        self.assertIs(returned, group)
        packed = self.maker.make_experience([returned])
        self.assertIs(packed.non_tensor_batch['multi_modal_data'][0],
                      records[0].non_tensor_batch['multi_modal_data'][0])
        self.assertNotIn('multi_modal_inputs', packed.non_tensor_batch)
        # Batch statistics are rebuilt in a new metadata dict, not deep-copied.
        self.assertEqual(packed.meta_info['global_token_num'], [4, 4])
        self.assertEqual(records[0].meta_info['global_token_num'], [999])

    def test_prompt_padding_preserves_responses_positions_and_group_membership(self):
        for mrope in (False, True):
            with self.subTest(mrope=mrope):
                originals = [sample(self.api, index=i, prompt=2 + i * 3, mrope=mrope) for i in range(2)]
                groups = self.maker.build_groups(originals, expected_trajectories=2)
                packed = self.maker.make_experience(groups)
                self.assertArray(packed.batch['prompts'][0], [10, 11, 0, 0, 0])
                self.assertArray(packed.batch['input_ids'][0], [10, 11, 0, 0, 0, 30, 31, 0])
                self.assertArray(packed.batch['attention_mask'][0], [1, 1, 0, 0, 0, 1, 1, 0])
                for i, source in enumerate(originals):
                    for key in ('responses', 'response_mask', 'action_mask', 'loss_mask',
                                'rollout_logprob_mask', 'rollout_log_probs', 'token_level_scores'):
                        self.assertArray(packed.batch[key][i], source.batch[key][0])
                    self.assertArray(packed.batch['position_ids'][i, ..., -3:],
                                     source.batch['position_ids'][0, ..., -3:])
                    self.assertArray(packed.non_tensor_batch['raw_prompt_ids'][i],
                                     source.non_tensor_batch['raw_prompt_ids'][0])
                self.assertEqual(packed.meta_info['global_token_num'], [4, 7])
                self.assertEqual(list(packed.non_tensor_batch['uid']), ['a', 'a'])
                self.assertEqual(groups[0].samples[0].batch['prompts'].shape, (1, 2))
                self.assertIs(packed.non_tensor_batch['task'][0],
                              groups[0].samples[0].non_tensor_batch['task'][0])

    def test_invalid_protocol_fields_are_rejected(self):
        mutations = {
            'missing': lambda s: s.batch.pop('rollout_log_probs'),
            'schema': lambda s: s.meta_info.update(schema_version=2),
            'mode': lambda s: s.meta_info.update(experience_mode='per_turn'),
            'metadata': lambda s: s.meta_info.pop('video_fps'),
            'demo': lambda s: s.non_tensor_batch['sample_source'].__setitem__(0, 'demonstration'),
            'identity': lambda s: s.non_tensor_batch['trajectory_id'].__setitem__(0, ''),
            'negative version': lambda s: s.batch['policy_version'].__setitem__(0, -1),
            'float version': lambda s: s.batch.__setitem__('policy_version', self.api.tensor([0.])),
            'turn': lambda s: s.batch['turn_count'].__setitem__(0, 0),
            'alignment': lambda s: s.batch['input_ids'].__setitem__((0, -2), 99),
            'loss mask': lambda s: s.batch['loss_mask'].__setitem__((0, 0), 0),
            'nonbinary': lambda s: s.batch['response_mask'].__setitem__((0, 0), 2),
            'nan probability': lambda s: s.batch['rollout_log_probs'].__setitem__((0, 0), float('nan')),
            'positive probability': lambda s: s.batch['rollout_log_probs'].__setitem__((0, 0), .1),
            'padded probability': lambda s: s.batch['rollout_log_probs'].__setitem__((0, -1), -.1),
            'padded score': lambda s: s.batch['token_level_scores'].__setitem__((0, -1), 1),
            'temperature': lambda s: s.batch['rollout_temperature'].__setitem__(0, float('inf')),
            'derived': lambda s: s.batch.__setitem__('advantages', self.api.tensor([[0., 0., 0.]])),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                record = sample(self.api)
                mutate(record)
                self.api.validate_sample(record)

    def test_empty_responses_and_noncontiguous_masks_are_rejected(self):
        for mask in ([0, 0, 0], [1, 0, 1]):
            record = sample(self.api)
            for key in ('response_mask', 'action_mask', 'loss_mask', 'rollout_logprob_mask'):
                record.batch[key] = self.api.tensor([mask])
            record.batch['attention_mask'] = self.api.tensor([[1, 1] + mask])
            with self.assertRaisesRegex(ValueError, 'valid token|right padding'):
                self.api.validate_sample(record)

    def test_incomplete_duplicate_and_mixed_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.maker.build_groups([sample(self.api)], expected_trajectories=2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.maker.build_groups([sample(self.api), sample(self.api)], expected_trajectories=2)
        records = [sample(self.api, index=i, version=i) for i in range(2)]
        with self.assertRaisesRegex(ValueError, 'versions'):
            self.maker.build_groups(records, expected_trajectories=2)
        records[1].batch['policy_version'][0] = 0
        records[1].batch['rollout_temperature'][0] = 0
        with self.assertRaisesRegex(ValueError, 'temperatures'):
            self.maker.build_groups(records, expected_trajectories=2)

    def test_packing_rejects_mixed_metadata_versions_layout_and_identities(self):
        for change, message in (
            (lambda g: g.samples[0].meta_info.update(video_fps=4.), 'metadata'),
            (lambda g: g.samples[0].non_tensor_batch.pop('task'), 'Non-tensor'),
            (lambda g: g.samples[0].batch.__setitem__('extra', self.api.tensor([1])), 'Tensor fields'),
            (lambda g: g.samples[0].non_tensor_batch['sample_id'].__setitem__(0, 'a:sample:0'), 'Duplicate'),
        ):
            with self.subTest(message=message):
                groups = self.groups('a', 'b')
                change(groups[1])
                with self.assertRaisesRegex(ValueError, message):
                    self.maker.make_experience(groups)
        with self.assertRaisesRegex(ValueError, 'versions'):
            self.maker.make_experience(self.groups('a') + self.groups('b', version=1))
        groups = self.groups('a', 'b')
        for record in groups[1].samples:
            record.batch['rollout_temperature'][0] = 0.5
        with self.assertRaisesRegex(ValueError, 'temperatures'):
            self.maker.make_experience(groups)
        groups = self.groups('a')
        with self.assertRaisesRegex(ValueError, 'Duplicate group'):
            self.maker.make_experience(groups + groups)
        # An otherwise valid record with a different fixed response width.
        records = [sample(self.api, index=i) for i in range(2)]
        for key in ('responses', 'response_mask', 'action_mask', 'loss_mask', 'rollout_logprob_mask',
                    'rollout_log_probs', 'token_level_scores', 'input_ids', 'attention_mask', 'position_ids'):
            records[1].batch[key] = records[1].batch[key][..., :-1]
        groups = self.maker.build_groups(records, expected_trajectories=2)
        with self.assertRaisesRegex(ValueError, 'response widths'):
            self.maker.make_experience(groups)

    def test_fifo_and_nonblocking_all_or_nothing_consumption(self):
        self.queue.put_groups(self.groups('a', 'b'))
        self.assertEqual(self.queue.remaining_capacity, 1)
        self.assertEqual(self.queue.pop_groups(3, policy_version=0), [])
        self.assertEqual(len(self.queue), 2)
        self.assertEqual([g.group_id for g in self.queue.pop_groups(1, policy_version=0)], ['a'])
        self.assertEqual([g.group_id for g in self.queue.pop_groups(1, policy_version=0)], ['b'])
        self.assertEqual(len(self.queue), 0)
        for count in (0, -1):
            with self.assertRaises(ValueError):
                self.queue.pop_groups(count, policy_version=0)

    def test_enqueue_failures_are_atomic_and_do_not_evict_or_reserve_ids(self):
        self.queue.put_groups(self.groups('a'))
        with self.assertRaises(self.api.QueueFullError):
            self.queue.put_groups(self.groups('b', 'c', 'd'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.queue.put_groups(self.groups('b', 'a'))
        with self.assertRaisesRegex(ValueError, 'policy_version'):
            self.queue.put_groups(self.groups('b') + self.groups('c', version=1))
        self.assertEqual(len(self.queue), 1)
        self.queue.put_groups(self.groups('b', 'c'))
        self.assertEqual([g.group_id for g in self.queue.pop_groups(3, policy_version=0)], ['a', 'b', 'c'])

    def test_dedup_covers_queued_data_and_version_advance_discards_old_groups(self):
        self.queue.put_groups(self.groups('a', 'b'))
        with self.assertRaisesRegex(ValueError, 'Duplicate group'):
            self.queue.put_groups(self.groups('a'))
        for field in ('trajectory_id', 'sample_id'):
            groups = self.groups('c')
            groups[0].samples[0].non_tensor_batch[field][0] = f'a:{"traj" if field == "trajectory_id" else "sample"}:0'
            with self.assertRaisesRegex(ValueError, f'Duplicate {field}'):
                self.queue.put_groups(groups)
        consumed = self.queue.pop_groups(1, policy_version=0)
        # The caller owns consumed data; the queue keeps no historical tombstones.
        self.queue.put_groups(consumed)
        self.assertEqual(self.queue.discard_stale(0), 0)
        self.assertEqual(self.queue.discard_stale(1), 2)
        self.assertEqual(self.queue.policy_version, 1)
        with self.assertRaisesRegex(ValueError, 'roll back'):
            self.queue.discard_stale(0)
        with self.assertRaisesRegex(ValueError, 'policy_version'):
            self.queue.put_groups(self.groups('a'))
        with self.assertRaisesRegex(ValueError, 'policy_version'):
            self.queue.pop_groups(1, policy_version=0)
        self.queue.put_groups(self.groups('a', version=1))
        self.assertEqual(len(self.queue), 1)

    def test_checkpoint_contains_fifo_groups_without_copies_or_history(self):
        self.queue.put_groups(self.groups('a', 'b', 'c'))
        self.queue.pop_groups(1, policy_version=0)
        state = self.queue.state_dict()
        self.assertEqual(set(state), {'schema_version', 'experience_mode', 'policy_version', 'groups'})
        restored = self.api.SampleQueue()
        restored.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            restored.put_groups(self.groups('b'))
        output = restored.pop_groups(2, policy_version=0)
        self.assertEqual([g.group_id for g in output], ['b', 'c'])
        self.assertIs(output[0], state['groups'][0])
        self.assertIs(self.queue.pop_groups(1, policy_version=0)[0], output[0])

    def test_invalid_checkpoint_does_not_modify_queue(self):
        self.queue.put_groups(self.groups('a'))
        changes = (
            lambda s: s.update(schema_version=999),
            lambda s: s.update(policy_version=1),
            lambda s: s['groups'].append(s['groups'][0]),
            lambda s: s.update(experience_mode='per_turn'),
        )
        for mutate in changes:
            state = self.queue.state_dict()
            mutate(state)
            with self.assertRaises(ValueError):
                self.queue.load_state_dict(state)
            self.assertEqual(len(self.queue), 1)
            self.assertEqual(self.queue.policy_version, 0)
        small = self.api.SampleQueue(self.api.ExperienceConfig(max_groups=1), policy_version=2)
        self.queue.put_groups(self.groups('b'))
        with self.assertRaises(self.api.QueueFullError):
            small.load_state_dict(self.queue.state_dict())
        self.assertEqual(small.policy_version, 2)
        self.assertEqual(len(small), 0)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('tensordict'),
                     'Real CPU integration requires torch and tensordict')
class RealTensorDictTests(unittest.TestCase):
    def test_variable_length_roundtrip_preserves_tensor_ownership(self):
        api = load_experience(real=True)
        maker = api.ExperienceMaker(0)
        records = [sample(api, index=i, prompt=2 + i * 3, mrope=True) for i in range(2)]
        groups = maker.build_groups(records, expected_trajectories=2)
        queue = api.SampleQueue()
        self.assertIs(groups[0].samples[0], records[0])
        queue.put_groups(groups)
        restored = api.SampleQueue()
        restored.load_state_dict(queue.state_dict())
        batch = maker.make_experience(restored.pop_groups(1, policy_version=0))
        self.assertIsInstance(batch, api.DataProto)
        self.assertEqual(tuple(batch.batch['position_ids'].shape), (2, 3, 8))
        self.assertEqual(batch.batch['input_ids'][0].tolist(), [10, 11, 0, 0, 0, 30, 31, 0])
        self.assertEqual(batch.batch['rollout_log_probs'].device, records[0].batch['rollout_log_probs'].device)
        self.assertEqual(batch.meta_info['global_token_num'], [4, 7])


if __name__ == '__main__':
    unittest.main()
