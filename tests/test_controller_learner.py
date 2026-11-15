"""Orchestration regression tests with fake roles, without GPU dependencies."""
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from test_generator import module
from test_rollout_manager import Tensor

ROOT = Path(__file__).resolve().parents[1]


class Values(Tensor):
    def bool(self):
        return self.astype(bool)

    def float(self):
        return self.astype(float)

    def masked_fill(self, mask, value):
        result = self.copy()
        result[mask] = value
        return result


class Data:
    def __init__(self, tensors=None):
        self.batch = tensors or {}
        self.non_tensor_batch = {}
        self.meta_info = {}

    @classmethod
    def from_dict(cls, tensors):
        return cls(tensors)

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        return self


def load_source(name, relative, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(loaded)
    return loaded


def common_dependencies():
    return {
        'rllava.utils': module('rllava.utils', torch_functional=NS()),
        'rllava.utils.py_functional': module('rllava.utils.py_functional', timer=lambda *a: nullcontext()),
        'rllava.utils.logger.aggregate_logger': module('rllava.utils.logger.aggregate_logger', print_rank_0=Mock()),
    }


def load_learner():
    deps = common_dependencies()
    deps.update({
        'torch': module('torch', Tensor=Tensor, allclose=np.allclose, full_like=np.full_like,
                        equal=np.array_equal),
        'rllava.roles.actor': module('rllava.roles.actor', Actor=Mock()),
        'rllava.roles.critic': module('rllava.roles.critic', Critic=Mock()),
        'rllava.roles.ref': module('rllava.roles.ref', Ref=Mock()),
        'rllava.train.core_algos': module('rllava.train.core_algos', kl_penalty=Mock(), get_kl_controller=Mock()),
        'rllava.data.protocol': module('rllava.data.protocol', DataProto=Data),
        'rllava.data.data_utils': module('rllava.data.data_utils', process_image=lambda image, *a: image, process_video=Mock()),
        'rllava.distribute.dist_utils': module('rllava.distribute.dist_utils', on_coordinator=lambda fn, **kw: fn()),
        'rllava.train.advantage': module('rllava.train.advantage', get_adv_estimator=Mock()),
        'rllava.train.policy_loss': module('rllava.train.policy_loss', get_policy_loss=Mock()),
    })
    return load_source('rllava.train.train', 'rllava/train/train.py', deps).Learner


def load_controller():
    deps = common_dependencies()
    deps.update({
        'tqdm': module('tqdm', tqdm=lambda values, **kw: values),
        'rllava.controller.base': module('rllava.controller.base', BaseController=object),
        'rllava.distribute.resource_manager': module('rllava.distribute.resource_manager', ResourceManager=Mock()),
        'rllava.distribute.batch_dispatcher': module('rllava.distribute.batch_dispatcher', BatchDispatcher=Mock()),
        'rllava.distribute.dist_utils': module('rllava.distribute.dist_utils', is_rank0=lambda: True),
        'rllava.rollout': module('rllava.rollout', RolloutManager=Mock()),
        'rllava.roles.reward': module('rllava.roles.reward', Reward=Mock()),
        'rllava.train.train': module('rllava.train.train', Learner=Mock()),
        'rllava.rollout.processors': module('rllava.rollout.processors',
                                                   PrefixRolloutProcessor=Mock(), HPTBatchRolloutProcessor=Mock()),
        'rllava.utils.metrics': module('rllava.utils.metrics', **{
            name: Mock(return_value={}) for name in ('compute_data_metrics', 'compute_throughout_metrics',
                                                    'compute_timing_metrics', 'reduce_metrics')
        }),
    })
    return load_source('rllava.controller.controller', 'rllava/controller/controller.py', deps)


class LearnerTests(unittest.TestCase):
    def test_role_construction_supports_peft_reference_and_requires_gae_values(self):
        cls = load_learner()
        actor_cls = make_dataclass('ActorConfig', ['ppo_mini_batch_size', 'policy_loss', 'parallel', 'model'])
        actor = actor_cls(4, NS(loss_mode='vanilla'), object(), NS(use_peft=True))
        config = NS(actor=actor, critic=None, rollout=NS(n=2),
                    algorithm=NS(adv_estimator='grpo', use_kl_loss=True, use_kl_in_reward=False))
        resources = NS(roles={'actor': NS(dp_size=2), 'ref': NS(parallel=object())})
        learner = cls(config, 'tokenizer', 'processor', resources=resources, dispatcher=Mock())
        self.assertIsNotNone(learner.ref)
        self.assertIsNone(learner.critic)
        self.assertEqual(actor.ppo_mini_batch_size, 4)
        config.algorithm.adv_estimator = 'gae'
        with self.assertRaisesRegex(ValueError, 'GAE requires a critic'):
            cls(config, 'tokenizer', 'processor', resources=resources, dispatcher=Mock())

    def make_learner(self):
        cls = load_learner()
        learner = cls.__new__(cls)
        learner._cache = {}
        learner.config = NS(algorithm=NS(use_kl_loss=False),
                            rollout=NS(temperature=1.), actor=NS(tis_imp_ratio_cap=0),
                            trainer=NS(critic_warmup=2, save_model_only=False))
        learner.use_critic = True
        learner.actor, learner.critic, learner.ref = Mock(), Mock(), None
        return learner

    def test_values_are_available_before_advantages_and_updates(self):
        learner = self.make_learner()
        data = Data()
        events = []
        learner.compute_log_probs = Mock(side_effect=lambda batch: events.append('prob') or batch)
        learner.compute_values = Mock(side_effect=lambda batch: events.append('value') or Values.make([2.]))
        def advantage(batch):
            self.assertEqual(batch.batch['values'][0], 2.)
            events.append('adv')
            return batch
        learner.compute_advantage = advantage
        learner.update_model = Mock(side_effect=lambda *a, **kw: events.append('update') or Data())
        learner.step(data, global_step=4)
        self.assertEqual(events, ['prob', 'value', 'adv', 'update'])
        learner.use_critic = False
        events.clear()
        learner.step(data, global_step=5)
        self.assertEqual(events, ['prob', 'adv', 'update'])

    def test_warmup_only_updates_critic_then_updates_both(self):
        learner = self.make_learner()
        learner._process_multi_modal_inputs = Mock()
        learner.actor.update.return_value = Data()
        learner.critic.update.return_value = Data()
        learner.update_model(Data(), global_steps=1)
        learner.actor.update.assert_not_called()
        learner.critic.update.assert_called_once()
        learner.update_model(Data(), global_steps=2)
        learner.actor.update.assert_called_once()
        self.assertEqual(learner.critic.update.call_count, 2)

    def test_probabilities_are_frozen_and_temperature_is_checked(self):
        learner = self.make_learner()
        raw = Values.make([[-1., -2.]])
        learner.actor.compute_log_probs.return_value = raw
        data = Data({'responses': Values.make([[1, 0]]), 'response_mask': Values.make([[1, 0]]),
                     'attention_mask': Values.make([[1, 1, 0]]), 'rollout_temperature': Values.make([1.])})
        learner.compute_log_probs(data)
        raw[:] = -9
        np.testing.assert_array_equal(data.batch['old_log_probs'], [[-1., 0.]])
        data.batch['rollout_temperature'][:] = 0.5
        with self.assertRaisesRegex(ValueError, 'temperature differs'):
            learner.compute_log_probs(data)
        learner.actor.compute_log_probs.assert_called_once()

    def test_visual_cache_does_not_mix_same_group_screenshots(self):
        learner = self.make_learner()
        learner.processor = NS(image_processor=Mock(side_effect=lambda images, **kw: {'image': images[0]}))
        data = Data({'input_ids': Values.make([[1], [1]])})
        data.non_tensor_batch = {'uid': np.array(['group', 'group']),
                                 'multi_modal_data': [{'images': ['A']}, {'images': ['B']}]}
        data.meta_info = {'min_pixels': None, 'max_pixels': None, 'video_fps': 2.}
        learner._process_multi_modal_inputs(data)
        self.assertEqual([v['image'] for v in data.non_tensor_batch['multi_modal_inputs']], ['A', 'B'])
        learner._process_multi_modal_inputs(data)
        self.assertEqual(learner.processor.image_processor.call_count, 2)

    def test_learning_failure_clears_derived_inputs(self):
        learner = self.make_learner()
        data = Data()
        data.non_tensor_batch['multi_modal_inputs'] = object()
        learner.compute_log_probs = Mock(side_effect=RuntimeError('forward failed'))
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            learner.step(data, 0)
        self.assertEqual(learner._cache, {})
        self.assertNotIn('multi_modal_inputs', data.non_tensor_batch)

    def test_actor_and_critic_checkpoint_paths(self):
        learner = self.make_learner()
        learner.save_checkpoint('/tmp/checkpoint')
        learner.load_checkpoint('/tmp/checkpoint')
        learner.actor.save_checkpoint.assert_called_once_with('/tmp/checkpoint/actor', save_model_only=False)
        learner.critic.load_checkpoint.assert_called_once_with('/tmp/checkpoint/critic')


class ControllerTests(unittest.TestCase):
    def make_controller(self):
        loaded = load_controller()
        controller = loaded.Controller.__new__(loaded.Controller)
        controller.config = NS(trainer=NS(val_before_train=False, val_only=False, val_freq=1, save_freq=1))
        controller.global_steps, controller.training_steps = 2, 4
        controller.logger = Mock()
        controller.resources = NS(world_size=1, close=Mock())
        controller.dispatcher = NS(gather=Mock(side_effect=lambda batch: batch))
        controller.learner = NS(use_critic=False, step=Mock(side_effect=lambda batch, **kw: (batch, Data())))
        controller.iter_train_batches = lambda: iter([1, 2])
        controller.collect = Mock(side_effect=lambda *a, **kw: Data())
        controller.validate = Mock(return_value={})
        controller.save_checkpoint = Mock()
        return controller, loaded

    def test_resume_runs_remaining_steps_through_one_learner(self):
        controller, _ = self.make_controller()
        controller.run()
        self.assertEqual([call.kwargs['global_step'] for call in controller.learner.step.call_args_list], [2, 3])
        self.assertEqual(controller.global_steps, 4)
        self.assertEqual(controller.dispatcher.gather.call_count, 4)
        self.assertEqual(controller.validate.call_count, 2)
        self.assertEqual(controller.save_checkpoint.call_count, 2)

    def test_validation_only_even_without_pretrain_validation(self):
        controller, _ = self.make_controller()
        controller.config.trainer.val_only = True
        controller.run()
        controller.validate.assert_called_once()
        controller.collect.assert_not_called()
        controller.learner.step.assert_not_called()

    def test_generation_failure_offloads_engine(self):
        controller, _ = self.make_controller()
        controller.learner.actor = NS(unwrap_model_for_generation=lambda: nullcontext('weights'))
        controller.rollout = NS(generator=Mock())
        with self.assertRaisesRegex(RuntimeError, 'sampling failed'):
            with controller.generate_context():
                raise RuntimeError('sampling failed')
        controller.rollout.generator.load.assert_called_once_with('weights')
        controller.rollout.generator.offload.assert_called_once()

    def test_algorithm_configuration_selects_sampling_transform(self):
        _, loaded = self.make_controller()
        config = NS(algorithm=NS(unify_strategy='switch', prefix_strategy='none'),
                    rollout=object(), data=NS(train_batch_size=2), trainer=NS(max_try_make_batch=3))
        loaded.build_rollout(config, 'reward', 'tokenizer', 'processor')
        loaded.HPTBatchRolloutProcessor.assert_called_once_with(config, 'tokenizer')
        config.algorithm.unify_strategy = 'none'
        config.algorithm.prefix_strategy = 'hint'
        loaded.build_rollout(config, 'reward', 'tokenizer', 'processor')
        loaded.PrefixRolloutProcessor.assert_called_once_with(config, 'tokenizer')


if __name__ == '__main__':
    unittest.main()
