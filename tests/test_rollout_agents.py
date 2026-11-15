import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from test_generator import module
from test_rollout_manager import ROOT, Tensor, TensorMap, Proto


def load_multiturn_agent():
    torch = module('torch',
        tensor=lambda value, **kw: Tensor.make(value),
        cat=lambda values, dim: Tensor.make(np.concatenate(values, axis=dim)),
        ones_like=lambda value, **kw: Tensor.make(np.ones_like(value)),
        arange=lambda *args, **kw: Tensor.make(np.arange(*args)))
    spec = importlib.util.spec_from_file_location('rllava.rollout.agents.multi_turn', ROOT/'rllava/rollout/agents/multi_turn.py')
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        'torch': torch,
        'tensordict': module('tensordict', TensorDict=TensorMap),
        'rllava.data.data_utils': module('rllava.data.data_utils', process_image=Mock()),
    }):
        spec.loader.exec_module(loaded)
    return loaded.MultiTurnAgent


class AgentTests(unittest.TestCase):
    def test_context_appends_actual_response_once_and_preserves_eos(self):
        tokenizer = SimpleNamespace(decode=Mock(return_value='tool call'),
                                    convert_tokens_to_ids=lambda value: 99)
        agent = load_multiturn_agent()(tokenizer)
        agent.data = Proto({'input_ids': Tensor.make([[1, 2]])},
                           {'step_reward': np.zeros(1, dtype=np.float32)})
        output = Proto({
            'prompts': Tensor.make([[1, 2]]),
            'responses': Tensor.make([[3, 0, 0]]),
            'response_mask': Tensor.make([[1, 1, 0]]),  # EOS == pad; first zero is real
            'input_ids': Tensor.make([[1, 2, 3, 0, 0]]),
            'attention_mask': Tensor.make([[1, 1, 1, 1, 0]]),
            'position_ids': Tensor.make([[0, 1, 2, 3, 4]]),
        })
        env = Mock()
        env.extract_action.return_value = 'click'
        action = agent.accept_output(output, env, [])
        self.assertEqual(action.kind, 'environment')
        self.assertEqual(agent.response_ids, [3, 0])
        env.extract_action.assert_called_once_with('tool call')
        agent._extend_batch_with_turn(output, agent.response_ids, [9], [], False)
        np.testing.assert_array_equal(output.batch['input_ids'], [[1, 2, 3, 0, 9]])
        np.testing.assert_array_equal(output.batch['attention_mask'], [[1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(output.batch['position_ids'], [[0, 1, 2, 3, 4]])

    def test_terminal_observation_records_reward_without_changing_context(self):
        agent = load_multiturn_agent()(SimpleNamespace())
        agent.output = Proto({'input_ids': Tensor.make([[1, 2]])}, {'step_reward': np.zeros(1)})
        agent._add_observation = Mock()
        agent.observe('last screenshot', 2.0, {}, continue_episode=False,
                      action=SimpleNamespace(kind='environment'))
        self.assertEqual(agent.output.non_tensor_batch['step_reward'][0], 2.0)
        agent._add_observation.assert_not_called()
        result = agent.finish(2.0, 3.0)
        self.assertEqual(result.non_tensor_batch['env_reward'][0], 5.0)


if __name__ == '__main__':
    unittest.main()
