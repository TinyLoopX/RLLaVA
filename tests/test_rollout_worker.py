import unittest
from unittest.mock import Mock

from rllava.rollout.rollout_worker import RolloutWorker
from rllava.rollout.agents.base import AgentAction


class WorkerTests(unittest.TestCase):
    def make_worker(self, actions, max_turns=3):
        agent = Mock()
        agent.reset_arguments.return_value = (("task",), {})
        agent.accept_output.side_effect = actions
        agent.build_inputs.return_value = "prompt"
        agent.finish.side_effect = lambda step, final: (step, final)
        env = Mock()
        env.step.return_value = ("obs", 2.0, False, {})
        env.evaluate.return_value = 3.0
        worker = RolloutWorker(agent, max_turns, env_factory=lambda: env, evaluate_env=True)
        return worker, agent, env

    def test_last_step_reward_is_recorded_before_limit_and_cleanup(self):
        worker, agent, env = self.make_worker([AgentAction("environment", "click")], max_turns=1)
        session = worker.run("task")
        self.assertEqual(next(session), "prompt")
        with self.assertRaises(StopIteration) as result:
            session.send("output")
        self.assertEqual(result.exception.value, (2.0, 3.0))
        env.reset.assert_called_once_with("task")
        agent.observe.assert_called_once_with("obs", 2.0, {}, continue_episode=False,
                                             action=AgentAction("environment", "click"))
        env.evaluate.assert_called_once()
        env.close.assert_called_once()

    def test_environment_done_stops_before_next_generation(self):
        worker, agent, env = self.make_worker([AgentAction("environment", "click")])
        env.step.return_value = ("final obs", 1.0, True, {})
        session = worker.run("task")
        next(session)
        with self.assertRaises(StopIteration):
            session.send("output")
        agent.build_inputs.assert_called_once()
        env.close.assert_called_once()

    def test_reset_failure_closes_environment(self):
        worker, _, env = self.make_worker([])
        env.reset.side_effect = ValueError("reset failed")
        with self.assertRaisesRegex(ValueError, "reset failed"):
            next(worker.run("task"))
        env.close.assert_called_once()

    def test_cancellation_releases_suspended_worker(self):
        worker, _, env = self.make_worker([])
        session = worker.run("task")
        next(session)
        session.close()
        env.close.assert_called_once()
        env.evaluate.assert_not_called()

    def test_step_failure_preserves_error_and_releases_tools(self):
        worker, _, env = self.make_worker([AgentAction("environment", "click")])
        tool = Mock()
        worker.tools_factory = lambda: [tool]
        env.step.side_effect = ValueError("step failed")
        session = worker.run("task")
        next(session)
        with self.assertRaisesRegex(ValueError, "step failed"):
            session.send("output")
        env.close.assert_called_once()
        tool.release.assert_called_once()

    def test_tool_execution_and_model_finish(self):
        worker, agent, env = self.make_worker([AgentAction("tool", "query", 0), AgentAction("finish")])
        tool = Mock()
        tool.execute.return_value = ("result", True)
        worker.tools_factory = lambda: [tool]
        session = worker.run("task")
        next(session)
        self.assertEqual(session.send("call"), "prompt")
        with self.assertRaises(StopIteration) as result:
            session.send("answer")
        self.assertEqual(result.exception.value, (1.0, 3.0))
        env.step.assert_not_called()
        tool.execute.assert_called_once_with("query")
        tool.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
