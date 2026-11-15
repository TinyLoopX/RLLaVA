"""One worker owns one trajectory, including exceptional cleanup.

The lifecycle is a coroutine: each yield requests generation; the manager
batches those requests and sends the corresponding model outputs back.
"""

import logging

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(self, agent, max_turns=1, env_factory=None, tools_factory=None,
                 evaluate_env=False):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.agent = agent
        self.max_turns = max_turns
        self.env_factory = env_factory
        self.tools_factory = tools_factory
        self.evaluate_env = evaluate_env

    def run(self, data):
        env, tools = None, []
        total_reward = 0.0
        try:
            self.agent.start(data)
            if self.env_factory is not None:
                env = self.env_factory()
            if self.tools_factory is not None:
                tools = self.tools_factory()
            if env is not None:
                args, kwargs = self.agent.reset_arguments()
                self.agent.observe_reset(env.reset(*args, **kwargs))

            for turn in range(self.max_turns):
                output = yield self.agent.build_inputs()
                action = self.agent.accept_output(output, env, tools)
                if action.kind == "finish":
                    break
                if action.kind == "environment":
                    observation, reward, done, info = env.step(action.payload)
                elif action.kind == "tool":
                    observation, success = tools[action.tool_index].execute(action.payload)
                    reward, done, info = float(success), False, {}
                elif action.kind == "feedback":
                    observation, reward, done, info = None, 0.0, False, {}
                else:
                    raise ValueError(f"Unknown agent action kind: {action.kind!r}")
                total_reward += reward
                continuing = not done and turn + 1 < self.max_turns
                self.agent.observe(observation, reward, info,
                                   continue_episode=continuing, action=action)
                if not continuing:
                    break

            eval_reward = 0.0
            if self.evaluate_env and env is not None and hasattr(env, "evaluate"):
                eval_reward = float(env.evaluate())
            return self.agent.finish(total_reward, eval_reward)
        finally:
            # A failure in one close must not leak the remaining resources or
            # replace the original generation/reset/step exception.
            for resource, method in [(env, "close"), *((tool, "release") for tool in tools)]:
                if resource is not None and hasattr(resource, method):
                    try:
                        getattr(resource, method)()
                    except Exception:
                        logger.exception("Failed to release rollout resource")
