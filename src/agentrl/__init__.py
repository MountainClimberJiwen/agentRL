"""agentRL — Extract session data from Codex / Kimi / Hermes / cc-connect for RL training.

v0.2.0  Key upgrade: Trajectory-Aware agentRL
  - TrajectoryExtractor: rebuild fine-grained action sequences from session logs
  - ActionRecommender: predict next action from historical transitions
  - OnlinePolicyUpdater: REINFORCE updates after every session
  - HermesHook: automatic lifecycle integration
"""

__version__ = "0.2.0"

from agentrl.extractors.trajectory import TrajectoryBuilder
from agentrl.policy.action_recommender import ActionRecommender
from agentrl.policy.online_updater import OnlinePolicyUpdater
from agentrl.policy.network import SoftmaxPolicy
from agentrl.hermes_plugin.hook import AgentRLHermesHook

__all__ = [
    "TrajectoryBuilder",
    "ActionRecommender",
    "OnlinePolicyUpdater",
    "SoftmaxPolicy",
    "AgentRLHermesHook",
]
