#!/usr/bin/env python3
"""
Migrate agentRL policy from tabular SoftmaxPolicy to neural MLP/MiniTransformer.

Usage:
    cd /opt/agentrl
    python scripts/migrate_to_nn.py --type mlp --data-dir ./data
    python scripts/migrate_to_nn.py --type transformer --data-dir ./data

This script:
  1. Loads historical trajectories from data/
  2. Replays them through the new neural policy
  3. Saves the pre-trained neural weights as JSON
  4. The new policy can then be used by ActionRecommender

Resource requirements (M3 Pro Mac friendly):
  - MLP: ~1,700 params, trains in <1s
  - Transformer: ~20K params, trains in ~5s
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure agentrl is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentrl.extractors.trajectory import TrajectoryBuilder
from agentrl.models import TaskTrajectory
from agentrl.policy import MLPPolicy, MiniTransformerPolicy, StateEncoder
from agentrl.policy.action_recommender import ActionRecommender


def load_trajectories(data_dir: Path) -> list[TaskTrajectory]:
    """Load all trajectories from data directory."""
    traj_file = data_dir / "trajectories.jsonl"
    if not traj_file.exists():
        print(f"No trajectories found at {traj_file}")
        return []

    trajs = []
    with open(traj_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            traj = TaskTrajectory(**data)
            trajs.append(traj)

    print(f"Loaded {len(trajs)} trajectories")
    return trajs


def train_mlp(trajs: list[TaskTrajectory], data_dir: Path, epochs: int = 10) -> MLPPolicy:
    """Train MLP policy from trajectories."""
    print("\n=== Training MLP Policy ===")
    encoder = StateEncoder()
    policy = MLPPolicy(
        input_dim=encoder.output_dim,
        hidden_dims=[32, 16],
        output_dim=7,
        temperature=1.0,
        lr=0.05,
    )

    action_to_idx = {
        "read_file": 0, "terminal": 1, "browser": 2, "search": 3,
        "llm_response": 4, "execute_code": 5, "user_input": 6,
    }

    for epoch in range(epochs):
        total_reward = 0.0
        update_count = 0

        for traj in trajs:
            agent_steps = [s for s in traj.steps if s.action_type != "user_input"]
            if not agent_steps:
                continue

            # Compute trajectory reward
            reward = {"approved": 1.0, "success": 1.0, "corrected": 0.3, "failed": -1.0, "unknown": 0.0}.get(traj.final_outcome, 0.0)
            reward -= len(traj.correction_points) * 0.2
            reward -= max(0, len(traj.steps) - 10) * 0.02
            total_reward += reward

            # Update policy for each agent step
            for i, step in enumerate(agent_steps):
                intent = ActionRecommender._detect_intent(traj.goal)
                state_str = f"{intent}:{step.action_type}:{i}"
                state_vec = encoder.encode(state_str)
                action_idx = action_to_idx.get(step.action_type, 0)

                # Corrected steps get lower reward
                full_idx = traj.steps.index(step)
                step_reward = reward - 0.5 if full_idx in traj.correction_points else reward

                policy.update(state_vec, action_idx, step_reward)
                update_count += 1

        avg_reward = total_reward / len(trajs) if trajs else 0
        print(f"  Epoch {epoch + 1}/{epochs}: {update_count} updates, avg reward = {avg_reward:.3f}")

    # Save
    save_path = data_dir / "agentrl_mlp_policy.json"
    policy.save(str(save_path))
    print(f"\n✅ MLP policy saved to {save_path}")
    print(f"   Parameters: ~{sum(w.size for w in policy.weights) + sum(b.size for b in policy.biases):,}")
    return policy


def train_transformer(trajs: list[TaskTrajectory], data_dir: Path, epochs: int = 10) -> MiniTransformerPolicy:
    """Train MiniTransformer policy from trajectories."""
    print("\n=== Training MiniTransformer Policy ===")
    encoder = StateEncoder()
    policy = MiniTransformerPolicy(
        input_dim=encoder.output_dim,
        d_model=32,
        nhead=2,
        num_layers=2,
        ff_dim=64,
        output_dim=7,
        max_seq_len=16,
    )

    action_to_idx = {
        "read_file": 0, "terminal": 1, "browser": 2, "search": 3,
        "llm_response": 4, "execute_code": 5, "user_input": 6,
    }

    for epoch in range(epochs):
        total_reward = 0.0
        update_count = 0

        for traj in trajs:
            agent_steps = [s for s in traj.steps if s.action_type != "user_input"]
            if not agent_steps:
                continue

            reward = {"approved": 1.0, "success": 1.0, "corrected": 0.3, "failed": -1.0, "unknown": 0.0}.get(traj.final_outcome, 0.0)
            reward -= len(traj.correction_points) * 0.2
            total_reward += reward

            # Build state sequence for transformer
            intent = ActionRecommender._detect_intent(traj.goal)
            seq_states = []
            for i, step in enumerate(agent_steps):
                state_str = f"{intent}:{step.action_type}:{i}"
                state_vec = encoder.encode(state_str)
                seq_states.append(state_vec)

            # Pad or truncate to max_seq_len
            seq = np.stack(seq_states[-policy.max_seq_len:])
            if len(seq) < policy.max_seq_len:
                pad = np.zeros((policy.max_seq_len - len(seq), encoder.output_dim), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

            # Update on last action (simplified for transformer)
            last_step = agent_steps[-1]
            action_idx = action_to_idx.get(last_step.action_type, 0)
            full_idx = traj.steps.index(last_step)
            step_reward = reward - 0.5 if full_idx in traj.correction_points else reward

            policy.update(seq, action_idx, step_reward)
            update_count += 1

        avg_reward = total_reward / len(trajs) if trajs else 0
        print(f"  Epoch {epoch + 1}/{epochs}: {update_count} updates, avg reward = {avg_reward:.3f}")

    save_path = data_dir / "agentrl_transformer_policy.json"
    policy.save(str(save_path))
    print(f"\n✅ Transformer policy saved to {save_path}")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate agentRL policy to neural network")
    parser.add_argument("--type", choices=["mlp", "transformer"], default="mlp", help="Policy type")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    data_dir = Path(os.environ.get("AGENTRL_DATA_DIR", str(args.data_dir)))
    trajs = load_trajectories(data_dir)

    if not trajs:
        print("No trajectories to train on. Run offline training first.")
        return 1

    if args.type == "mlp":
        train_mlp(trajs, data_dir, args.epochs)
    else:
        train_transformer(trajs, data_dir, args.epochs)

    print("\n✅ Migration complete!")
    print(f"   To use the new policy, update your code:")
    print(f"     from agentrl.policy import MLPPolicy")
    print(f"     policy = MLPPolicy.load('{data_dir}/agentrl_mlp_policy.json')")
    print(f"     rec = ActionRecommender(policy=policy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
