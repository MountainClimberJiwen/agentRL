"""
NeuralNetworkPolicy — Lightweight MLP policy using only NumPy.

Designed for:
  - Training on M3 Pro Mac (18GB unified memory)
  - Inference on CentOS 7 (no PyTorch, only NumPy)
  - JSON-serializable weights for easy transport

Architecture choices (kept tiny for resource constraints):
  - MLP: input(24) → hidden(32) → hidden(16) → output(5)
    ~1,500 parameters, fits in L1 cache, trains in <1s per epoch
  - Optional MiniTransformer: d_model=32, nhead=2, 2 layers
    ~20K parameters, still tiny enough for M3 Pro
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# State Encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """Encode a state string into a fixed-length NumPy vector."""

    INTENTS = ["coding", "doc", "deploy", "config", "debug", "test", "git", "general"]
    ACTIONS = ["read_file", "terminal", "browser", "search", "llm_response", "execute_code", "user_input"]

    def __init__(self, max_step: int = 50) -> None:
        self.max_step = max_step
        self.intent_dim = len(self.INTENTS)
        self.action_dim = len(self.ACTIONS)
        self.output_dim = self.intent_dim + self.action_dim + 1 + 3  # +3 for extra stats features

    def encode(self, state: str, stats: dict[str, float] | None = None) -> np.ndarray:
        """
        state format: '{intent}:{current_action}:{step_idx}'
        Returns float32 vector of shape (output_dim,).
        """
        parts = state.split(":")
        intent = parts[0] if len(parts) > 0 else "general"
        action = parts[1] if len(parts) > 1 else "start"
        step = int(parts[2]) if len(parts) > 2 else 0

        # One-hot intent
        intent_vec = np.zeros(self.intent_dim, dtype=np.float32)
        if intent in self.INTENTS:
            intent_vec[self.INTENTS.index(intent)] = 1.0

        # One-hot action
        action_vec = np.zeros(self.action_dim, dtype=np.float32)
        if action in self.ACTIONS:
            action_vec[self.ACTIONS.index(action)] = 1.0

        # Normalized step
        step_val = np.array([min(step, self.max_step) / self.max_step], dtype=np.float32)

        # Extra stats features (visit count, success rate, correction rate)
        extra = np.zeros(3, dtype=np.float32)
        if stats:
            extra[0] = stats.get("visit_count", 0) / 100.0
            extra[1] = stats.get("success_rate", 0.5)
            extra[2] = stats.get("correction_rate", 0.0)

        return np.concatenate([intent_vec, action_vec, step_val, extra])


# ---------------------------------------------------------------------------
# MLP Policy (NumPy only)
# ---------------------------------------------------------------------------

class MLPPolicy:
    """
    Tiny MLP policy network.
    Default: input(24) → 32 → 16 → output(7)
    ~1,700 params. Trains on CPU in milliseconds.
    """

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dims: list[int] | None = None,
        output_dim: int = 7,
        temperature: float = 1.0,
        min_temp: float = 0.2,
        temp_decay: float = 0.9995,
        lr: float = 0.05,
        momentum: float = 0.9,
        l2_reg: float = 1e-5,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [32, 16]
        self.output_dim = output_dim
        self.temperature = temperature
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.lr = lr
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.global_step = 0

        # Xavier init
        self._init_weights()
        self._init_velocity()

    def _init_weights(self) -> None:
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        for i in range(len(dims) - 1):
            limit = math.sqrt(6.0 / (dims[i] + dims[i + 1]))
            w = np.random.uniform(-limit, limit, (dims[i], dims[i + 1])).astype(np.float32)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def _init_velocity(self) -> None:
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return logits for all actions. x shape: (input_dim,) or (batch, input_dim)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = self._activations[-1] @ w + b
            if i < len(self.weights) - 1:
                z = np.maximum(0, z)  # ReLU
            self._activations.append(z)
        return z.squeeze() if z.shape[0] == 1 else z

    def _softmax(self, logits: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Stable softmax with optional action mask."""
        if mask is not None:
            logits = logits.copy()
            logits[~mask] = -1e9
        max_logit = np.max(logits)
        exp = np.exp((logits - max_logit) / max(self.temperature, 1e-6))
        return exp / np.sum(exp)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select(
        self,
        state_vec: np.ndarray,
        available_actions: list[str],
        action_to_idx: dict[str, int],
    ) -> tuple[str, dict[str, float]]:
        """Sample action from softmax distribution."""
        logits = self.forward(state_vec)
        mask = np.zeros(self.output_dim, dtype=bool)
        for a in available_actions:
            if a in action_to_idx:
                mask[action_to_idx[a]] = True
        probs = self._softmax(logits, mask)
        idx = np.random.choice(len(probs), p=probs)
        action = [k for k, v in action_to_idx.items() if v == idx][0]
        prob_dict = {a: float(probs[action_to_idx[a]]) for a in available_actions if a in action_to_idx}
        return action, prob_dict

    def get_best(
        self,
        state_vec: np.ndarray,
        available_actions: list[str],
        action_to_idx: dict[str, int],
    ) -> str:
        """Greedy best action."""
        logits = self.forward(state_vec)
        best_idx = None
        best_val = -float("inf")
        for a in available_actions:
            if a in action_to_idx:
                idx = action_to_idx[a]
                if logits[idx] > best_val:
                    best_val = logits[idx]
                    best_idx = idx
        if best_idx is None:
            return available_actions[0] if available_actions else ""
        return [k for k, v in action_to_idx.items() if v == best_idx][0]

    def get_probs(
        self,
        state_vec: np.ndarray,
        available_actions: list[str],
        action_to_idx: dict[str, int],
    ) -> dict[str, float]:
        logits = self.forward(state_vec)
        mask = np.zeros(self.output_dim, dtype=bool)
        for a in available_actions:
            if a in action_to_idx:
                mask[action_to_idx[a]] = True
        probs = self._softmax(logits, mask)
        return {a: float(probs[action_to_idx[a]]) for a in available_actions if a in action_to_idx}

    # ------------------------------------------------------------------
    # Learning (policy gradient with momentum SGD)
    # ------------------------------------------------------------------

    def update(
        self,
        state_vec: np.ndarray,
        action_idx: int,
        reward: float,
        lr: float | None = None,
    ) -> None:
        """
        REINFORCE-style single-step update.
        Uses cross-entropy gradient scaled by reward.
        """
        if state_vec.ndim == 1:
            state_vec = state_vec.reshape(1, -1)

        logits = self.forward(state_vec)
        if logits.ndim == 0:
            logits = logits.reshape(1)

        # Softmax probabilities
        probs = self._softmax(logits)

        # Cross-entropy gradient: dL/dz = p - one_hot(y)
        # For REINFORCE: scale by reward (negative because we maximize)
        dlogits = probs.copy()
        dlogits[action_idx] -= 1.0
        dlogits *= -reward  # negative sign for gradient ascent
        dlogits = dlogits.reshape(1, -1)

        # Backpropagation
        grads_w = []
        grads_b = []
        da = dlogits

        for i in range(len(self.weights) - 1, -1, -1):
            # ReLU grad
            if i < len(self.weights) - 1:
                da = da * (self._activations[i + 1] > 0).astype(np.float32)
            dz = da
            a_prev = self._activations[i]
            dw = a_prev.T @ dz + self.l2_reg * self.weights[i]
            db = np.sum(dz, axis=0)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                da = dz @ self.weights[i].T

        # Momentum SGD update
        step_lr = lr if lr is not None else self.lr
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - step_lr * grads_w[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - step_lr * grads_b[i]
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

        # Temperature decay
        self.global_step += 1
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)

    # ------------------------------------------------------------------
    # Persistence (JSON-serializable for CentOS 7 compatibility)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save weights as JSON (human-readable, cross-platform)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "mlp",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "temperature": float(self.temperature),
            "min_temp": float(self.min_temp),
            "temp_decay": float(self.temp_decay),
            "lr": float(self.lr),
            "momentum": float(self.momentum),
            "l2_reg": float(self.l2_reg),
            "global_step": self.global_step,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "velocity_w": [v.tolist() for v in self.velocity_w],
            "velocity_b": [v.tolist() for v in self.velocity_b],
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MLPPolicy":
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            input_dim=data["input_dim"],
            hidden_dims=data["hidden_dims"],
            output_dim=data["output_dim"],
            temperature=data["temperature"],
            min_temp=data["min_temp"],
            temp_decay=data["temp_decay"],
            lr=data["lr"],
            momentum=data["momentum"],
            l2_reg=data["l2_reg"],
        )
        policy.global_step = data.get("global_step", 0)
        policy.weights = [np.array(w, dtype=np.float32) for w in data["weights"]]
        policy.biases = [np.array(b, dtype=np.float32) for b in data["biases"]]
        policy.velocity_w = [np.array(v, dtype=np.float32) for v in data.get("velocity_w", [[0.0] * len(b) for b in policy.biases])]
        policy.velocity_b = [np.array(v, dtype=np.float32) for v in data.get("velocity_b", [[0.0] * len(b) for b in policy.biases])]
        return policy


# ---------------------------------------------------------------------------
# Mini Transformer Policy (NumPy only, optional)
# ---------------------------------------------------------------------------

class MiniTransformerPolicy:
    """
    Tiny Transformer for sequential state modeling.
    Input: sequence of state vectors (history of actions)
    Output: logits for next action.

    Default: d_model=32, nhead=2, num_layers=2, ff_dim=64
    ~20K parameters. Small enough for M3 Pro Mac.
    """

    def __init__(
        self,
        input_dim: int = 24,
        d_model: int = 32,
        nhead: int = 2,
        num_layers: int = 2,
        ff_dim: int = 64,
        output_dim: int = 7,
        max_seq_len: int = 16,
        temperature: float = 1.0,
        lr: float = 0.01,
    ) -> None:
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.lr = lr
        self.global_step = 0

        # Input projection
        limit = math.sqrt(6.0 / (input_dim + d_model))
        self.W_in = np.random.uniform(-limit, limit, (input_dim, d_model)).astype(np.float32)
        self.b_in = np.zeros(d_model, dtype=np.float32)

        # Positional encoding (sinusoidal, frozen)
        self.pos_enc = self._build_pos_enc()

        # Transformer layers
        self.layers: list[dict[str, np.ndarray]] = []
        for _ in range(num_layers):
            self.layers.append(self._init_transformer_layer())

        # Output head
        limit = math.sqrt(6.0 / (d_model + output_dim))
        self.W_out = np.random.uniform(-limit, limit, (d_model, output_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def _build_pos_enc(self) -> np.ndarray:
        pos = np.arange(self.max_seq_len)[:, None]
        div = np.exp(np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_seq_len, self.d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def _init_transformer_layer(self) -> dict[str, np.ndarray]:
        d = self.d_model
        # Multi-head attention: W_q, W_k, W_v, W_o
        limit = math.sqrt(6.0 / (d + d))
        return {
            "W_q": np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
            "W_k": np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
            "W_v": np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
            "W_o": np.random.uniform(-limit, limit, (d, d)).astype(np.float32),
            "b_qkv": np.zeros(d, dtype=np.float32),
            "b_o": np.zeros(d, dtype=np.float32),
            "W_ff1": np.random.uniform(-limit, limit, (d, self.ff_dim)).astype(np.float32),
            "b_ff1": np.zeros(self.ff_dim, dtype=np.float32),
            "W_ff2": np.random.uniform(-limit, limit, (self.ff_dim, d)).astype(np.float32),
            "b_ff2": np.zeros(d, dtype=np.float32),
            "gamma1": np.ones(d, dtype=np.float32),
            "beta1": np.zeros(d, dtype=np.float32),
            "gamma2": np.ones(d, dtype=np.float32),
            "beta2": np.zeros(d, dtype=np.float32),
        }

    def _attention(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        """Simplified multi-head self-attention."""
        batch, seq, d = x.shape
        head_dim = d // self.nhead

        Q = x @ layer["W_q"] + layer["b_qkv"]
        K = x @ layer["W_k"] + layer["b_qkv"]
        V = x @ layer["W_v"] + layer["b_qkv"]

        # Reshape for multi-head
        Q = Q.reshape(batch, seq, self.nhead, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.nhead, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.nhead, head_dim).transpose(0, 2, 1, 3)

        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        # Causal mask
        mask = np.triu(np.ones((seq, seq)), k=1) * -1e9
        scores = scores + mask
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        out = attn @ V
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq, d)
        return out @ layer["W_o"] + layer["b_o"]

    def _ffn(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        h = np.maximum(0, x @ layer["W_ff1"] + layer["b_ff1"])
        return h @ layer["W_ff2"] + layer["b_ff2"]

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + 1e-6) + beta

    def forward(self, seq_states: np.ndarray) -> np.ndarray:
        """
        seq_states: (seq_len, input_dim) or (batch, seq_len, input_dim)
        Returns logits: (output_dim,)
        """
        if seq_states.ndim == 2:
            seq_states = seq_states[None, ...]
        batch, seq, _ = seq_states.shape

        # Project + pos enc
        x = seq_states @ self.W_in + self.b_in
        x = x + self.pos_enc[:seq]

        for layer in self.layers:
            # Self-attention + residual
            attn_out = self._attention(x, layer)
            x = self._layer_norm(x + attn_out, layer["gamma1"], layer["beta1"])
            # FFN + residual
            ff_out = self._ffn(x, layer)
            x = self._layer_norm(x + ff_out, layer["gamma2"], layer["beta2"])

        # Use last token for prediction
        last = x[:, -1, :]  # (batch, d_model)
        logits = last @ self.W_out + self.b_out
        return logits.squeeze() if batch == 1 else logits

    def select(self, seq_states, available_actions, action_to_idx):
        logits = self.forward(seq_states)
        mask = np.zeros(self.output_dim, dtype=bool)
        for a in available_actions:
            if a in action_to_idx:
                mask[action_to_idx[a]] = True
        probs = self._softmax(logits, mask)
        idx = np.random.choice(len(probs), p=probs)
        action = [k for k, v in action_to_idx.items() if v == idx][0]
        return action, {a: float(probs[action_to_idx[a]]) for a in available_actions if a in action_to_idx}

    def get_best(self, seq_states, available_actions, action_to_idx):
        logits = self.forward(seq_states)
        best = max(
            (a for a in available_actions if a in action_to_idx),
            key=lambda a: logits[action_to_idx[a]],
            default=available_actions[0] if available_actions else "",
        )
        return best

    def _softmax(self, logits, mask=None):
        if mask is not None:
            logits = logits.copy()
            logits[~mask] = -1e9
        max_logit = np.max(logits)
        exp = np.exp((logits - max_logit) / max(self.temperature, 1e-6))
        return exp / np.sum(exp)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "transformer",
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "output_dim": self.output_dim,
            "max_seq_len": self.max_seq_len,
            "temperature": float(self.temperature),
            "lr": float(self.lr),
            "global_step": self.global_step,
            "W_in": self.W_in.tolist(),
            "b_in": self.b_in.tolist(),
            "layers": [
                {k: v.tolist() for k, v in layer.items()} for layer in self.layers
            ],
            "W_out": self.W_out.tolist(),
            "b_out": self.b_out.tolist(),
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MiniTransformerPolicy":
        with open(path) as f:
            data = json.load(f)
        policy = cls(
            input_dim=data["input_dim"],
            d_model=data["d_model"],
            nhead=data["nhead"],
            num_layers=data["num_layers"],
            ff_dim=data["ff_dim"],
            output_dim=data["output_dim"],
            max_seq_len=data["max_seq_len"],
            temperature=data["temperature"],
            lr=data["lr"],
        )
        policy.global_step = data.get("global_step", 0)
        policy.W_in = np.array(data["W_in"], dtype=np.float32)
        policy.b_in = np.array(data["b_in"], dtype=np.float32)
        policy.layers = [
            {k: np.array(v, dtype=np.float32) for k, v in layer.items()}
            for layer in data["layers"]
        ]
        policy.W_out = np.array(data["W_out"], dtype=np.float32)
        policy.b_out = np.array(data["b_out"], dtype=np.float32)
        return policy
