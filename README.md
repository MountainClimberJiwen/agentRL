# agentRL — Agent Self-Evolution via Reinforcement Learning

> **Core Constraint**: We do **not** touch the underlying LLM weights. All evolution happens in the **prompt space** and **memory space**.

## Why This Project Exists

Your agent has talked to you thousands of times. It should be learning from those conversations. Most agents treat every session as a fresh start — they don't remember what worked, what failed, or how *you* prefer to work.

We mined **376 sessions / 2,325 turns** from real agent usage and found patterns that directly explain why agents fail and how they could improve:

### What the User is Actually Trying to Do (Intent Distribution)

| Intent | Sessions | Success Rate | Insight |
|--------|---------|-------------|---------|
| **explain**（解释/答疑） | 32 | **53%** ✅ | Knowledge questions succeed most often |
| **refactor**（重构） | 5 | **60%** ✅ | Small sample but high success |
| **feature**（加功能） | 124 | 41% | Most common task, but ~60% fail |
| **review**（审查） | 104 | 41% | Same failure rate as feature work |
| **test**（测试） | 103 | 37% | Testing requests often go wrong |
| **debug**（调试） | 48 | 31% | Debugging is hard for agents |
| **deploy**（部署） | 39 | 36% | Deployment tasks frequently fail |
| **doc**（文档/注释） | 76 | **14%** ❌ | **Documentation is the hardest task** |

> **Key insight**: `explain` (53%) and `doc` (14%) have a **4x success gap**. The agent is great at answering questions but terrible at writing docs. If the agent knew this about itself, it could be more cautious and ask for confirmation before writing documentation.

### Root-Cause Failure Patterns

| Failure Mode | Count | Root Cause | Actionable Fix |
|-------------|-------|-----------|---------------|
| **Agent changes code without reading files first** | 6x | No context before writing | → Read existing code before any modification |
| **Agent writes files, user abandons session** | 3x | Wrong file or unwanted change | → Confirm file selection and preview before writing |

### Proven Success Sequences

| Sequence | Frequency | What It Means |
|---------|-----------|--------------|
| **Read → Understand → Write** | 12x | The classic reliable path |
| **Answer directly (no file changes)** | 73x | Pure explanations almost never fail |

### User's Preferred Workflows (High Success Rate)

| Workflow | Uses | Success Rate |
|---------|------|-------------|
| Review documentation (README, .md) before implementation | 37x | **91%** |
| Read project config files early | 16x | **93%** |
| Read existing tests before modifying code | 4x | **88%** |

> **Key insight**: When the agent starts by reading README/config/test files, success jumps to **90%+**. When it doesn't, success drops to ~40%. This is a learnable behavior — and it's just a prompt away.

---

## The Problem

Agent sessions produce a rich signal — **outcomes** (approved/exited/corrected), **corrections**, **tool usage**, **file access patterns** — but this signal is almost always discarded. Every new session starts from scratch.

**agentRL** turns that signal into a self-improvement loop:

```
Session History → Extract Patterns → Compute Rewards → Evolve Prompts → Better Agent
       ▲                                                            │
       └──────────────────── New Sessions ──────────────────────────┘
```

## The Solution: Prompt-as-Policy

We don't train the LLM. We **evolve the prompts and memory** that feed it.

Think of it as: the LLM is a fixed engine; agentRL learns to assemble better fuel (prompts) and better navigation (memory) for that engine.

### Why Prompt + Memory, Not Model Weights?

- **Portability**: Works with any backend (Kimi, Codex, Hermes, Claude, GPT-4) without retraining
- **Cost**: No GPU clusters for fine-tuning; iteration is instant
- **Interpretability**: You can inspect and edit what the agent learned
- **Safety**: Bad learnings can be rolled back by reverting a prompt or memory rule

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        agentRL Self-Evolution                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      Extract       ┌──────────────┐               │
│  │   Sessions  │ ─────────────────► │ UnifiedTurns │               │
│  │  (4 backends)                    │  + Rewards   │               │
│  └─────────────┘                    └──────┬───────┘               │
│                                            │                        │
│                                            ▼                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              SQLite Experience Buffer                   │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │       │
│  │  │memory_feedback│  │prompt_registry│  │eval_results    │  │       │
│  │  │(session data) │  │(prompt versions)│  │(offline eval)  │  │       │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                           ▲                    │                    │
│                           │                    │                    │
│        ┌──────────────────┘                    │                    │
│        │                                       ▼                    │
│        │  ┌─────────────────────────────────────────────┐          │
│        │  │         Prompt Evolution Engine             │          │
│        │  │  ┌─────────────┐      ┌───────────────┐     │          │
│        │  │  │   GEPA      │◄────►│  LLM Judge    │     │          │
│        │  │  │  Mutator    │      │  (scorer)     │     │          │
│        │  │  └──────┬──────┘      └───────────────┘     │          │
│        │  │         │                                    │          │
│        │  │         ▼                                    │          │
│        │  │  ┌─────────────────────────────────────┐    │          │
│        │  │  │      Prompt Version Registry        │    │          │
│        │  │  │  router_v1, router_v2, ...          │    │          │
│        │  │  │  selector_v1, selector_v2, ...      │    │          │
│        │  │  └─────────────────────────────────────┘    │          │
│        │  └─────────────────────────────────────────────┘          │
│        │                                                           │
│        └────────────── 部署最优 prompt ────────────────────────────┘
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Runtime Agent                             │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │   │
│  │  │User Query   │───►│Memory Router│───►│Coarse Filter│     │   │
│  │  │             │    │  Prompt     │    │  (rules)    │     │   │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘     │   │
│  │                                                │             │   │
│  │  ┌─────────────┐    ┌─────────────┐           │             │   │
│  │  │  Frozen LLM │◄───│ Final Prompt│◄──────────┘             │   │
│  │  │             │    │Assembler    │                          │   │
│  │  └─────────────┘    └─────────────┘                          │   │
│  │                                 ▲                             │   │
│  │  ┌─────────────────────────────┘                             │   │
│  │  │ Evidence Selection Prompt + Retrieved Context             │   │
│  │  └───────────────────────────────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## What Gets Optimized?

| Layer | What It Is | RL Target | Example Learned Behavior |
|-------|-----------|-----------|-------------------------|
| **System Prompt** | The meta-instruction given to the LLM | Which prompt variant produces higher reward | "Always ask clarifying questions before writing code" vs "Write code immediately" |
| **Memory Retrieval** | What past context is injected into the prompt | Which retrieved sessions improve grounding | Prefer recent sessions from the same project; downweight generic tutorials |
| **User Preference Memory** | Learned user behavioral patterns | What workflows/patterns lead to approval | "User prefers reading README before implementation (91% success)" |
| **Tool Selection Bias** | Priors over which tools to call | Tool call sequences that lead to approval | "Read existing tests before modifying code" |
| **Self-Reflection Prompt** | Prompt that asks the model to critique its own output | Reflection patterns that catch errors | "Check if the file path exists before writing" |

## Current Pipeline

1. **Extract** — Parse session logs from multiple agent backends into a unified `UnifiedTurn` format
2. **Reward** — Compute multi-level rewards (accuracy, grounding, temporal) from user outcomes
3. **Pattern Mine** — Extract user preferences, workflows, failure modes, and success sequences
4. **Store** — Persist into SQLite `memory_feedback` table + `user_memory.json` profile
5. **Evaluate** — Offline metrics + optional LLM Judge for precise scoring
6. **Evolve** — GEPA-style prompt evolution (coming next)

### Supported Backends

| Backend | Source path | Events parsed |
|---------|-------------|---------------|
| **Claude Code** | `~/.claude/sessions/<id>/history.json` | `user`, `assistant`, `tool_use`, `tool_result` |
| Codex | `~/.codex/sessions/**/*.jsonl` | `task_complete`, `turn_aborted`, `exec_command`, `user_message` |
| Kimi | `~/.kimi/sessions/<workspace>/<session>/wire.jsonl` | `TurnBegin`, `TurnEnd`, `StepInterrupted`, `ToolCall` |
| Hermes | `~/.hermes/sessions/*.jsonl` | `user`, `assistant`, `tool` messages |
| cc-connect | `~/.cc-connect/sessions/*.json` | `history` list |

### Outcome Taxonomy

| Outcome | Meaning | Reward |
|---------|---------|--------|
| `approved` | User explicitly approved or task completed | +1.0 |
| `completed` | Assistant responded but no explicit approval | +0.5 |
| `corrected` | User corrected agent before continuing | -0.3 |
| `rejected` | User explicitly rejected | -0.8 |
| `exited` | User abandoned / closed session without resolving | -1.0 |
| `unknown` | Cannot determine outcome | 0.0 |

## RL Methods (Model-Free, Prompt-Only)

### 1. Prompt Bandits / Contextual Bandits
Treat each prompt variant as an arm. Use Thompson Sampling or LinUCB to select the best system prompt template given the current task context (coding vs writing vs analysis).

### 2. Preference-Based Optimization (DPO-style)
From the same session, build preference pairs:
- **Chosen**: The turn that led to `approved`
- **Rejected**: The turn that led to `exited` or `corrected`

Learn a ranking model over (prompt, memory context) pairs. The ranking model is a lightweight classifier (e.g., logistic regression on prompt embeddings), **not** the LLM itself.

### 3. Memory Retrieval as Policy Gradient
Frame memory retrieval as a sequential decision:
- **State**: Current user query + project context
- **Action**: Which memory chunks to retrieve and how to rank them
- **Reward**: Final session outcome

Use REINFORCE or PPO on a small retrieval policy network (again, **not** the LLM — a separate network that decides what goes into the prompt).

### 4. Self-Critique Loop with Learned Reflection Prompts
Generate N candidate responses with different self-reflection prompts, score by learned reward model, deploy the winner. Over time, learn which reflection prompts work for which error patterns.

## Project Structure

```
agentRL/
├── src/agentrl/
│   ├── __init__.py
│   ├── models.py          # UnifiedTurn, UnifiedSession
│   ├── utils.py           # Parsers, temporal keyword detection
│   ├── rewards.py         # Reward computation (the RL signal)
│   ├── db.py              # SQLite schema and ops (experience replay buffer)
│   ├── llm_judge.py       # Kimi API client for precise LLM-as-Judge scoring
│   ├── prompts/           # Prompt templates (evolvable text assets)
│   │   ├── registry.py    # PromptRegistry: load, version, A/B switch
│   │   ├── assembler.py   # PromptAssembler: build final messages
│   │   ├── router.txt     # Memory Router Prompt (learned)
│   │   ├── selector.txt   # Evidence Selection Prompt (learned)
│   │   └── system.txt     # System Prompt
│   ├── memory/            # Memory retrieval system
│   │   └── retrieval.py   # CoarseFilter + MemoryRouter + EvidenceSelector
│   ├── eval/              # Offline evaluation framework
│   │   ├── dataset.py     # Train/val/holdout splits
│   │   ├── metrics.py     # Retrieval quality metrics (NDCG, MRR, etc.)
│   │   └── offline.py     # OfflineEvaluator: run retrieval + aggregate scores
│   ├── patterns/          # User pattern mining
│   │   └── miner.py       # PatternMiner + UserProfile
│   ├── user_memory/       # User preference storage
│   │   └── store.py       # UserMemoryStore: JSON persistence
│   └── extractors/
│       ├── codex.py
│       ├── kimi.py
│       ├── hermes.py
│       ├── ccconnect.py
│       └── unified.py
├── scripts/
│   ├── extract.py         # Export sessions to JSONL
│   ├── ingest.py          # Ingest into SQLite with rewards
│   ├── extract_user_patterns.py  # Mine user behavior from history
│   └── eval_with_judge.py # Run offline eval + optional LLM Judge
├── data/
│   ├── agentrl.db         # Experience buffer
│   └── user_memory.json   # Learned user preference profile
└── pyproject.toml
```

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Extract and print stats
python scripts/extract.py --stats

# 3. Ingest into SQLite (with computed rewards)
python scripts/ingest.py

# 4. Mine user patterns (generates user_memory.json)
python scripts/extract_user_patterns.py --db data/agentrl.db --show-prompt

# 5. Run offline evaluation (fast, $0)
python scripts/eval_with_judge.py --db data/agentrl.db

# 6. Run with LLM Judge for precise scoring (~$0.10-0.50)
python scripts/eval_with_judge.py --db data/agentrl.db --llm-judge --judge-n 10
```

## Guiding Principles (Read Before Any Task)

1. **The LLM is frozen**. We do not call `model.fit()`, `trainer.train()`, or any weight update API. The LLM is a black-box oracle.
2. **The policy is in the prompt**. If you want the agent to behave differently, change what it sees, not what it is.
3. **Memory is part of the policy**. What we retrieve and inject is as important as the system prompt itself.
4. **Rewards come from user outcomes**. No synthetic reward models unless validated against real approval/rejection signals.
5. **Interpretability over complexity**. A learned prompt template you can read is better than a neural policy you cannot debug.

## Next Steps

- [ ] **Bandit Prompt Selection**: A/B test 3-5 system prompt variants, learn the best one per task type
- [ ] **Retrieval Policy**: Learn which past sessions to inject based on current project/query
- [ ] **DPO Pairs**: Build explicit preference pairs from `approved` vs `exited` turns in the same project
- [ ] **Reflection Prompt Optimization**: Learn "self-check" prompts that reduce grounding errors
- [ ] **Reward Model**: Train a lightweight reward model on `computed_reward_accuracy` to enable online RL

---

*agentRL: Don't train the model. Train the interface to the model.*


---

## Author & Contact

**MountainClimberJiwen**

- 📧 Email: ljwscu@gmail.com
- 💬 WeChat: 扫码添加好友
  
  <img src="assets/wechat-contact-qr.jpg" width="200" alt="WeChat QR Code">

- 🐙 GitHub: [@MountainClimberJiwen](https://github.com/MountainClimberJiwen)

## Support This Project

如果这个项目对你有帮助，欢迎请我喝杯咖啡 ☕

**Buy Me a Coffee** 🍵

<img src="assets/payment-qr.jpg" width="200" alt="Support QR Code">

> "每一杯咖啡，都是对一个工程师深夜写代码的温柔慰藉。"
