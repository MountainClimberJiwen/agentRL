# AgentRL Policy Network Plugin for Hermes

> **Prompt-as-Policy**: No model weights are trained. All evolution happens in the prompt space.

This plugin injects learned behavioral strategies into each Hermes turn via the `pre_llm_call` hook. It reads patterns from [agentRL](https://github.com/MountainClimberJiwen/agentrl)'s `user_memory.json` and selects the most relevant strategy based on the current task context.

## How It Works

```
User Query
    ↓
State Analysis (intent detection: coding/doc/debug/deploy...)
    ↓
Policy Lookup (match learned patterns against current state)
    ↓
Strategy Injection (via pre_llm_call hook → user message context)
    ↓
Hermes LLM Call (with learned strategy visible)
```

## Features

- **Zero core code changes** — pure plugin, works with any Hermes installation
- **Intent-aware matching** — 8 task types (coding, doc, debug, deploy, test, review, explain, config)
- **Fallback defaults** — works immediately even before agentRL data is generated
- **Platform-adaptive formatting** — compact text for mobile, markdown for CLI
- **Success-rate ranked** — higher-confidence patterns are prioritized

## Installation

```bash
# Clone into Hermes plugins directory
git clone git@github.com:MountainClimberJiwen/agentrl-policy.git \
  ~/.hermes/plugins/agentrl-policy

# Restart Hermes (or run `hermes` again)
# Plugin auto-registers on import
```

## Configuration

Optional: set these in your shell or `~/.hermes/.env`:

```bash
# Path to agentRL data (default: /opt/agentrl/data)
AGENTRL_DATA_DIR=/path/to/agentrl/data

# Max strategies per turn (default: 3)
AGENTRL_MAX_STRATEGIES=3

# Minimum success rate threshold (default: 0.15)
AGENTRL_MIN_SUCCESS_RATE=0.15
```

## Data Flow

1. **agentRL** mines your session history → writes `data/user_memory.json`
2. **This plugin** reads `user_memory.json` on each `pre_llm_call`
3. **Hermes** injects the selected strategy into the current turn's user message

Run agentRL pipeline to generate data:

```bash
cd /path/to/agentrl
python scripts/extract.py --stats
python scripts/ingest.py
python scripts/extract_user_patterns.py --show-prompt
```

## Default Strategies (Built-in)

Even before agentRL data exists, the plugin ships with proven patterns mined from 376 real sessions:

| Strategy | Success Rate | When Applied |
|----------|-------------|--------------|
| Read README/config before implementing | **91%** | coding, first turn |
| Read existing tests before modifying code | **88%** | coding, testing |
| Read existing files before writing | **85%** | general |
| Ask for confirmation on doc tasks | **14%** → caution | documentation |
| Verify prerequisites before deploying | **72%** | deployment |
| Gather logs before proposing fixes | **68%** | debugging |
| Read target files completely before reviewing | **76%** | code review |

## Architecture

```
├── plugin.yaml              # Plugin metadata
├── src/
│   └── __init__.py          # Hook registration + policy engine
└── data/                    # Plugin-local cache (optional)
```

## Extending

To add your own strategies, edit `src/__init__.py`:

```python
DEFAULT_LEARNED_PATTERNS.append({
    "description": "Your custom strategy here",
    "success_rate": 0.80,
    "context": "coding",
    "tags": ["your_tag"],
})
```

Or generate them automatically via agentRL's pattern miner.

## Future: PR to Hermes Core

This plugin proves the value of **runtime prompt strategy selection**. The next step is a PR to Hermes adding a `build_system_prompt` hook so plugins can modify the system prompt directly (higher weight than user message injection).

## License

MIT
