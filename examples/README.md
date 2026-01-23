# Examples

This directory contains runnable examples demonstrating how to use the Skill Optimizer.

## Available Examples

| Example | Description |
|---------|-------------|
| `basic_optimization.py` | Complete workflow: load → analyze → optimize → save |
| `agent_usage.py` | Interactive agent for natural language optimization |
| `create_skill.py` | Programmatically create custom skills |

## Running Examples

```bash
# From the project root
cd skill-optimizer

# Run basic optimization workflow
uv run python examples/basic_optimization.py

# Run agent example
uv run python examples/agent_usage.py

# Create a custom skill
uv run python examples/create_skill.py
```

## Prerequisites

Make sure you have:
1. Installed dependencies: `uv sync`
2. Set up `.env` with your API keys
3. Have at least one skill in the `skills/` directory
