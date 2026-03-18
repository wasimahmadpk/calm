# CALM: Causal discovery with an Agent and LLM

Use an LLM-powered agent to run **causal discovery** and **effect estimation** via natural language. The agent can load data, run algorithms (PC, GES, FCI, LiNGAM), and estimate causal effects.

## Setup

```bash
cd calm
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

## Quick start

**CLI**

```bash
python run_agent.py "Load the sachs dataset and run PC causal discovery"
python run_agent.py "Load sachs, run LiNGAM, and estimate the effect of pip2 on pkc"
```

**Python**

```python
from calm.agent import run_agent

reply, history = run_agent("Load auto_mpg and run GES. What edges did we find?")
print(reply)
```

## What the agent can do

| Tool | Description |
|------|-------------|
| `load_data` | Load a CSV path or built-in dataset: `sachs`, `auto_mpg` |
| `list_discovery_methods` | List PC, GES, FCI, LiNGAM and when to use them |
| `run_causal_discovery` | Run one of PC, GES, FCI, or LiNGAM on the current data |
| `get_graph_description` | Return the last discovered graph (nodes, edges, DOT) |
| `estimate_effect` | Estimate causal effect of treatment on outcome (uses last graph) |

- **PC / GES / FCI**: constraint- or score-based; may return CPDAGs (undirected edges).
- **LiNGAM**: linear non-Gaussian; returns a DAG — use this before asking for effect estimates.

## Project layout

```
calm/
  calm/
    __init__.py
    agent.py          # LLM agent (OpenAI tool calling)
    tools/
      __init__.py
      discovery.py    # load_data, run_causal_discovery, estimate_effect
  requirements.txt
  run_agent.py       # CLI
  README.md
```

## Dependencies

- **causal-learn**: PC, GES, FCI, LiNGAM
- **DoWhy**: effect estimation (backdoor adjustment)
- **OpenAI**: agent with function calling

## License

MIT.
