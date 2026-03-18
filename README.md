# CALM: Causal discovery with an Agent and LLM

Use an LLM-powered agent to run **causal discovery** and **effect estimation** via natural language. The agent can load data, run algorithms (PC, GES, FCI, LiNGAM), and estimate causal effects.

## Setup

```bash
cd calm
pip install -r requirements.txt
```

**Run locally (recommended: use a venv to avoid NumPy conflicts with Anaconda):**
```bash
# One-time setup
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit .env and set OPENAI_API_KEY=sk-...

# Run (single question)
python run_agent.py "Load the sachs dataset and run PC causal discovery"

# Or chat continuously (type 'quit' to exit)
python run_agent.py --chat
```
Or use the script: `chmod +x run_local.sh && ./run_local.sh "Load sachs and run PC"` (uses `.venv` if present).

**Keep your API key private (never in code or git):**

1. Copy the example env file and add your key only in the local file:
   ```bash
   cp .env.example .env
   # Edit .env and set: OPENAI_API_KEY=sk-your-actual-key
   ```
2. `.env` is gitignored—it will not be committed. Only you (and your code running locally) can read it.

Alternatively, set the key in your shell: `export OPENAI_API_KEY=sk-...` (don’t paste keys in shared docs or chat).

**If a key was ever exposed** (e.g. pasted in chat or committed), [rotate it immediately](https://platform.openai.com/api-keys) and use the new key only in `.env`.

**Using Cursor subscription (OpenAI-compatible proxy)**  
Cursor doesn’t expose a public chat API; use an OpenAI-compatible proxy that talks to Cursor’s models (e.g. [cursor-api-proxy](https://github.com/anyrobert/cursor-api-proxy)). Then point CALM at the proxy:

```bash
# Start the proxy (in another terminal), then:
export CALM_LLM_BASE_URL=http://127.0.0.1:8765/v1
export CURSOR_API_KEY=your-proxy-key   # if the proxy requires one
export CALM_LLM_MODEL=default          # or a specific model ID from the proxy
python run_agent.py "Load sachs and run LiNGAM"
```

## Quick start

**CLI — single question**

```bash
python run_agent.py "Load the sachs dataset and run PC causal discovery"
python run_agent.py "Load sachs, run LiNGAM, and estimate the effect of pip2 on pkc"
```

**CLI — interactive chat (keep talking until you type `quit` or `exit`)**

```bash
python run_agent.py --chat
```
Then type your questions; the agent keeps context (e.g. "Show me the graph", "What's the effect of X on Y?").

**Python**

```python
from calm.agent import run_agent

reply, history = run_agent("Load auto_mpg and run GES. What edges did we find?")
print(reply)
```

## What the agent can do

| Tool | Description |
|------|-------------|
| `load_data` | Load a CSV path or built-in dataset: `sachs`, `auto_mpg`, `simset` (simulated: x→y, x→z) |
| `list_discovery_methods` | List PC, GES, FCI, LiNGAM and when to use them |
| `run_causal_discovery` | Run one of PC, GES, FCI, or LiNGAM on the current data |
| `get_graph_description` | Return the last discovered graph (nodes, edges, DOT) |
| `get_metrics` | Compare discovered vs ground-truth graph (edge/arrow precision, recall, F1, SHD). Only for **simset** (true: x→y, x→z). |
| `visualize_graph` | Draw the causal graph with NetworkX (circular layout), save as PNG, and open it in your default viewer |
| `estimate_effect` | Estimate causal effect of treatment on outcome (uses last graph) |

- **PC / GES / FCI**: constraint- or score-based; may return CPDAGs (undirected edges).
- **LiNGAM**: linear non-Gaussian; returns a DAG — use this before asking for effect estimates.
- **Graph**: Ask "show me the graph" after discovery; the image opens automatically (`causal_graph.png`).

## Project layout

```
calm/
  calm/
    __init__.py
    agent.py          # LLM agent (OpenAI tool calling)
    tools/
      __init__.py
      discovery.py    # load_data, run_causal_discovery, visualize_graph, estimate_effect
  requirements.txt
  run_agent.py       # CLI
  README.md
```

## LLM / API options

| Env / arg | Purpose |
|-----------|--------|
| `OPENAI_API_KEY` | API key for OpenAI (or for the proxy, if it uses one). |
| `CALM_LLM_BASE_URL` | Use another OpenAI-compatible endpoint (e.g. Cursor proxy, local server). |
| `CALM_LLM_MODEL` | Model ID (default: `gpt-4o-mini`). For Cursor proxy use `default` or the proxy’s model list. |
| `CURSOR_API_KEY` | Alternative key env when using a Cursor proxy. |

So you can use **OpenAI**, **Cursor (via proxy)**, or any **OpenAI-compatible** API (e.g. LiteLLM, local LLMs).

## Dependencies

- **causal-learn**: PC, GES, FCI, LiNGAM
- **DoWhy**: effect estimation (backdoor adjustment)
- **networkx** + **matplotlib**: causal graph visualization (circular layout)
- **openai**: client (works with OpenAI and any OpenAI-compatible endpoint)

## License

MIT.
