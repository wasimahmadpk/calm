"""
LLM agent that performs causal discovery by calling discovery tools.

Uses OpenAI API with tool/function calling. The agent can load data,
run PC/GES/FCI/LiNGAM, inspect the graph, and estimate causal effects.
"""

from __future__ import annotations

import json
import os
from typing import Any

from calm.tools.discovery import (
    load_data,
    list_discovery_methods,
    run_causal_discovery,
    get_graph_description,
    estimate_effect,
)

# Tool implementations: name -> (fn, description, params_schema)
TOOL_IMPLEMENTATIONS = {
    "load_data": (
        load_data,
        "Load a dataset for causal discovery. Use a path to a CSV file, or a built-in name: 'sachs' (protein signaling), 'auto_mpg' (UCI Auto-MPG).",
        {
            "type": "object",
            "properties": {
                "path_or_name": {
                    "type": "string",
                    "description": "Path to CSV file or built-in name: sachs, auto_mpg",
                },
            },
            "required": ["path_or_name"],
        },
    ),
    "list_discovery_methods": (
        list_discovery_methods,
        "List available causal discovery methods (pc, ges, fci, lingam) with short descriptions. Call this to help the user choose a method.",
        {"type": "object", "properties": {}},
    ),
    "run_causal_discovery": (
        run_causal_discovery,
        "Run causal discovery on the currently loaded dataset. Must call load_data first. Method: pc, ges, fci, or lingam. Use lingam if the user wants to estimate causal effects later (it returns a DAG).",
        {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["pc", "ges", "fci", "lingam"],
                    "description": "Discovery algorithm to run",
                },
            },
            "required": ["method"],
        },
    ),
    "get_graph_description": (
        get_graph_description,
        "Get a text description of the last discovered causal graph (nodes and graph in DOT format). Call after run_causal_discovery.",
        {"type": "object", "properties": {}},
    ),
    "estimate_effect": (
        estimate_effect,
        "Estimate the causal effect of a treatment variable on an outcome variable using the last discovered graph. Best when the last discovery was lingam (DAG).",
        {
            "type": "object",
            "properties": {
                "treatment": {"type": "string", "description": "Name of the treatment variable"},
                "outcome": {"type": "string", "description": "Name of the outcome variable"},
            },
            "required": ["treatment", "outcome"],
        },
    ),
}


def build_openai_tools() -> list[dict]:
    """Build OpenAI tool definitions for the client."""
    tools = []
    for name, (_, description, params) in TOOL_IMPLEMENTATIONS.items():
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params,
            },
        })
    return tools


def run_tool(name: str, arguments: dict) -> Any:
    """Execute a tool by name with given arguments. Returns result (often a dict)."""
    if name not in TOOL_IMPLEMENTATIONS:
        return {"error": f"Unknown tool: {name}"}
    fn, _, _ = TOOL_IMPLEMENTATIONS[name]
    return fn(**arguments)


def run_agent(
    user_message: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_turns: int = 15,
) -> tuple[str, list[dict]]:
    """
    Run the causal discovery agent: send the user message to the LLM and
    execute tool calls until the model returns a final answer.

    Returns:
        (final assistant message, full message history).
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key to run_agent.")

    system = """You are a causal discovery assistant. You help users:
1. Load datasets (CSV or built-in like 'sachs', 'auto_mpg').
2. Run causal discovery (PC, GES, FCI, or LiNGAM) and explain the results.
3. Estimate causal effects when the user asks (e.g. effect of X on Y).

Use the tools in order: load_data first if they have data, then list_discovery_methods or run_causal_discovery. If they want an effect estimate, run_causal_discovery with method "lingam" first if needed, then estimate_effect. Explain findings in plain language and mention assumptions (e.g. LiNGAM assumes linear non-Gaussian)."""

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    tools_spec = build_openai_tools()

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_spec if tools_spec else None,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(name, args)
                result_str = json.dumps(result, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
            continue

        # No tool calls: final text response
        final = (msg.content or "").strip()
        messages.append({"role": "assistant", "content": final})
        return final, messages

    return "I hit the turn limit. Please try a shorter request.", messages
