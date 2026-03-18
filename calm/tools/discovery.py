"""
Causal discovery and effect estimation tools.

These are designed to be called by the LLM agent via tool/function calling.
State: we keep "current" dataset and "current" graph so the agent can chain
load_data -> run_causal_discovery -> estimate_effect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Lazy imports for heavy libs
_causal_learn = None
_dowhy = None


def _get_causal_learn():
    global _causal_learn
    if _causal_learn is None:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ScoreBased.GES import ges
        from causallearn.search.FCMBased import lingam
        from causallearn.utils.GraphUtils import GraphUtils
        from causallearn.utils.Dataset import load_dataset as cl_load_dataset
        _causal_learn = {
            "pc": pc,
            "ges": ges,
            "lingam": lingam,
            "GraphUtils": GraphUtils,
            "load_dataset": cl_load_dataset,
        }
    return _causal_learn


def _get_dowhy():
    global _dowhy
    if _dowhy is None:
        from dowhy import CausalModel
        _dowhy = {"CausalModel": CausalModel}
    return _dowhy


# In-memory state for the agent session
_current_data: pd.DataFrame | None = None
_current_labels: list[str] | None = None
_current_graph_dot: str | None = None
_current_method: str | None = None


def _graph_to_dot_lingam(adjacency_matrix: np.ndarray, labels: list[str]) -> str:
    """Convert LiNGAM adjacency matrix to DOT string."""
    n = adjacency_matrix.shape[0]
    lines = ["digraph {"]
    for i in range(n):
        for j in range(n):
            coef = adjacency_matrix[j, i]
            if abs(coef) > 1e-6:
                lines.append(f'  {labels[i]} -> {labels[j]} [label="{coef:.3f}"]')
    lines.append("}")
    return "\n".join(lines)


def _str_to_dot(s: str) -> str:
    """Convert graphviz source to valid DOT for DoWhy."""
    s = s.strip().replace("\n", " ").replace("\t", " ")
    return s


def load_data(path_or_name: str) -> dict[str, Any]:
    """
    Load a dataset for causal discovery.

    Args:
        path_or_name: Either a path to a CSV file, or a built-in name:
                      'sachs' (protein signaling), 'auto_mpg' (sample from UCI Auto-MPG).

    Returns:
        Summary with shape, columns, and sample.
    """
    global _current_data, _current_labels, _current_graph_dot, _current_method
    _current_graph_dot = None
    _current_method = None

    path_or_name = path_or_name.strip()
    if path_or_name.lower() == "sachs":
        cl = _get_causal_learn()
        data, labels = cl["load_dataset"]("sachs")
        _current_data = pd.DataFrame(data, columns=labels)
        _current_labels = list(labels)
    elif path_or_name.lower() == "auto_mpg":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
        df = pd.read_csv(
            url,
            delim_whitespace=True,
            header=None,
            names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"],
        )
        df = df.dropna().drop(columns=["model year", "origin", "car name"])
        _current_data = df
        _current_labels = list(df.columns)
    else:
        p = Path(path_or_name)
        if not p.exists():
            return {"ok": False, "error": f"File not found: {path_or_name}"}
        _current_data = pd.read_csv(p)
        _current_labels = list(_current_data.columns)

    return {
        "ok": True,
        "shape": _current_data.shape,
        "columns": _current_labels,
        "sample": _current_data.head(3).to_dict(),
    }


def list_discovery_methods() -> dict[str, Any]:
    """
    List available causal discovery methods and when to use them.

    Returns:
        Method names and short descriptions.
    """
    return {
        "ok": True,
        "methods": {
            "pc": "Constraint-based, uses conditional independence tests. Good default for general data; result is a CPDAG (may have undirected edges).",
            "ges": "Score-based (BIC). Fast; result is a CPDAG.",
            "fci": "Constraint-based, allows latent confounders and selection bias. Result may have bidirected edges.",
            "lingam": "Assumes linear non-Gaussian data. Returns a DAG (fully directed), so best for downstream effect estimation.",
        },
    }


def run_causal_discovery(method: str) -> dict[str, Any]:
    """
    Run causal discovery on the currently loaded dataset.

    Args:
        method: One of 'pc', 'ges', 'fci', 'lingam'.

    Returns:
        Summary of the discovered graph (edges, method used).
    """
    global _current_data, _current_labels, _current_graph_dot, _current_method

    if _current_data is None or _current_labels is None:
        return {"ok": False, "error": "No data loaded. Call load_data first."}

    method = method.lower().strip()
    cl = _get_causal_learn()
    data = _current_data.to_numpy()
    labels = _current_labels

    try:
        if method == "pc":
            cg = cl["pc"](data)
            pyd = cl["GraphUtils"].to_pydot(cg.G, labels=labels)
            graph_str = pyd.to_string()
            _current_graph_dot = _str_to_dot(graph_str)
            edges = _describe_causal_learn_graph(cg.G, labels)
        elif method == "ges":
            record = cl["ges"](data)
            pyd = cl["GraphUtils"].to_pydot(record["G"], labels=labels)
            graph_str = pyd.to_string()
            _current_graph_dot = _str_to_dot(graph_str)
            edges = _describe_causal_learn_graph(record["G"], labels)
        elif method == "fci":
            from causallearn.search.ConstraintBased.FCI import fci
            cg, _ = fci(data, alpha=0.05)
            pyd = cl["GraphUtils"].to_pydot(cg.G, labels=labels)
            graph_str = pyd.to_string()
            _current_graph_dot = _str_to_dot(graph_str)
            edges = _describe_causal_learn_graph(cg.G, labels)
        elif method == "lingam":
            model = cl["lingam"].ICALiNGAM()
            model.fit(data)
            _current_graph_dot = _graph_to_dot_lingam(model.adjacency_matrix_, labels)
            edges = _describe_lingam_graph(model.adjacency_matrix_, labels)
        else:
            return {"ok": False, "error": f"Unknown method: {method}. Use pc, ges, fci, or lingam."}

        _current_method = method
        return {
            "ok": True,
            "method": method,
            "nodes": labels,
            "edges_summary": edges,
            "note": "Use get_graph_description() for full graph. Use estimate_effect(treatment, outcome) if you have a DAG (e.g. from lingam).",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _describe_causal_learn_graph(G, labels: list[str]) -> list[str]:
    """Describe causal-learn graph edges for LLM."""
    n = G.get_num_nodes()
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # causal-learn: -1 = tail, 1 = arrowhead, 2 = circle
            # graph[i,j] describes edge from i to j: (i,j) entry
            val_ij = G.graph[i, j]
            val_ji = G.graph[j, i]
            if val_ij == 1 and val_ji == -1:
                edges.append(f"{labels[i]} -> {labels[j]}")
            elif val_ij == -1 and val_ji == 1:
                edges.append(f"{labels[j]} -> {labels[i]}")
            elif val_ij == -1 and val_ji == -1:
                edges.append(f"{labels[i]} - {labels[j]} (undirected)")
            elif val_ij == 1 and val_ji == 1:
                edges.append(f"{labels[i]} <-> {labels[j]} (bidirected)")
    return edges


def _describe_lingam_graph(adj: np.ndarray, labels: list[str]) -> list[str]:
    """Describe LiNGAM adjacency matrix (adj[j,i] = effect of i on j)."""
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(adj[j, i]) > 1e-6:
                edges.append(f"{labels[i]} -> {labels[j]} (coef={adj[j, i]:.3f})")
    return edges


def get_graph_description() -> dict[str, Any]:
    """
    Return a text description of the last discovered causal graph.

    Returns:
        DOT string and edges list, or error if no graph.
    """
    global _current_graph_dot, _current_method, _current_labels

    if _current_graph_dot is None:
        return {"ok": False, "error": "No graph yet. Run run_causal_discovery(method) first."}

    return {
        "ok": True,
        "method_used": _current_method,
        "nodes": _current_labels,
        "graph_dot": _current_graph_dot,
    }


def estimate_effect(treatment: str, outcome: str) -> dict[str, Any]:
    """
    Estimate the causal effect of treatment on outcome using the last discovered graph.

    Uses backdoor adjustment when possible. Works best when the last discovery method
    was 'lingam' (DAG). If the graph has undirected edges, identification may fail.

    Args:
        treatment: Name of the treatment variable (must be a column).
        outcome: Name of the outcome variable (must be a column).

    Returns:
        Estimate value, confidence interval, and method.
    """
    global _current_data, _current_labels, _current_graph_dot, _current_method

    if _current_data is None:
        return {"ok": False, "error": "No data loaded. Call load_data first."}
    if _current_graph_dot is None:
        return {"ok": False, "error": "No graph discovered. Run run_causal_discovery first (lingam recommended for effect estimation)."}

    for name in (treatment, outcome):
        if name not in _current_data.columns:
            return {"ok": False, "error": f"Variable '{name}' not in data. Columns: {list(_current_data.columns)}"}

    try:
        dowhy = _get_dowhy()
        model = dowhy["CausalModel"](
            data=_current_data,
            treatment=treatment,
            outcome=outcome,
            graph=_current_graph_dot,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
            confidence_intervals=True,
            test_significance=True,
        )
        return {
            "ok": True,
            "treatment": treatment,
            "outcome": outcome,
            "causal_estimate": estimate.value,
            "confidence_interval": getattr(estimate, "confidence_interval", None),
            "interpretation": f"A one-unit increase in {treatment} is associated with a {estimate.value:.4f} change in {outcome} (linear regression backdoor).",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
