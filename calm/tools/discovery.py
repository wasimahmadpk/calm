"""
Causal discovery and effect estimation tools.

These are designed to be called by the LLM agent via tool/function calling.
State: we keep "current" dataset and "current" graph so the agent can chain
load_data -> run_causal_discovery -> estimate_effect.
"""

from __future__ import annotations

import os
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
_current_edges: list[tuple[str, str, str]] | None = None  # (u, v, 'directed'|'undirected'|'bidirected')
_true_edges: list[tuple[str, str]] | None = None  # ground-truth directed edges (only set for simset)


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


def _generate_simset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulated time series with 3 variables: x causes y, x causes z.
    Structure: x -> y, x -> z (no direct y-z or z-y).
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n)) * 0.1  # random walk scaled
    y = np.zeros(n)
    z = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * x[t] + 0.3 * y[t - 1] + 0.1 * rng.standard_normal()
        z[t] = 0.4 * x[t] + 0.3 * z[t - 1] + 0.1 * rng.standard_normal()
    return pd.DataFrame({"x": x, "y": y, "z": z})


def load_data(path_or_name: str) -> dict[str, Any]:
    """
    Load a dataset for causal discovery.

    Args:
        path_or_name: Either a path to a CSV file, or a built-in name:
                      'sachs' (protein signaling), 'auto_mpg' (UCI Auto-MPG), 'simset' (simulated: x causes y, x causes z).

    Returns:
        Summary with shape, columns, and sample.
    """
    global _current_data, _current_labels, _current_graph_dot, _current_method, _current_edges, _true_edges
    _current_graph_dot = None
    _current_method = None
    _current_edges = None
    _true_edges = None

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
    elif path_or_name.lower() == "simset":
        _current_data = _generate_simset()
        _current_labels = ["x", "y", "z"]
        _true_edges = [("x", "y"), ("x", "z")]  # ground truth for metrics
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
    global _current_data, _current_labels, _current_graph_dot, _current_method, _current_edges

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
            edges, _current_edges = _causal_learn_edges(cg.G, labels)
        elif method == "ges":
            record = cl["ges"](data)
            pyd = cl["GraphUtils"].to_pydot(record["G"], labels=labels)
            graph_str = pyd.to_string()
            _current_graph_dot = _str_to_dot(graph_str)
            edges, _current_edges = _causal_learn_edges(record["G"], labels)
        elif method == "fci":
            from causallearn.search.ConstraintBased.FCI import fci
            cg, _ = fci(data, alpha=0.05)
            pyd = cl["GraphUtils"].to_pydot(cg.G, labels=labels)
            graph_str = pyd.to_string()
            _current_graph_dot = _str_to_dot(graph_str)
            edges, _current_edges = _causal_learn_edges(cg.G, labels)
        elif method == "lingam":
            model = cl["lingam"].ICALiNGAM()
            model.fit(data)
            _current_graph_dot = _graph_to_dot_lingam(model.adjacency_matrix_, labels)
            edges = _describe_lingam_graph(model.adjacency_matrix_, labels)
            _current_edges = [(labels[i], labels[j], "directed") for i in range(len(labels)) for j in range(len(labels)) if abs(model.adjacency_matrix_[j, i]) > 1e-6]
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


def _causal_learn_edges(G, labels: list[str]) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (text edges for LLM, structured edges for visualization)."""
    n = G.get_num_nodes()
    text_edges = []
    struct_edges: list[tuple[str, str, str]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val_ij = G.graph[i, j]
            val_ji = G.graph[j, i]
            if val_ij == 1 and val_ji == -1:
                text_edges.append(f"{labels[i]} -> {labels[j]}")
                struct_edges.append((labels[i], labels[j], "directed"))
            elif val_ij == -1 and val_ji == 1:
                text_edges.append(f"{labels[j]} -> {labels[i]}")
                struct_edges.append((labels[j], labels[i], "directed"))
            elif val_ij == -1 and val_ji == -1:
                text_edges.append(f"{labels[i]} - {labels[j]} (undirected)")
                struct_edges.append((labels[i], labels[j], "undirected"))
            elif val_ij == 1 and val_ji == 1:
                text_edges.append(f"{labels[i]} <-> {labels[j]} (bidirected)")
                struct_edges.append((labels[i], labels[j], "bidirected"))
    return text_edges, struct_edges


def _describe_causal_learn_graph(G, labels: list[str]) -> list[str]:
    """Describe causal-learn graph edges for LLM."""
    text_edges, _ = _causal_learn_edges(G, labels)
    return text_edges


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


def get_metrics() -> dict[str, Any]:
    """
    Compare the discovered causal graph to the ground-truth graph and return metrics.
    Only available when the loaded dataset has known ground truth (e.g. simset: x→y, x→z).
    Call after run_causal_discovery.
    """
    global _current_edges, _true_edges, _current_method

    if _true_edges is None:
        return {
            "ok": False,
            "error": "No ground-truth graph for this dataset. Metrics are only available for 'simset' (true graph: x→y, x→z).",
        }
    if _current_edges is None:
        return {"ok": False, "error": "No graph discovered yet. Run run_causal_discovery(method) first."}

    true_set = set(_true_edges)
    true_undirected = set((min(u, v), max(u, v)) for u, v in true_set)

    discovered_directed = [(u, v) for u, v, t in _current_edges if t == "directed"]
    discovered_undirected = set((min(u, v), max(u, v)) for u, v, t in _current_edges if t == "undirected")
    discovered_pairs = set((min(u, v), max(u, v)) for u, v in discovered_directed) | discovered_undirected

    # Edge metrics (edge present, direction ignored)
    tp_edge = sum(1 for (u, v) in true_set if (min(u, v), max(u, v)) in discovered_pairs)
    fp_edge = len(discovered_pairs) - tp_edge
    edge_recall = tp_edge / len(true_set) if true_set else 0.0
    edge_precision = tp_edge / len(discovered_pairs) if discovered_pairs else 0.0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0.0

    # Arrow metrics (correct direction)
    discovered_directed_set = set(discovered_directed)
    tp_arrow = sum(1 for (u, v) in true_set if (u, v) in discovered_directed_set)
    fp_arrow = len(discovered_directed) - tp_arrow
    arrow_recall = tp_arrow / len(true_set) if true_set else 0.0
    arrow_precision = tp_arrow / len(discovered_directed) if discovered_directed else 0.0
    arrow_f1 = 2 * arrow_precision * arrow_recall / (arrow_precision + arrow_recall) if (arrow_precision + arrow_recall) > 0 else 0.0

    # Structural Hamming Distance: missing + extra + reversed
    missing = len(true_set) - tp_arrow
    extra = fp_arrow
    reversed_ = sum(1 for (u, v) in true_set if (v, u) in discovered_directed_set)
    shd = missing + extra + reversed_

    return {
        "ok": True,
        "true_graph": [f"{u}→{v}" for u, v in _true_edges],
        "method": _current_method,
        "metrics": {
            "edge_precision": round(edge_precision, 4),
            "edge_recall": round(edge_recall, 4),
            "edge_f1": round(edge_f1, 4),
            "arrow_precision": round(arrow_precision, 4),
            "arrow_recall": round(arrow_recall, 4),
            "arrow_f1": round(arrow_f1, 4),
            "structural_hamming_distance": shd,
        },
        "interpretation": "Edge metrics ignore direction; arrow metrics require correct direction. SHD = missing + extra + reversed edges (lower is better).",
    }


def visualize_graph(save_path: str = "causal_graph.png") -> dict[str, Any]:
    """
    Draw the last discovered causal graph with NetworkX and save as a PNG.

    Args:
        save_path: Filename or path for the image (default: causal_graph.png in project root).

    Returns:
        ok, path where the image was saved, or error.
    """
    global _current_labels, _current_edges, _current_method

    if _current_edges is None or _current_labels is None:
        return {"ok": False, "error": "No graph yet. Run run_causal_discovery(method) first."}

    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from pathlib import Path

        G = nx.DiGraph()
        G.add_nodes_from(_current_labels)
        directed_edges: list[tuple[str, str]] = []
        undirected_edges: list[tuple[str, str]] = []
        bidirected_edges: list[tuple[str, str]] = []
        seen_undirected = set()
        seen_bidirected = set()
        for u, v, etype in _current_edges:
            if etype == "directed":
                directed_edges.append((u, v))
                G.add_edge(u, v)
            elif etype == "undirected":
                key = (min(u, v), max(u, v))
                if key not in seen_undirected:
                    seen_undirected.add(key)
                    undirected_edges.append((u, v))
                    G.add_edge(u, v)
                    G.add_edge(v, u)
            elif etype == "bidirected":
                key = (min(u, v), max(u, v))
                if key not in seen_bidirected:
                    seen_bidirected.add(key)
                    bidirected_edges.append((u, v))
                    G.add_edge(u, v)
                    G.add_edge(v, u)

        # Resolve save path to project root if relative (calm/tools/discovery.py -> project root)
        p = Path(save_path)
        if not p.is_absolute():
            _root = Path(__file__).resolve().parent.parent.parent  # tools -> calm -> project
            p = _root / save_path
        p = p.with_suffix(".png")
        save_path = str(p)

        # Circular layout: nodes on a circle
        pos = nx.circular_layout(G)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("white")

        # Node size (same value for nodes and edge trimming so arrows don't hide under nodes)
        node_size = 2800

        # Draw undirected as simple lines (no arrows)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=undirected_edges,
            edge_color="#6c757d", width=2, alpha=0.8, ax=ax,
            connectionstyle="arc3,rad=0.1", node_size=node_size,
        )
        # Draw bidirected (one arc per pair)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=bidirected_edges,
            edge_color="#9b59b6", width=2, alpha=0.8, ax=ax,
            connectionstyle="arc3,rad=0.2", node_size=node_size,
        )
        # Draw directed edges (arrows) — node_size shortens edges so arrowheads sit at node border
        nx.draw_networkx_edges(
            G, pos,
            edgelist=directed_edges,
            edge_color="#2c3e50", width=2.5, alpha=0.9, ax=ax,
            arrows=True, arrowsize=22, arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
            node_size=node_size,
        )

        # Nodes: soft gradient feel
        node_colors = ["#3498db"] * G.number_of_nodes()
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_size,
            alpha=0.95, edgecolors="#2c3e50", linewidths=2, ax=ax,
        )
        nx.draw_networkx_labels(
            G, pos, font_size=10, font_weight="600", font_family="sans-serif",
            ax=ax,
        )

        ax.set_title(f"Causal graph ({_current_method or 'discovery'})", fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        # Open the image with the system default viewer so the user sees it
        import subprocess
        try:
            if hasattr(os, "startfile"):  # Windows
                os.startfile(save_path)
            else:
                try:
                    subprocess.run(["open", save_path], check=True, capture_output=True, timeout=2)  # macOS
                except (subprocess.CalledProcessError, FileNotFoundError):
                    subprocess.run(["xdg-open", save_path], capture_output=True, timeout=2)  # Linux
        except Exception:
            pass

        return {"ok": True, "path": save_path, "message": f"Graph saved and opened: {save_path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
