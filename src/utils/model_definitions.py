"""
Biến thể model của PAPER (Section V-B-2, ablation). Clean-room: không có biến thể ngoài paper.
"""

PAPER_ABLATIONS = {
    "Full model": {"gnn": True, "rl": True, "federated": True},
    "w/o GNN": {"gnn": False, "rl": True, "federated": True},
    "w/o RL": {"gnn": True, "rl": False, "federated": True},
    "w/o FL": {"gnn": True, "rl": True, "federated": False},
    "w/o GNN&RL": {"gnn": False, "rl": False, "federated": True},
    "w/o GNN&FL": {"gnn": False, "rl": True, "federated": False},
    "w/o RL&FL": {"gnn": True, "rl": False, "federated": False},
}


def resolve_model_name(flags: dict) -> str:
    """Tên model từ 3 cờ của paper (gnn / rl / federated)."""
    cur = {k: bool(flags.get(k, True)) for k in ("gnn", "rl", "federated")}
    for name, f in PAPER_ABLATIONS.items():
        if f == cur:
            return f"FraudGNN-RL ({name})"
    return "FraudGNN-RL (custom)"
