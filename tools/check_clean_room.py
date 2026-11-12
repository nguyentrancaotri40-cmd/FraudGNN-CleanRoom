#!/usr/bin/env python3
"""Kiem tra repo la clean-room reproduction (khong lan soft-edge / pruning / hybrid).

Chay:  python tools/check_clean_room.py        (exit 0 = sach, 1 = con dau vet)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATHS = [
    "src/extensions", "src/graph/hybrid_graph.py", "src/graph/soft_behavior_graph.py",
    "scripts/run", "scripts/compare.py", "audit_runtime.py", "note.txt", "EXTENSIONS.md",
]
FORBIDDEN_TOKENS = re.compile(
    r"soft_edges|soft_graph|soft_behavior|hybrid|weighted_fusion|pruning|magnitude_prune|"
    r"augmented_graph|FraudGNN-RL\+|hard_edges|ablation_soft|ablation_pruning|run_ablation",
    re.IGNORECASE,
)
SCAN_SUFFIX = {".py", ".yaml", ".yml", ".md", ".txt"}
SKIP_DIRS = {".git", "venv", "__pycache__", "outputs", "data"}
SKIP_FILES = {"check_clean_room.py", "KNOWN_LIMITATIONS.md"}  # file nay nhac den cac tu khoa de giai thich

bad = []
for rel in FORBIDDEN_PATHS:
    if (ROOT / rel).exists():
        bad.append(f"[PATH ] ton tai: {rel}")

for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix not in SCAN_SUFFIX or p.name in SKIP_FILES:
        continue
    if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
        continue
    for ln, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        m = FORBIDDEN_TOKENS.search(line)
        if m:
            bad.append(f"[TOKEN] {p.relative_to(ROOT)}:{ln}: '{m.group(0)}'  | {line.strip()[:90]}")

# Cac gia tri 'paper' khong duoc lech trong config goc (khong tinh *_sample / test / smoke)
try:
    import yaml
    for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
        if any(k in cfg_path.stem for k in ("sample", "test", "smoke")):
            continue
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        g, ds = cfg.get("graph", {}), cfg.get("dataset", {})
        if ds.get("sample_frac") not in (None, 1.0):
            bad.append(f"[CFG  ] {cfg_path.name}: sample_frac={ds['sample_frac']} (khong phai paper)")
        if g.get("time_window_hours", 1.0) not in (1.0, None):
            bad.append(f"[CFG  ] {cfg_path.name}: time_window_hours={g['time_window_hours']} (paper: 1.0)")
        if g.get("similarity_threshold", 0.9) != 0.9:
            bad.append(f"[CFG  ] {cfg_path.name}: similarity_threshold={g['similarity_threshold']} (paper: 0.9)")
except ImportError:
    print("[WARN] thieu pyyaml, bo qua kiem tra config")

if bad:
    print("KHONG SACH - con dau vet ngoai paper:\n" + "\n".join(bad))
    sys.exit(1)
print("OK: clean-room, khong con dau vet extension.")
