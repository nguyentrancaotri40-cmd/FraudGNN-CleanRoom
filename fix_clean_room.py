# fix_clean_room.py
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1. Xóa file rác
files_to_delete = [
    "src/utils/model_definitions.py",
    "full_code_dump.txt", "audit_runtime.py", "note.txt", "verify_all.py",
]
for f in files_to_delete:
    p = ROOT / f
    if p.exists():
        p.unlink()
        print(f"[DELETE] {f}")

ext_dir = ROOT / "src/extensions"
if ext_dir.exists():
    shutil.rmtree(ext_dir)
    print("[DELETE] src/extensions/")

# 2. Các file cần dọn dẹp (CHỈ xóa dòng chứa từ khóa cấm)
FILES = [
    "configs/paysim.yaml", "configs/ieee_cis.yaml", "configs/creditcard2023.yaml",
    "configs/paysim_sample.yaml", "configs/ieee_cis_sample.yaml", "configs/creditcard2023_sample.yaml",
    "configs/smoke_test.yaml", "configs/test.yaml",
    "src/utils/config.py",
    "src/graph/build_graph.py",
    "src/train/federated.py",
    "src/train/pipeline_fraudgnn.py",
    "src/train/train_gnn.py",
    "README.md",
]

# Xóa các dòng chứa từ khóa cấm
KEYWORDS = ["soft_edges", "soft_graph", "pruning", "hard_edges", "hybrid", "EXTENSION", "NGOÀI paper"]

def clean_file(filepath: Path):
    if not filepath.exists():
        return
    lines = filepath.read_text(encoding="utf-8").splitlines()
    new_lines = [ln for ln in lines if not any(kw in ln for kw in KEYWORDS)]
    new_text = "\n".join(new_lines)
    if new_text != "\n".join(lines):
        filepath.write_text(new_text, encoding="utf-8")
        print(f"[CLEAN] {filepath}")

for f in FILES:
    clean_file(ROOT / f)

print("\n✅ Đã dọn dẹp. Chạy lại check_clean_room.py để kiểm tra.")