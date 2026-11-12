import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(json_path: str, output_dir: str = "outputs/figures"):
    # Đọc file JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy metrics
    test_m = data.get("test_metrics", {})
    val_m = data.get("val_metrics", {})
    
    if not test_m:
        print("Không tìm thấy test_metrics trong file JSON.")
        return

    # Các chỉ số cần vẽ
    metrics = ["auc_roc", "auc_pr", "f1", "recall_at_1pct"]
    labels = ["AUC-ROC", "AUC-PR", "F1-score", "Recall@1%"]
    test_vals = [test_m.get(k, 0) for k in metrics]
    val_vals = [val_m.get(k, 0) for k in metrics]

    # Tạo thư mục output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vẽ biểu đồ cột so sánh Val vs Test
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_vals, width, label='Validation', color='skyblue')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('FraudGNN-RL: Validation vs Test Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Ghi số liệu lên đầu mỗi cột
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "metrics_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Đã lưu biểu đồ tại: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python plot_metrics.py <đường_dẫn_file_json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not Path(json_file).exists():
        print(f"Không tìm thấy file: {json_file}")
        sys.exit(1)
        
    plot_metrics(json_file)