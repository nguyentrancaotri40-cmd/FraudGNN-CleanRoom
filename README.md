# FraudGNN-RL — Reproduction (clean-room)

Dựng lại từ **mô tả** trong paper:

> Cui et al., *FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection*,
> IEEE Open Journal of the Computer Society, vol. 6, 2025. DOI 10.1109/OJCS.2025.3543450

Paper **không công bố source code**. Đây là *reproduction from description*, không phải replication: công thức nào paper
có thì code theo đúng công thức; chỗ paper không nói, hoặc tự mâu thuẫn, thì code chọn một phương án và **ghi rõ**
trong [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md). Hãy đọc file đó trước khi dùng bất kỳ con số nào trong repo này.

## Trạng thái hiện tại (đọc trước)

- Repo là nhánh **clean-room**: chỉ chứa các thành phần của paper. Không có soft edges, pruning, hybrid graph hay ablation ngoài paper.
  Kiểm tra tự động: `python tools/check_clean_room.py`.
- **Chưa có kết quả đầy đủ nào được báo cáo trong repo này.** Lần smoke test gần nhất (PaySim 5%, 2 round FL) cho F1 = 0
  tại ngưỡng do RL chọn. Nguyên nhân khả dĩ và các chỉnh sửa tương ứng nằm ở `KNOWN_LIMITATIONS.md` mục 9. Các chỉnh sửa
  đó **chưa được chạy lại trên dữ liệu thật**, nên chưa biết kết quả.
- Repo **không chứa và không sinh dữ liệu**. Bạn phải tự đặt dữ liệu thật vào `data/raw/`; thiếu file thì code báo lỗi.
- Không so trực tiếp từng chữ số với Table 2 của paper. Paper có mâu thuẫn số học nội bộ (`KNOWN_LIMITATIONS.md` C6–C8).

## Paper nói gì và repo làm gì

| Thành phần | Paper | Repo | Nhãn |
|---|---|---|---|
| Node / edge | Mỗi giao dịch là một node; nối nếu cách nhau ≤ 1 giờ và cosine ≥ 0.9 (V-A-4) | Đúng như vậy, tìm kiếm chính xác theo cửa sổ thời gian, vô hướng, không self-loop | [PAPER] + hướng cạnh [INFERRED] |
| TSSGC | 3 lớp, hidden 64, Eq. 5–11, BatchNorm sau mỗi lớp, ReLU | Đúng. Temporal = GRU + attention suy giảm theo thời gian (Eq. 6–7); Spatial = GAT 1 head (Eq. 8–9); Semantic = `W_m[h‖e_type]` (Eq. 10); gộp theo Eq. 11 | [PAPER]; cách đưa chuỗi vào GRU, dấu của β [INFERRED] |
| RL | NAF (Gu et al. 2016), Q-net 2×128, Adam 1e-3, loss Eq. 12 với target network; action = (ngưỡng, trọng số đặc trưng) | NAF mặc định, DQN rời rạc 19 bin là tuỳ chọn | Kiến trúc [PAPER]; reward, state, episode [INFERRED] |
| Reward | "kết hợp độ chính xác và FPR", **không có công thức** | `F1 + 0.25·Recall − 1.0·FPR` trên một cửa sổ giao dịch | [INFERRED] tự thiết kế |
| Federated | FedAvg, trung bình **đều** 1/\|C_t\| (Algorithm 1) | `fedavg_uniform`, 3 client chia theo thời gian, 50 round × 2 epoch cục bộ | Thuật toán [PAPER]; số client, cách chia [INFERRED] |
| Graph alignment (FL) | Có nhắc, **không có chi tiết** | Giả thuyết của người thực hiện: schema đặc trưng chung + lớp projection dùng chung được FedAvg | [UNREPRODUCIBLE] |
| Huấn luyện | Adam 1e-3, batch 64, 100 epoch, early stopping 10, seed 42 | Giống, trừ batch (xem dưới) | [PAPER]; batch [DEVIATION] |
| Metric | AUC-ROC, AUC-PR, F1, Recall@1% | Đúng 4 metric này (+ precision, FPR, TP/FP/TN/FN để chẩn đoán) | [PAPER] |
| Ablation | w/o GNN, w/o RL, w/o FL (và tổ hợp) | Bật/tắt bằng `flags.gnn / flags.rl / flags.federated` | [PAPER] |

### Những gì repo **không** làm được, và lý do

| Việc | Lý do |
|---|---|
| Batch size 64 trên đồ thị giao dịch | Message passing 3 lớp cần láng giềng 3-hop; batch 64 node không khả thi trên đồ thị hàng triệu node. Dùng `batch_mode: full` hoặc `subgraph` (cạnh xuyên chunk bị bỏ). |
| Graph alignment đúng như paper | Paper không mô tả. |
| Vòng phản hồi online trong Fig. 2 | Paper cũng liệt kê online learning là future work. Repo chạy offline: RL huấn luyện trên **val**, áp lên **test**, không cập nhật trên test. |
| Thí nghiệm concept drift / adversarial của paper | Paper tuyên bố kháng drift/tấn công nhưng Sec. V không có giao thức. |
| Không giới hạn số láng giềng trên dataset lớn | Có thể hết bộ nhớ. Config chính dùng `max_neighbors_per_node` (ghi trong `deviations_from_paper`). Có `*_strict.yaml` không cap, **chưa kiểm chứng bộ nhớ**. |
| Khớp Table 2 cho Credit Card 2023 | Mô tả của paper (có thời gian giao dịch, fraud 0.172%) khớp bộ ULB 2013 hơn bản 2023 (xem `KNOWN_LIMITATIONS.md` 4.2; **cần xác minh bằng log `[LOAD]`**). |
| Môi trường của paper | Paper: PyTorch 1.8.0, PyG 2.0.1, V100 + 32 GB. Repo cần phiên bản mới hơn. |

## Cài đặt

```bash
pip install -r requirements.txt
```

Phiên bản thực tế đã dùng được ghi vào mỗi file kết quả (`env`). Cập nhật `requirements_freeze.txt` theo môi trường bạn chạy.

## Dữ liệu (tự cung cấp)

| Dataset | File | Ghi chú |
|---|---|---|
| PaySim | `data/raw/paysim_fast.csv` (đổi trong `configs/paysim*.yaml`) | Paper: 6,362,620 giao dịch, 8,213 fraud. Hãy kiểm tra file của bạn bằng dòng log `[LOAD]`. |
| Credit Card 2023 | `data/raw/creditcard_2023.csv` | Kiểm tra `labels=` và có/không cột thời gian. |
| IEEE-CIS | `data/raw/train_transaction.csv`, `data/raw/train_identity.csv` | Hợp nhất bằng `TransactionID` (LEFT JOIN). Chỉ có tập train công khai có nhãn. |

Tiền xử lý đúng như paper V-A-1-d: impute mean/mode, min-max, one-hot, đặc trưng thời gian (giờ trong ngày, thứ trong tuần, Δt).
Fit chỉ trên train.

## Chạy

```bash
python main.py --config configs/paysim.yaml            # cấu hình chính (practical)
python main.py --config configs/paysim_strict.yaml     # không cap kỹ thuật (có thể hết bộ nhớ)
python main.py --config configs/paysim.yaml --cv       # 5-fold CV (mean ± std)
python main.py --config configs/paysim.yaml --seed 7
python main.py --config configs/paysim.yaml --set flags.rl=false          # ablation "w/o RL"
python main.py --config configs/paysim.yaml --set flags.federated=false   # ablation "w/o FL"
python main.py --config configs/paysim.yaml --set flags.gnn=false         # ablation "w/o GNN"
python tools/check_clean_room.py                       # kiểm tra clean-room
```

Config `*_sample.yaml`, `test.yaml`, `smoke_test.yaml` chỉ để kiểm tra luồng chạy (`sample_frac < 1`, ít epoch) và
**không phải kết quả paper**.

## Đọc kết quả

Kết quả lưu ở `outputs/results/<experiment>_metrics.json`. Các trường quan trọng:

| Trường | Ý nghĩa |
|---|---|
| `test_metrics` | **Số chính thức**: ngưỡng do RL chọn (hoặc 0.5 nếu tắt RL) |
| `diagnostic_val_best_f1_threshold` | Chẩn đoán: ngưỡng tối đa F1 chọn trên val, áp lên test. **Không phải số của paper**; dùng để biết lỗi nằm ở mô hình hay ở RL |
| `rl_info` | Cỡ cửa sổ RL hiệu dụng, số fraud trung bình mỗi cửa sổ, tỉ lệ cửa sổ không có fraud, reward đầu/cuối |
| `graph_info` | Số cạnh, bậc trung bình, tỉ lệ fraud của train/val/test |
| `deviations_from_paper` | Mọi cap kỹ thuật/giả thuyết đang bật trong lần chạy này |
| `env` | Phiên bản torch / PyG / scikit-learn / numpy |

Khi báo cáo kèm theo: số dòng dữ liệu thật, tỉ lệ fraud, danh sách `deviations_from_paper`, và khuyến nghị báo cáo cả
cấu hình *strict* lẫn *practical*.

### Lưu ý về RL trong bản này

Với fraud ≈ 0.1%, một batch nhỏ thường không có fraud nên reward trở thành `−FPR` và policy học "không báo gì". Vì vậy cửa sổ
RL được nâng để chứa tối thiểu `rl.env_min_pos_per_batch` (mặc định 20) fraud, lấy theo tỉ lệ fraud của **val**
(`KNOWN_LIMITATIONS.md` R13–R14). Khi chạy, hãy xem dòng `[RL] fraud/cửa sổ trung bình=...`: phải ≫ 1.

## Cấu trúc thư mục

```
configs/                 YAML cho 3 dataset (chính, strict, sample) + smoke_test/test
main.py                  điểm vào
src/data/                load_data, preprocess, split
src/graph/               graph_utils (tìm cạnh chính xác), build_graph
src/models/              tssgc (Eq. 5-11), fraudgnn_rl, classifier, naf_agent, dqn_agent, reward (môi trường RL)
src/train/               train_gnn (centralized), federated (Algorithm 1), train_rl, pipeline_fraudgnn
src/eval/                metrics, evaluate
src/utils/               config, seed, logger, timer, model_definitions (tên biến thể ablation)
tools/check_clean_room.py
KNOWN_LIMITATIONS.md     mọi điểm [INFERRED] / [DEVIATION] / [CONTRADICTION]
```

### Mã không thuộc luồng chính

`src/models/baselines.py` (8 baseline), `src/eval/concept_drift.py`, `src/eval/adversarial.py` **không được gọi bởi `main.py`**
và không phải một phần của việc tái hiện FraudGNN-RL. Paper chỉ nêu tên baseline, không có siêu tham số, nên toàn bộ siêu
tham số trong các file đó là [INFERRED] (`KNOWN_LIMITATIONS.md` mục 5). Hiện **chưa có unit test** nào trong repo.

## Trích dẫn

Nếu dùng repo này, vui lòng trích dẫn paper gốc ở trên và nêu rõ đây là bản reproduction from description kèm
`KNOWN_LIMITATIONS.md`.
