# KNOWN LIMITATIONS & TRANSPARENCY REPORT

Reproduction của **FraudGNN-RL** (Cui et al., IEEE Open J. Comput. Soc., vol. 6, 2025, DOI 10.1109/OJCS.2025.3543450).

Tài liệu này liệt kê **mọi** chỗ bản reproduction này KHÔNG phải là bản sao trực tiếp của paper. Paper không kèm code chính thức ("code will be made available upon publication"), nên đây là *reproduction from description*, không phải replication.

Nhãn dùng trong code và tài liệu:

| Nhãn | Ý nghĩa |
|---|---|
| **[PAPER]** | Paper nêu rõ (công thức hoặc con số). Code khớp. |
| **[CONTRADICTION]** | Paper tự mâu thuẫn. Tác giả repo phải chọn một phía. |
| **[INFERRED]** | Paper không nói. Tác giả repo tự chọn. |
| **[DEVIATION]** | Cố ý khác paper hoặc bị giới hạn kỹ thuật (bộ nhớ, thời gian chạy). |
| **[UNREPRODUCIBLE]** | Thiếu thông tin nên không thể phục dựng. |

> Giá trị tham số trong tài liệu này khớp với `configs/*.yaml` tại commit tương ứng. Nếu bạn sửa config, hãy cập nhật lại bảng ở mục 3.

---

## 1. Mâu thuẫn nội tại của paper [CONTRADICTION]

| # | Mâu thuẫn trong paper | Lựa chọn của repo | Lý do |
|---|---|---|---|
| C1 | Sec. III-A: node = entity (user/merchant), edge = giao dịch. Sec. V-A-4: "each transaction is a node". | **node = transaction**; edge = quan hệ giữa 2 giao dịch | Sec. V-A-4 mô tả cách chạy thực nghiệm thực tế; Sec. III-A chỉ là định nghĩa tổng quát. |
| C2 | Văn xuôi Sec. IV-A nói chọn "GCN đơn giản". Eq. 8-9 là công thức GAT (attention α_ij). | **GATConv 1 head** theo Eq. 8-9 | Công thức kiểm chứng được; văn xuôi có thể là lỗi biên tập. Không dùng GCN. |
| C3 | Sec. V-A-1c: "80-20 **random** split **while preserving temporal order**" cho PaySim. | Cắt **theo thời gian** (train/val/test = 64/16/20) | Random shuffle và giữ thứ tự thời gian không thể đồng thời đúng. Cắt theo thời gian tránh rò rỉ tương lai. |
| C4 | Sec. III-E: FedAvg là "weighted average". Algorithm 1 dòng 7: trung bình **đều** 1/\|C_t\|. | Trung bình đều (`fedavg_uniform`) | Algorithm 1 là thuật toán chính thức của paper. |
| C5 | Sec. IV-B: "DQN" nhưng không gian action liên tục; "discretize ... and use NAF". | Cài **cả hai**: NAF (mặc định) và DQN rời rạc 19 bin | Paper nhắc cả hai. |
| C6 | Table 1: Credit Card 2023 có ">550,000 giao dịch, 31 feature, có transaction time, imbalanced". Sec. V-B-3: fraud rate 0.172%. | Xem mục 4.2 | Mô tả này khớp bộ ULB 2013, không khớp bản Kaggle 2023. |
| C7 | Abstract: "97.3% F1-score" (một dataset). Table 2: F1 = 0.923 / 0.928 / 0.925; **97.3 là Recall@1% của PaySim**. | Không dùng làm mốc đối chiếu | Số liệu nội bộ không nhất quán. |
| C8 | Table 2 PaySim: AUC-PR = 0.647 nhưng F1 = 0.923. Hai số này gần như không thể cùng đúng: F1 ≈ 0.92 ở một điểm vận hành kéo theo AP cao hơn nhiều 0.647. | Không kỳ vọng khớp đồng thời | Đây là mâu thuẫn số học trong Table 2. |
| C9 | Text nói FraudGNN-RL "outperforms across all metrics", nhưng F1 của AutoGCN (0.925 / 0.932) cao hơn FraudGNN-RL (0.923 / 0.928) trên PaySim và CC2023. | Chỉ ghi nhận | Không phải lỗi của reproduction. |
| C10 | Abstract và Sec. VI nêu "resilience to concept drift and adversarial attacks", nhưng Sec. V không có thí nghiệm nào. Sec. VI-B liệt kê "adversarial robustness" và "online learning" là future work. | Xem mục 5 | Không có giao thức để tái hiện. |
| C11 | Sec. VI-B(4) liệt kê federated learning là future work, trong khi FL là thành phần cốt lõi (Sec. IV-C). | Cài FL theo Algorithm 1 | Có thể là lỗi biên tập. |
| C12 | Preliminaries nói "FedAvg ... weighted"; Algorithm 1 không có trọng số. | Như C4 | |
| C13 | Implementation nói "batch size 64" và "5-fold CV" nhưng không nêu cách batch trên graph 6M node, và Table 2 không có độ lệch chuẩn. | Xem D3, E1 | |

---

## 2. Phần paper không công bố [INFERRED] / [UNREPRODUCIBLE]

### 2.1 Graph construction

| ID | Nội dung | Lựa chọn | Nhãn |
|---|---|---|---|
| G1 | Edge có hướng hay không | **Vô hướng** (thêm cạnh ngược, loại trùng) | [INFERRED] |
| G2 | Self-loop | **Không** (`add_self_loops=False` ở cả đồ thị lẫn GATConv, nên không tính 2 lần) | [INFERRED] |
| G3 | Giới hạn số láng giềng | Paper không giới hạn. Repo dùng `max_neighbors_per_node` (top-k theo cosine) | [DEVIATION] |
| G4 | Tìm kiếm cạnh | Chính xác (sort theo thời gian + nhân ma trận trong cửa sổ \[t-1h, t+1h\]). Chỉ khác paper khi `max_neighbors_per_node` ≠ null | [INFERRED] tối ưu tính toán, không đổi ngữ nghĩa |
| G5 | Đồ thị dựng riêng cho train, val, test; **không có cạnh xuyên split** | Cạnh chỉ nối giao dịch trong cùng split | [INFERRED] |
| G6 | Đồ thị tĩnh dựng một lần; Fig. 2 mô tả đồ thị động cập nhật theo giao dịch mới | Dựng 1 lần/split, không streaming, cạnh không nhân quả (có thể trỏ tới giao dịch muộn hơn) | [DEVIATION] |
| G7 | Edge feature f_k | Không có f_k thật. Dùng embedding của node láng giềng làm "message" cho GRU | [INFERRED] |

### 2.2 TSSGC

| ID | Nội dung | Lựa chọn | Nhãn |
|---|---|---|---|
| T1 | Dấu của β (Eq. 7) | β = softplus(β_raw) > 0, learnable | [INFERRED] |
| T2 | `t_now - t_k` trong Eq. 7 | Dùng \|t_i − t_j\| của cạnh (đồ thị vô hướng, không có "now") | [INFERRED] |
| T3 | Cách GRU nhận chuỗi (α_k, f_k) | Phương án tuần tự: sắp cạnh cũ → mới, đưa α_k·lin(x_j) qua `nn.GRU`, lấy hidden cuối | [INFERRED] |
| T4 | Độ dài chuỗi GRU | Cap `temporal_max_seq_len` (giữ các cạnh mới nhất) | [DEVIATION] |
| T5 | W_s xuất hiện trong GATConv và lại ở Eq. 11 | Hiểu W_s (Eq. 8) bên trong GATConv; W_t, W_s, W_m ở Eq. 11 là 3 ma trận riêng cộng lại. Paper dùng chung ký hiệu cho hai thứ khác nhau | [INFERRED] |
| T6 | Số head của GAT | 1 (Eq. 9 chỉ có một α_ij) | [INFERRED] |
| T7 | `GATConv` bias và attention dropout | **Đã tắt** (`bias=False`, `dropout=0.0`) vì Eq. 8-9 không có hai thứ này. Dropout chỉ còn ở cuối mỗi lớp TSSGC (T10) | Khớp Eq. 8-9 |
| T8 | e_type(i) khi node = transaction | PaySim: cột `type` (5 loại, paper nêu). IEEE-CIS: `ProductCD` (W,C,H,R,S). CC2023: single type | PaySim [PAPER]; còn lại [INFERRED] |
| T9 | Thứ tự BatchNorm, ReLU | Paper: "batch normalization after each convolution layer". Repo: tổng 3 nhánh + b → BN → ReLU | BN [PAPER]; thứ tự [INFERRED] |
| T10 | Dropout (0.2) | Paper không nhắc | [DEVIATION] |
| T11 | Phép chiếu đầu vào dùng chung (`shared_projection`) | Linear(in → 64) trước TSSGC | [INFERRED] xem F1 |
| T12 | Classifier head | Linear → ReLU → Dropout → Linear(1) | [INFERRED] |

### 2.3 Reinforcement learning

| ID | Nội dung | Lựa chọn | Nhãn |
|---|---|---|---|
| R1 | Reward | `F1 + 0.25·Recall − 1.0·FPR` theo batch. Paper chỉ nói "combination of detection accuracy and false positive rate" | [INFERRED] tự thiết kế |
| R2 | State | Mean-pooling embedding TSSGC của batch giao dịch hiện tại | [INFERRED] |
| R3 | Action | (threshold ∈ \[0,1\], feature weights) | threshold [PAPER] |
| R4 | Feature weights tác dụng lên đâu | Lên **64 chiều embedding** (trước classifier), không phải đặc trưng đầu vào (có thể hàng trăm đến nghìn chiều) | [INFERRED] |
| R5 | Chuẩn hóa feature weights | w = D·a/Σa (trung bình = 1) | [INFERRED] |
| R6 | Chuyển trạng thái | State kế tiếp = batch kế tiếp, không phụ thuộc action (thực chất là contextual bandit; γ gần như vô tác dụng) | [INFERRED] |
| R7 | Cập nhật target network | Polyak τ = 0.005. Paper chỉ nói "periodically updated" | [INFERRED] |
| R8 | Replay buffer, γ, noise thăm dò | 10 000; γ = 0.99; noise Gaussian giảm dần | [INFERRED] |
| R9 | RL huấn luyện trên tập nào | **Validation**; test không dùng để cập nhật policy | [INFERRED] |
| R10 | Threshold cuối | Trung bình threshold greedy trên val (`inference: mean`), tùy chọn `per_batch` | [INFERRED] |
| R11 | Online adaptation trên test (Fig. 2 có vòng feedback) | **Không cài**. Paper cũng liệt kê "online learning" là future work | [UNREPRODUCIBLE] / [DEVIATION] |
| R12 | NAF: ma trận L | L chéo = exp(·) kẹp \[−4, 2\] để ổn định số | [INFERRED] |

### 2.4 Federated learning

| ID | Nội dung | Lựa chọn | Nhãn |
|---|---|---|---|
| F1 | **Graph alignment** (Sec. IV-C) | Paper không đưa bất kỳ chi tiết nào. Repo giả thuyết rằng nó tương đương schema đặc trưng chung (cùng một preprocessor) + lớp projection chung được FedAvg. **Đây là giả thuyết của người thực hiện, không trích dẫn từ paper.** | [UNREPRODUCIBLE] |
| F2 | Số client | 3, chia theo khoảng thời gian liên tục | [INFERRED] |
| F3 | Chọn client mỗi round | Tất cả (`client_fraction: 1.0`) | [INFERRED] |
| F4 | LocalUpdate ("several steps of SGD") | `local_epochs` epoch Adam | [INFERRED] |
| F5 | Số round | 50 round × 2 local epoch (≈100 epoch tương đương); early stop theo val AUC-PR | [INFERRED] |
| F6 | Cạnh xuyên client | Bị loại (mỗi client chỉ thấy đồ thị cục bộ) | [INFERRED] |
| F7 | Preprocessor (min/max, mode, vocabulary one-hot) | Fit **tập trung** trên toàn bộ train rồi chia cho client. Điều này không mô phỏng đúng FL thực (client không chia sẻ thống kê) | [DEVIATION] |
| F8 | Optimizer state cục bộ | Giữ qua các round; đồng bộ cả buffer BatchNorm | [INFERRED] |

### 2.5 Huấn luyện và tiền xử lý

| ID | Nội dung | Lựa chọn | Nhãn |
|---|---|---|---|
| D1 | Loss | BCE có `pos_weight = n_neg/n_pos` (cắt ở 100). Paper chỉ nói "techniques to handle class imbalance" | [INFERRED] |
| D2 | Gradient clipping 5.0 | Paper không nhắc | [DEVIATION] |
| D3 | "Batch size 64" | Không khả thi cho message passing trên đồ thị giao dịch lớn. Repo dùng `batch_mode: full` (CC2023) hoặc `subgraph` (chunk nút liên tục theo thời gian; cạnh xuyên chunk bị bỏ). `batch_size: 64` chỉ áp dụng cho update của RL | [DEVIATION] |
| D4 | Suy luận theo chunk (`eval_chunk_nodes`) | Dùng khi hết bộ nhớ; bỏ cạnh xuyên chunk | [DEVIATION] |
| D5 | Early stopping | Theo val AUC-PR (paper không nêu tiêu chí) | [INFERRED] |
| D6 | Temporal features | hour, day-of-week, Δt giữa giao dịch liền kề tính trong từng split sau khi sort | [INFERRED] |
| D7 | Tỉ lệ split | Train/val/test = 0.64/0.16/0.20 (80% ngoài, val lấy từ 80% đó) | [INFERRED] |
| D8 | IEEE-CIS | Chỉ có `train_transaction.csv` có nhãn công khai nên dùng cắt thời gian 64/16/20, không dùng "original split" | [INFERRED] |
| D9 | One-hot IEEE-CIS | `onehot_min_frequency: null` (đúng paper) nên có thể thành hàng nghìn cột; nếu hết RAM phải đặt giá trị và ghi vào kết quả | [DEVIATION] nếu bật |
| D10 | 5-fold CV | Có chế độ `--cv`. Dữ liệu có thời gian dùng `TimeSeriesSplit` (expanding window); không có thời gian dùng `StratifiedKFold`. Paper không nói cách chia fold | [INFERRED] |
| D11 | Class-imbalance experiment (Fig. 4) | Ép tỉ lệ fraud về 0.1/0.5/1/2% bằng random sampling giữ thứ tự thời gian | [INFERRED] |

---

## 3. Tham số có mặt trong paper và trạng thái

| Tham số | Paper | Repo | Trạng thái |
|---|---|---|---|
| Số lớp TSSGC | 3 | 3 | [PAPER] |
| Hidden | 64 | 64 | [PAPER] |
| Activation | ReLU | ReLU | [PAPER] |
| BatchNorm sau mỗi lớp | Có | Có | [PAPER] |
| Q-network | 2 × 128 | 2 × 128 | [PAPER] |
| Optimizer / LR | Adam / 0.001 | Adam / 0.001 | [PAPER] |
| Batch size | 64 | xem D3 | [DEVIATION] |
| Epoch / patience | 100 / 10 | 100 / 10 | [PAPER] |
| Seed | 42 (Python, NumPy, PyTorch) | 42 | [PAPER] |
| Ngưỡng cosine | ≥ 0.9 | 0.9 | [PAPER] |
| Cửa sổ thời gian | 1 giờ | 1.0 | [PAPER] |
| PyTorch / PyG | 1.8.0 / 2.0.1 | Phiên bản mới hơn (xem mục 6) | [DEVIATION] |
| Phần cứng | V100 + 32 GB RAM | Kaggle GPU (khác) | [DEVIATION] |

---

## 4. Dữ liệu

### 4.1 PaySim
- Paper: 6,362,620 giao dịch, 8,213 fraud (0.129%), `step` = 1 giờ. Repo dùng `step` trực tiếp làm giờ.
- Nếu dùng `paysim_fast.csv` hoặc `sample_frac < 1`, kết quả **không phải** kết quả paper. Số dòng và số fraud được in ở log `[LOAD]` và phải được báo cáo.

### 4.2 Credit Card 2023
- Paper mô tả có thời gian giao dịch và tỉ lệ fraud 0.172%. Bản Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023` theo hiểu biết của người viết (**cần xác minh bằng log `[LOAD] labels=`**) không có cột thời gian và gần cân bằng 50/50.
- Nếu đúng như vậy: (a) không áp được cửa sổ 1 giờ, graph chỉ dùng cosine; (b) Recall@1% có trần rất thấp về mặt toán học, nên **không thể** khớp số 97.8% của paper; (c) AUC-PR của dự đoán ngẫu nhiên đã là ~0.5.
- Kết quả CC2023 vì thế **không so sánh trực tiếp được với Table 2**.

### 4.3 IEEE-CIS
- Hợp nhất `train_transaction` + `train_identity` bằng `TransactionID` (LEFT JOIN, giữ NaN rồi impute). Chỉ dùng tập train công khai (có nhãn).
- `TransactionDT` là giây từ một mốc tham chiếu; quy đổi sang giờ trước khi áp cửa sổ 1 giờ.

---

## 5. Phần KHÔNG thuộc paper (tách riêng khỏi phần "reproduction")

| Thành phần | Tình trạng |
|---|---|
| 8 baseline (`src/models/baselines.py`: XGBoost, IF, LOF, DeepAE, GCN, GraphSemi, AutoGCN, FedFraud) | Paper chỉ nêu tên và mô tả một dòng, không có siêu tham số. **Toàn bộ siêu tham số là [INFERRED]**. GraphSemi dùng chung đồ thị cosine thay vì đồ thị Pearson và không có label propagation (đơn giản hóa). FedFraud dùng CNN-1D (biến thể tốt nhất theo paper). |
| Concept drift test | Paper tuyên bố nhưng không có giao thức. Repo chia test thành N cửa sổ thời gian bằng nhau và so sánh metric. [INFERRED] |
| Adversarial test | Paper tuyên bố nhưng không có giao thức. Repo dùng FGSM, nhiễu ngẫu nhiên, đảo cạnh. [INFERRED] |
| Soft edges, pruning, hybrid graph | **Không thuộc paper.** Đã xóa khỏi nhánh reproduction (`src/extensions/`, cờ `soft_edges`/`pruning`/`hard_edges`, khối `soft_graph:`/`pruning:` trong config). Nếu cần giữ để làm nghiên cứu nối tiếp, đặt ở nhánh git riêng. |

---

## 6. Môi trường và tái lập

- Paper dùng PyTorch 1.8.0 / PyG 2.0.1. Repo cần phiên bản mới hơn (ví dụ `scatter_reduce_` cần torch ≥ 1.12; `OneHotEncoder(sparse_output=...)` cần scikit-learn ≥ 1.2). Phiên bản thực tế phải được ghi trong `requirements_freeze.txt` và trong file kết quả.
- Seed 42 được đặt cho Python, NumPy, PyTorch. Một số phép scatter của PyG trên CUDA **không hoàn toàn deterministic** nên kết quả có thể dao động nhẹ giữa các lần chạy.
- Chưa có độ lệch chuẩn qua 5-fold cho mọi dataset; chỉ báo cáo số có CV cùng ± std.

---

## 7. Trạng thái clean-room

- Nhánh reproduction **chỉ** chứa các cờ thuộc paper: `flags.gnn`, `flags.rl`, `flags.federated` (dùng cho ablation của paper).
- Không có soft edges, pruning, hybrid graph, weighted fusion, "augmented graph".
- `python tools/check_clean_room.py` kiểm tra tự động và phải trả về mã 0 trước khi chạy thực nghiệm để báo cáo.
- Các config `*_sample.yaml`, `test.yaml`, `smoke_test.yaml` là để chạy thử nhanh (`sample_frac < 1`, ít epoch) và **không phải kết quả paper**; checker bỏ qua chúng.
- Mọi file kết quả có trường `deviations_from_paper` liệt kê các cap kỹ thuật đang bật (G3, T4, T7, T10, D3, D4, D9 ...).

---

## 8. Cách đọc kết quả một cách trung thực

1. Không so sánh trực tiếp từng chữ số với Table 2 (xem C6, C7, C8).
2. Chạy hai cấu hình: **strict** (`dropout: 0`, `shared_projection: false`, không cap) và **practical** (có cap để vừa bộ nhớ). Báo cáo cả hai.
3. Báo cáo kèm: số dòng dữ liệu thật, tỉ lệ fraud, số cạnh trung bình/node, phiên bản thư viện, và danh sách `deviations_from_paper`.
4. "Khớp" ở đây nghĩa là **khớp công thức và quy trình được paper mô tả**, không có nghĩa là tái hiện được con số.
