# ============================================================
# src/models/reward.py — Reward + môi trường RL (Section IV-B / IV-D).
#
# Paper: State = graph embedding (TSSGC); Action = (fraud threshold, feature importance
# weights); Reward = "combination of detection accuracy and false positive rate".
# Paper KHÔNG cho công thức -> [INFERRED]:
#       reward = F1 + recall_weight * Recall - fpr_penalty * FPR
#   (accuracy thuần bị chi phối bởi lớp normal khi fraud ~0.1% nên dùng F1/Recall làm "detection accuracy")
#
# Môi trường (Fig. 2: Update Graph -> TSSGC -> RL action -> Classification -> Feedback -> Update policy):
#   - mode="random" (HUẤN LUYỆN): mỗi step = 1 cửa sổ liên tục (theo thứ tự thời gian) bắt đầu tại vị trí ngẫu nhiên;
#     1 episode = `steps_per_episode` cửa sổ. [INFERRED]
#   - mode="sweep"  (SUY LUẬN): quét tuần tự các cửa sổ KHÔNG chồng lấn trên toàn tập.
#   - Kích thước cửa sổ phải đủ lớn để chứa fraud (resolve_env_batch_size): với fraud ~0.1% và batch 512 thì
#     ~70% batch KHÔNG có fraud -> reward = -FPR -> "không dự đoán gì" là tối ưu -> F1 = 0 (đã gặp ở smoke test).
#   - state  = mean pooling embedding (trước feature weights) của batch hiện tại.
#   - action = [threshold, raw feature weights] ; w = D * a_fw / sum(a_fw).
#   - score  = sigmoid(classifier(embedding * w)) => feature weights THỰC SỰ ảnh hưởng điểm số/reward.
# ============================================================
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_reward(tp: int, fp: int, tn: int, fn: int, fpr_penalty: float = 1.0,
                   recall_weight: float = 0.25) -> float:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fpr = fp / max(1, fp + tn)
    return float(f1 + recall_weight * recall - fpr_penalty * fpr)


def weights_from_action(fw_raw: np.ndarray) -> np.ndarray:
    """a_fw in [0,1]^D  ->  w, mean(w)=1."""
    fw = np.clip(np.asarray(fw_raw, dtype=np.float64), 1e-6, None)
    return (fw.size * fw / fw.sum()).astype(np.float32)


def resolve_env_batch_size(labels, requested: int = 256, min_pos: int = 20, max_frac: float = 0.5) -> int:
    """Kích thước cửa sổ RL hiệu dụng [INFERRED].

    = max(requested, ceil(min_pos / fraud_rate)), nhưng không quá max_frac * N (để còn >= 2 cửa sổ khác nhau).
    Chỉ dùng tỉ lệ fraud của tập mà RL huấn luyện (val) -> không rò rỉ từ test.
    """
    y = np.asarray(labels)
    n = int(len(y))
    req = max(1, int(requested))
    n_pos = int((y == 1).sum())
    if n == 0 or n_pos == 0 or int(min_pos) <= 0:
        return min(req, max(1, n))
    need = int(math.ceil(int(min_pos) * n / n_pos))
    return int(max(1, min(max(req, need), max(1, int(n * max_frac)))))


class BatchFraudEnv:
    """
    embeddings: [N, D] (tensor, đã ở thiết bị tính toán) ; labels: [N] ;
    classifier: FraudClassifier (eval, đóng băng) dùng để chấm điểm cửa sổ.
    mode: "random" (huấn luyện) | "sweep" (suy luận, quét hết tập).
    """

    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray, classifier: torch.nn.Module,
                 batch_size: int = 256, fpr_penalty: float = 1.0, recall_weight: float = 0.25,
                 mode: str = "sweep", steps_per_episode: int = 32, rng: Optional[np.random.Generator] = None):
        if mode not in ("random", "sweep"):
            raise ValueError(f"mode phải là 'random' hoặc 'sweep', nhận {mode!r}")
        self.emb = embeddings
        self.labels = np.asarray(labels, dtype=np.int64)
        self.clf = classifier.eval()
        self.n, self.state_dim = embeddings.size(0), embeddings.size(1)
        self.bs = max(1, min(int(batch_size), self.n))
        self.fpr_penalty, self.recall_weight = float(fpr_penalty), float(recall_weight)
        self.mode, self.steps_per_episode = mode, max(1, int(steps_per_episode))
        self.rng = rng if rng is not None else np.random.default_rng()
        self._cum = np.concatenate([[0], np.cumsum(self.labels == 1)])   # đếm fraud trong cửa sổ O(1)
        self.n_pos_total = int(self._cum[-1])
        self.pos, self.steps = 0, 0
        self.window_pos_history: list = []

    def _pos_in(self, a: int, b: int) -> int:
        return int(self._cum[b] - self._cum[a])

    def _draw_start(self) -> int:
        """Vị trí bắt đầu ngẫu nhiên; thử lại tối đa 20 lần để cửa sổ có >= 1 fraud (nếu tập có fraud)."""
        hi = max(0, self.n - self.bs)
        a = int(self.rng.integers(0, hi + 1))
        if self.n_pos_total == 0:
            return a
        for _ in range(20):
            if self._pos_in(a, a + self.bs) >= 1:
                break
            a = int(self.rng.integers(0, hi + 1))
        return a

    def _state(self) -> np.ndarray:
        if self.pos >= self.n:
            return np.zeros(self.state_dim, dtype=np.float32)
        end = self.n if (self.mode == "sweep" and self.n - (self.pos + self.bs) < self.bs // 2) else self.pos + self.bs
        return self.emb[self.pos:end].mean(0).float().cpu().numpy()

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.pos = self._draw_start() if self.mode == "random" else 0
        return self._state()

    @torch.no_grad()
    def step(self, threshold: float, weights: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, float, bool, Dict]:
        a, b = self.pos, min(self.n, self.pos + self.bs)
        if self.mode == "sweep" and self.n - b < self.bs // 2:
            b = self.n                      # gộp đuôi quá ngắn vào cửa sổ cuối (tránh cửa sổ vài dòng, trạng thái nhiễu)
        e = self.emb[a:b]
        if weights is not None:
            e = e * torch.as_tensor(weights, dtype=e.dtype, device=e.device)
        s = torch.sigmoid(self.clf(e)).cpu().numpy()
        y = self.labels[a:b]
        pred = s >= threshold
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
        r = compute_reward(tp, fp, tn, fn, self.fpr_penalty, self.recall_weight)
        self.window_pos_history.append(tp + fn)
        self.steps += 1
        if self.mode == "random":
            done = self.steps >= self.steps_per_episode
            self.pos = self._draw_start()
        else:
            self.pos = b
            done = self.pos >= self.n
        info = dict(tp=tp, fp=fp, tn=tn, fn=fn, threshold=float(threshold), start=a, end=b)
        return self._state(), r, done, info


__all__ = ["compute_reward", "weights_from_action", "resolve_env_batch_size", "BatchFraudEnv"]
