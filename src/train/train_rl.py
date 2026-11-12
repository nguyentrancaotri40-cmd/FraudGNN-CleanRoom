# ============================================================
# src/train/train_rl.py — Huấn luyện / áp dụng RL agent (NAF mặc định | DQN rời rạc).
#
# [INFERRED] Paper không nói:
#   - RL train trên tập nào  -> VALIDATION (test giữ độc lập; Fig.2 'feedback' dùng nhãn của val).
#   - Cửa sổ (batch) RL      -> đủ lớn để chứa >= env_min_pos_per_batch fraud (resolve_env_batch_size),
#                               nếu không reward suy biến thành -FPR và policy học "không dự đoán gì" (F1=0).
#   - Episode                -> `steps_per_episode` cửa sổ liên tục ở vị trí ngẫu nhiên (mode="random").
#   - Threshold cuối cùng    -> trung bình threshold của policy greedy khi quét hết val (rl.inference="mean"),
#                               hoặc theo từng cửa sổ ("per_batch").
#   - Feature weights cuối   -> trung bình trọng số greedy trên các cửa sổ val.
#   - Không update online trên test (online learning là FUTURE WORK của paper).
# ============================================================
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.models.dqn_agent import DQNAgent
from src.models.naf_agent import NAFAgent
from src.models.reward import BatchFraudEnv, resolve_env_batch_size, weights_from_action


VALID_INFERENCE = ("mean", "per_batch")


def _check_inference(rl: Dict[str, Any]) -> str:
    inf = str(rl.get("inference", "mean"))
    if inf not in VALID_INFERENCE:
        raise ValueError(f"rl.inference='{inf}' không hợp lệ; chỉ hỗ trợ {VALID_INFERENCE}. "
                         f"(Bản cũ ghi 'grid' nhưng thực tế chạy 'mean' -> đã loại bỏ nhãn sai.)")
    return inf


def effective_env_batch_size(labels, rl: Dict[str, Any]) -> int:
    return resolve_env_batch_size(labels, int(rl.get("env_batch_size", 256)), int(rl.get("env_min_pos_per_batch", 20)))


def _env(emb: torch.Tensor, labels, classifier, rl: Dict[str, Any], mode: str, batch_size: int) -> BatchFraudEnv:
    return BatchFraudEnv(emb, labels, classifier, batch_size=batch_size,
                         fpr_penalty=float(rl.get("fpr_penalty", 1.0)), recall_weight=float(rl.get("recall_weight", 0.25)),
                         mode=mode, steps_per_episode=int(rl.get("steps_per_episode", 32)))


def make_agent(state_dim: int, n_features: int, rl: Dict[str, Any], device: str):
    common = dict(hidden_dim=int(rl.get("hidden_dim", 128)), gamma=float(rl.get("gamma", 0.99)),
                  lr=float(rl.get("learning_rate", 1e-3)), tau=float(rl.get("tau", 0.005)),
                  buffer_size=int(rl.get("buffer_size", 10000)), batch_size=int(rl.get("update_batch_size", 64)),
                  device=device)
    if str(rl.get("type", "naf")).lower() == "dqn":
        return DQNAgent(state_dim, n_features, n_bins=int(rl.get("n_threshold_bins", 19)), **common)
    return NAFAgent(state_dim, n_features, **common)


def train_rl_on_validation(val_emb: torch.Tensor, val_labels: np.ndarray, classifier, cfg: Dict[str, Any],
                           device: str) -> Tuple[Any, Dict[str, Any]]:
    rl = cfg.get("rl", {})
    _check_inference(rl)
    bs = effective_env_batch_size(val_labels, rl)
    env = _env(val_emb, val_labels, classifier, rl, "random", bs)
    agent = make_agent(env.state_dim, val_emb.size(1), rl, device)
    epochs = int(rl.get("epochs", 100))
    losses, ep_rewards = [], []
    n_pos = int((np.asarray(val_labels) == 1).sum())
    print(f"[RL] {agent.kind.upper()} | val N={env.n:,} (fraud={n_pos}) | env_batch requested={rl.get('env_batch_size')} "
          f"-> effective={bs} | steps/episode={env.steps_per_episode} | epochs={epochs}")
    if n_pos == 0:
        print("[RL] [WARN] val không có fraud -> reward vô nghĩa, kết quả RL không đáng tin.")
    for ep in range(epochs):
        s, done, tot = env.reset(), False, 0.0
        while not done:
            a, thr, fw = agent.act(s, explore=True)
            s2, r, done, _ = env.step(thr, weights_from_action(fw))
            agent.remember(s, a, r, s2, float(done))
            l = agent.update()
            if l is not None:
                losses.append(l)
            s, tot = s2, tot + r
        ep_rewards.append(tot / env.steps_per_episode)       # reward trung bình mỗi step (dễ đọc hơn tổng)
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[RL] epoch {ep+1:3d}/{epochs} mean_step_reward={ep_rewards[-1]:.3f} "
                  f"loss={np.mean(losses[-100:]) if losses else float('nan'):.5f}")
    wp = np.asarray(env.window_pos_history, dtype=float)
    stats = {"env_batch_size_effective": int(bs), "mean_fraud_per_window": float(wp.mean()) if wp.size else 0.0,
             "frac_windows_without_fraud": float((wp == 0).mean()) if wp.size else 1.0}
    print(f"[RL] fraud/cửa sổ trung bình={stats['mean_fraud_per_window']:.1f} | "
          f"tỉ lệ cửa sổ không có fraud={stats['frac_windows_without_fraud']:.2%}")
    return agent, {"losses": losses, "episode_rewards": ep_rewards, **stats}


@torch.no_grad()
def apply_policy(agent, emb: torch.Tensor, labels, classifier, cfg: Dict[str, Any],
                 batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Quét greedy policy trên (emb, labels) -> thresholds theo cửa sổ, trọng số trung bình.

    batch_size: dùng ĐÚNG kích thước cửa sổ đã dùng khi huấn luyện (truyền vào khi áp lên test)."""
    rl = cfg.get("rl", {})
    bs = int(batch_size) if batch_size else effective_env_batch_size(labels, rl)
    env = _env(emb, labels, classifier, rl, "sweep", bs)
    s, done = env.reset(), False
    thr_list, ws, spans = [], [], []
    while not done:
        _, thr, fw = agent.act(s, explore=False)
        w = weights_from_action(fw)
        s, _, done, info = env.step(thr, w)
        thr_list.append(thr); ws.append(w); spans.append((info["start"], info["end"]))
    w_mean = np.mean(ws, axis=0).astype(np.float32)
    return {"threshold": float(np.mean(thr_list)), "thresholds": thr_list, "spans": spans,
            "feature_weights": w_mean, "num_steps": len(thr_list)}


__all__ = ["make_agent", "train_rl_on_validation", "apply_policy", "effective_env_batch_size"]
