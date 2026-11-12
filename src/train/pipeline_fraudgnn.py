# ============================================================
# src/train/pipeline_fraudgnn.py — Pipeline đầy đủ (Fig. 2, Section IV-D):
#   New Transactions -> Update Graph -> TSSGC Processing -> RL Action Selection
#   -> Classification -> Feedback -> Update RL Policy
#
# Trong benchmark offline:
#   - Update Graph        : dựng graph 1 lần cho mỗi split (train / val / test)
#   - TSSGC Processing    : huấn luyện encoder+classifier (FL hoặc centralized)
#   - RL Action Selection : NAF/DQN huấn luyện trên VAL (state = embedding, action = threshold + feature weights)
#   - Classification      : áp (threshold, feature weights) lên TEST — KHÔNG feedback online (future work của paper)
#
# Flags (ablation của paper): gnn / rl / federated.
# ============================================================
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.load_data import load_dataset
from src.data.preprocess import FraudPreprocessor
from src.data.split import get_cv_splits, resample_fraud_ratio, split_dataframe
from src.eval.evaluate import predict_embeddings, predict_scores, save_metrics
from src.eval.metrics import best_f1_threshold, classification_metrics
from src.graph.build_graph import build_transaction_graph, extract_entity_type
from src.train.federated import train_federated
from src.train.train_gnn import train_tssgc_classifier
from src.train.train_rl import apply_policy, train_rl_on_validation
from src.utils.config import get_flags
from src.utils.model_definitions import resolve_model_name
from src.utils.seed import set_seed


def get_device(cfg: Dict[str, Any]) -> str:
    d = cfg.get("runtime", {}).get("device", "auto")
    return ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d


def prepare_splits(train_df, val_df, test_df, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess (fit trên train) + dựng 3 graph. Dùng chung cho FraudGNN-RL và baseline."""
    ds = cfg.get("dataset", {})
    vals = ds.get("entity_type_values")
    cfg.setdefault("model", {})["num_node_types"] = len(vals) if vals else 1

    timing = {}
    t0 = time.perf_counter()
    pre = FraudPreprocessor(cfg)
    x_tr, y_tr, t_tr = pre.fit_transform(train_df)
    x_va, y_va, t_va = pre.transform(val_df)
    x_te, y_te, t_te = pre.transform(test_df)
    timing["preprocessing_sec"] = time.perf_counter() - t0
    print(f"[PREP] features={x_tr.shape[1]} (train fit only) in {timing['preprocessing_sec']:.1f}s")

    t0 = time.perf_counter()
    graphs = [build_transaction_graph(x, y, t, cfg, extract_entity_type(df, cfg))
              for x, y, t, df in ((x_tr, y_tr, t_tr, train_df), (x_va, y_va, t_va, val_df), (x_te, y_te, t_te, test_df))]
    timing["graph_building_sec"] = time.perf_counter() - t0
    return {"pre": pre, "x": (x_tr, x_va, x_te), "y": (y_tr, y_va, y_te), "t": (t_tr, t_va, t_te),
            "graphs": tuple(graphs), "timing": timing}


def run_pipeline_on_split(train_df, val_df, test_df, cfg: Dict[str, Any], prepared: Optional[dict] = None,
                          seed: Optional[int] = None, return_artifacts: bool = False):
    seed = int(seed if seed is not None else cfg.get("seed", cfg.get("dataset", {}).get("random_state", 42)))
    set_seed(seed)
    flags, device = get_flags(cfg), get_device(cfg)
    print("=" * 60 + f"\n[PIPE] {resolve_model_name(flags)} | flags={flags} | device={device} | seed={seed}\n" + "=" * 60)
    t_all = time.perf_counter()
    prep = prepared or prepare_splits(train_df, val_df, test_df, cfg)
    g_tr, g_va, g_te = prep["graphs"]
    timing = {"preprocessing_sec": 0.0, "graph_building_sec": 0.0, "gnn_training_sec": 0.0,
              "rl_training_sec": 0.0, "inference_sec": 0.0, **prep["timing"]}
    eval_chunk = cfg.get("train", {}).get("eval_chunk_nodes")

    # ---- 1) TSSGC + classifier (FL | centralized) ----------------------
    if flags["federated"]:
        fed = train_federated(g_tr, g_va, cfg, device, timing)
        model, hist = fed["model"], fed["history"]
        fl_info = {k: fed[k] for k in ("num_clients", "num_rounds", "avg_round_time_sec")}
    else:
        out = train_tssgc_classifier(g_tr, g_va, cfg, device, timing)
        model, hist = out["model"], out["history"]
        fl_info = {"centralized": True}
    model.reset_feature_weights()

    # ---- 2) RL ---------------------------------------------------------
    rl_cfg, rl_info, y_pred_te, env_bs = cfg.get("rl", {}), {"enabled": False}, None, None
    threshold, thr_method = 0.5, "fixed_0.5"
    if flags["rl"]:
        val_emb = predict_embeddings(model, g_va, device, eval_chunk)
        y_va = g_va.y.numpy()
        t0 = time.perf_counter()
        agent, rl_hist = train_rl_on_validation(val_emb, y_va, model.classifier, cfg, device)
        timing["rl_training_sec"] = time.perf_counter() - t0
        env_bs = rl_hist["env_batch_size_effective"]
        pol = apply_policy(agent, val_emb, y_va, model.classifier, cfg, batch_size=env_bs)
        model.set_feature_weights(pol["feature_weights"])      # trọng số RL thực sự đi vào điểm số
        threshold = pol["threshold"]
        thr_method = f"{agent.kind}_policy_{rl_cfg.get('inference', 'mean')}"
        rl_info = {"enabled": True, "agent": agent.kind, "num_updates": len(rl_hist["losses"]),
                   "final_loss": float(np.mean(rl_hist["losses"][-100:])) if rl_hist["losses"] else None,
                   "last_episode_mean_step_reward": rl_hist["episode_rewards"][-1],
                   "first_episode_mean_step_reward": rl_hist["episode_rewards"][0],
                   "policy_threshold": threshold, "threshold_method": thr_method,
                   "env_batch_size_requested": rl_cfg.get("env_batch_size"),
                   "env_batch_size_effective": env_bs,
                   "mean_fraud_per_window": rl_hist["mean_fraud_per_window"],
                   "frac_windows_without_fraud": rl_hist["frac_windows_without_fraud"],
                   "feature_weight_std": float(np.std(pol["feature_weights"]))}
    elif cfg.get("evaluation", {}).get("threshold_when_rl_off", "fixed_0.5") == "val_best_f1":
        vs, vl = predict_scores(model, g_va, device, None, eval_chunk)
        threshold, thr_method = best_f1_threshold(vl, vs), "val_best_f1 (w/o RL)"
        rl_info = {"enabled": False, "threshold_method": thr_method}
    else:
        rl_info = {"enabled": False, "threshold_method": thr_method}
    print(f"[PIPE] threshold={threshold:.4f} ({thr_method})")

    # ---- 3) Đánh giá (không feedback tiếp) ---------------------------------
    t0 = time.perf_counter()
    val_scores, val_labels = predict_scores(model, g_va, device, timing, eval_chunk)
    test_scores, test_labels = predict_scores(model, g_te, device, timing, eval_chunk)
    if flags["rl"] and rl_cfg.get("inference", "mean") == "per_batch":
        te_emb = predict_embeddings(model, g_te, device, eval_chunk)
        tp = apply_policy(agent, te_emb, test_labels, model.classifier, cfg, batch_size=env_bs)   # action chỉ phụ thuộc state (không dùng nhãn)
        y_pred_te = np.zeros(len(test_scores), dtype=bool)
        for (a, b), th in zip(tp["spans"], tp["thresholds"]):
            y_pred_te[a:b] = test_scores[a:b] >= th
    val_m = classification_metrics(val_labels, val_scores, threshold)
    test_m = classification_metrics(test_labels, test_scores, threshold, y_pred=y_pred_te)
    timing["eval_sec"] = time.perf_counter() - t0

    # ---- Chẩn đoán (KHÔNG phải kết quả paper): ngưỡng tối đa F1 chọn trên VAL, áp lên test ----
    # Dùng để tách "model xếp hạng kém" (AUC thấp) khỏi "ngưỡng RL đặt sai" (AUC cao nhưng F1 = 0).
    diag_thr = best_f1_threshold(val_labels, val_scores) if int(np.sum(val_labels)) > 0 else 0.5
    diagnostic = {"note": "ngưỡng chọn bằng val F1 tối đa, KHÔNG phải ngưỡng của RL; chỉ để chẩn đoán",
                  "threshold": float(diag_thr),
                  "val_metrics": classification_metrics(val_labels, val_scores, diag_thr),
                  "test_metrics": classification_metrics(test_labels, test_scores, diag_thr)}
    if val_m["num_fraud"] > 0 and val_m["tp"] == 0:
        print(f"[PIPE] [WARN] ngưỡng {threshold:.4f} không bắt được fraud nào trên val "
              f"(val AUC-ROC={val_m['auc_roc']:.3f}). Xem diagnostic_val_best_f1_threshold.")
    timing["total_runtime_sec"] = time.perf_counter() - t_all

    extra = {}
    deviations = []
    g = cfg.get("graph", {})
    if g.get("max_neighbors_per_node") is not None:
        deviations.append(f"graph.max_neighbors_per_node={g['max_neighbors_per_node']} (cap kỹ thuật, paper không giới hạn)")
    if cfg.get("model", {}).get("temporal_max_seq_len") is not None:
        deviations.append(f"model.temporal_max_seq_len={cfg['model']['temporal_max_seq_len']} (cap kỹ thuật cho GRU)")
    if cfg.get("dataset", {}).get("sample_frac") not in (None, 1.0):
        deviations.append(f"dataset.sample_frac={cfg['dataset']['sample_frac']} (KHÔNG phải paper)")
    m_, t_, r_ = cfg.get("model", {}), cfg.get("train", {}), cfg.get("rl", {})
    if float(m_.get("dropout", 0)) > 0:
        deviations.append(f"model.dropout={m_['dropout']} (T10: paper không nhắc)")
    if m_.get("shared_projection"):
        deviations.append("shared_projection=True (F1/T11: GIẢ THUYẾT graph alignment, không có trong paper)")
    if cfg.get("evaluation", {}).get("threshold_when_rl_off") == "val_best_f1" and not flags["rl"]:
        deviations.append("w/o RL: ngưỡng = val best-F1 thay vì 0.5 (E2: paper không nêu)")
    if t_.get("batch_mode") == "subgraph":
        deviations.append("train.batch_mode=subgraph (D3: bỏ cạnh xuyên chunk; paper: batch 64)")
    if t_.get("eval_chunk_nodes"):
        deviations.append(f"train.eval_chunk_nodes={t_['eval_chunk_nodes']} (D4: suy luận theo chunk, bỏ cạnh xuyên chunk)")
    deviations += [
        "BCE pos_weight + grad_clip=5.0 (D1/D2: paper không nêu)",
        "Classifier head Linear-ReLU-Dropout-Linear (T12: paper không nêu)",
        f"RL: Polyak tau={r_.get('tau')} (R7); reward tự thiết kế (R1); cửa sổ RL ngẫu nhiên >= {r_.get('env_min_pos_per_batch', 20)} fraud (R13)",
        "Early stopping theo val AUC-PR (D5: paper không nêu tiêu chí)",
    ]
    import torch_geometric, sklearn
    extra["env"] = {"torch": torch.__version__, "pyg": torch_geometric.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}

    print(f"\n[PIPE] TEST AUC-ROC={test_m['auc_roc']:.4f} AUC-PR={test_m['auc_pr']:.4f} F1={test_m['f1']:.4f} "
          f"Recall@1%={test_m['recall_at_1pct']*100:.1f}% FPR={test_m['fpr']:.5f} | total {timing['total_runtime_sec']:.0f}s")
    result = {
        "model_name": resolve_model_name(flags), "flags": flags, "seed": seed,
        "selected_threshold": float(threshold), "threshold_method": thr_method, "val_metrics": val_m, "test_metrics": test_m,
        "runtime": timing, "federated_metrics": fl_info, "rl_info": rl_info,
        "graph_info": {"train_edges": int(g_tr.edge_index.size(1)), "val_edges": int(g_va.edge_index.size(1)),
                       "test_edges": int(g_te.edge_index.size(1)), "num_features": int(g_tr.x.size(1)),
                       "train_avg_degree": float(g_tr.edge_index.size(1)) / max(1, g_tr.num_nodes),
                       "val_avg_degree": float(g_va.edge_index.size(1)) / max(1, g_va.num_nodes),
                       "test_avg_degree": float(g_te.edge_index.size(1)) / max(1, g_te.num_nodes),
                       "train_fraud_rate": float((g_tr.y == 1).float().mean()),
                       "val_fraud_rate": float((g_va.y == 1).float().mean()),
                       "test_fraud_rate": float((g_te.y == 1).float().mean())},
        "diagnostic_val_best_f1_threshold": diagnostic,
        "history_len": len(hist), "deviations_from_paper": deviations, **extra,
    }
    if return_artifacts:
        return result, {"model": model, "graphs": prep["graphs"], "prepared": prep, "device": device,
                        "val_scores": val_scores, "val_labels": val_labels, "test_scores": test_scores,
                        "test_labels": test_labels, "threshold": threshold, "y_pred_test": y_pred_te}
    return result


def _save(result: dict, cfg: Dict[str, Any], suffix: str) -> None:
    out = Path(cfg.get("output", {}).get("dir", "outputs")) / "results"
    save_metrics(result, str(out / f"{cfg.get('experiment', {}).get('name', 'experiment')}_{suffix}.json"))


def run_pipeline(cfg: Dict[str, Any], return_artifacts: bool = False, save: bool = True):
    df = load_dataset(cfg)
    ds = cfg["dataset"]
    if ds.get("fraud_ratio_pct") is not None:   # thí nghiệm imbalance (Fig. 4)
        df = resample_fraud_ratio(df, float(ds["fraud_ratio_pct"]), ds["label_col"], ds.get("time_col"),
                                  int(ds.get("random_state", 42)))
        print(f"[PIPE] fraud ratio ép về {ds['fraud_ratio_pct']}% -> N={len(df):,}")
    train_df, val_df, test_df = split_dataframe(df, cfg)
    res = run_pipeline_on_split(train_df, val_df, test_df, cfg, return_artifacts=return_artifacts)
    result = res[0] if return_artifacts else res
    if save:
        _save(result, cfg, "metrics")
    return res


def run_cv_pipeline(cfg: Dict[str, Any], n_folds: int = 5) -> Dict[str, Any]:
    """5-fold CV (paper V-A-4)."""
    df = load_dataset(cfg)
    runs = []
    for i, (tr, va, te) in enumerate(get_cv_splits(df, cfg, n_folds), 1):
        print(f"\n[CV] Fold {i}/{n_folds}: train={len(tr):,} val={len(va):,} test={len(te):,}")
        try:
            r = run_pipeline_on_split(tr, va, te, copy.deepcopy(cfg))
            runs.append({"fold": i, **r})
        except Exception as e:  # noqa
            import traceback; traceback.print_exc()
            runs.append({"fold": i, "error": str(e)})
    ok = [r for r in runs if "error" not in r]
    summary = {"n_folds": n_folds, "n_valid": len(ok), "folds": runs}
    for m in ("auc_roc", "auc_pr", "f1", "recall", "precision", "fpr", "recall_at_1pct"):
        v = [r["test_metrics"][m] for r in ok]
        if v:
            summary[f"{m}_mean"], summary[f"{m}_std"] = float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    _save(summary, cfg, "cv_results")
    for m in ("auc_roc", "auc_pr", "f1", "recall_at_1pct"):
        if f"{m}_mean" in summary:
            print(f"[CV] {m:<15}: {summary[m+'_mean']:.4f} +/- {summary[m+'_std']:.4f}")
    return summary


__all__ = ["get_device", "prepare_splits", "run_pipeline_on_split", "run_pipeline", "run_cv_pipeline"]
