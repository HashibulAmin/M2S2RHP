"""Leakage-aware offline evaluation for the synthetic M²S²-Rec experiment."""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from data.dataset import Config, MockWorldData
from model.m2s2rec import build_ctx, encode_all_reels, score_candidates, user_representation


def _rank_metrics(scores: np.ndarray, targets: np.ndarray, seen: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masked = scores.copy()
    masked[seen] = -np.inf
    rows = np.arange(scores.shape[0])
    # Always restore the evaluation target even if a malformed synthetic split marked it seen.
    masked[rows, targets] = scores[rows, targets]
    order = np.argpartition(-masked, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    order_scores = np.take_along_axis(masked, order, axis=1)
    sort_idx = np.argsort(-order_scores, axis=1)
    topk = np.take_along_axis(order, sort_idx, axis=1)
    hits = np.any(topk == targets[:, None], axis=1).astype(np.float32)
    rank0 = np.argmax(topk == targets[:, None], axis=1)
    ndcg = np.where(hits > 0, 1.0 / np.log2(rank0 + 2.0), 0.0).astype(np.float32)
    return hits, ndcg, masked


def evaluate(params, world: MockWorldData, cfg: Config) -> dict:
    ctx = build_ctx(world, world.g_all, cfg)
    _, situation_probs = encode_all_reels(params, cfg, ctx)
    uids = jnp.arange(cfg.n_users, dtype=jnp.int32)
    rep = user_representation(params, cfg, ctx, uids, situation_probs)

    n_users, n_reels = cfg.n_users, cfg.n_reels
    all_uids = jnp.repeat(uids, n_reels)
    all_rids = jnp.tile(jnp.arange(n_reels, dtype=jnp.int32), n_users)
    expanded = {k: jnp.repeat(v, n_reels, axis=0) for k, v in rep.items()}
    scores_flat, _ = score_candidates(params, cfg, ctx, all_uids, all_rids, expanded, situation_probs)
    scores = np.asarray(scores_flat).reshape(n_users, n_reels)

    targets = world.test_reel.copy().astype(np.int32)
    valid = targets >= 0
    targets = np.clip(targets, 0, n_reels - 1)
    seen = np.zeros((n_users, n_reels), dtype=bool)
    for uid in range(n_users):
        seen[uid, world.hist_reel[uid, world.hist_mask[uid]]] = True
    hits, ndcg, masked = _rank_metrics(scores[valid], targets[valid], seen[valid], cfg.eval_k)

    aucs = []
    city_hits = []
    group_hits = defaultdict(list)
    for row, uid in enumerate(np.flatnonzero(valid)):
        t = int(targets[uid])
        negatives = np.delete(masked[row], np.where(np.arange(n_reels) == t)[0][0])
        aucs.append(float(np.mean(masked[row, t] > negatives)))
        city_hits.append(float(world.reel_city[t] == world.user_city[uid]))
        group_hits[int(world.user_gender[uid])].append(float(hits[row]))

    probs_np = np.asarray(situation_probs)
    labels = world.reel_weak_situation
    classifier_bce = float(-np.mean(labels * np.log(np.clip(probs_np, 1e-7, 1)) + (1 - labels) * np.log(np.clip(1 - probs_np, 1e-7, 1))))

    cold = set(world.cold_users.tolist())
    cold_rows = np.asarray([i for i, uid in enumerate(np.flatnonzero(valid)) if int(uid) in cold], dtype=np.int32)
    warm_rows = np.asarray([i for i, uid in enumerate(np.flatnonzero(valid)) if int(uid) not in cold], dtype=np.int32)

    gate_np = np.asarray(rep["gate"])
    return {
        "test_recall_at_k": float(np.mean(hits)),
        "test_ndcg_at_k": float(np.mean(ndcg)),
        "pairwise_auc": float(np.mean(aucs)) if aucs else 0.0,
        "city_hit_rate": float(np.mean(city_hits)) if city_hits else 0.0,
        "cold_recall_at_k": float(np.mean(hits[cold_rows])) if cold_rows.size else 0.0,
        "warm_recall_at_k": float(np.mean(hits[warm_rows])) if warm_rows.size else 0.0,
        "situation_classifier_bce": classifier_bce,
        "mean_gate": np.mean(gate_np[valid], axis=0).round(6).tolist(),
        "gender_recall_at_k": {str(g): float(np.mean(v)) for g, v in sorted(group_hits.items())},
        "gender_recall_gap": float(max((np.mean(v) for v in group_hits.values()), default=0.0) - min((np.mean(v) for v in group_hits.values()), default=0.0)),
    }
