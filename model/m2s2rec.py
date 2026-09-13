"""Full M²S²-Rec model wiring.

The trainable relevance path follows the paper's consolidated architecture:
1. demographics + behavioral engagement style -> fairness gate b_i
2. temporal interest/music/situation state from chronological training history
3. private city graph -> K-hop LightGCN geo representation
4. per-user softmax gate mixes situation, city, interest and geo scores
5. a small trend feature is added outside the gate

Exploration is intentionally separated from training and applied only by `rank_scores`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from model.adversary import adversary_forward, fair_gate_forward, init_fair_gate
from model.layers import embedding_lookup, init_embedding, init_mlp, mlp_forward
from model.scoring import (
    apply_explicit_situation_floor,
    city_match_score,
    cosine_sim,
    recency_trending_boost,
    temporal_interest_vector,
    temporal_situation_vector,
)
from model.text_encoder import encode_reels, init_text_encoder


class Ctx(NamedTuple):
    city_dist_km: jnp.ndarray
    g_all: jnp.ndarray
    user_city: jnp.ndarray
    user_age_norm: jnp.ndarray
    user_edu: jnp.ndarray
    user_gender: jnp.ndarray
    user_occ: jnp.ndarray
    user_behavior: jnp.ndarray
    user_s_explicit: jnp.ndarray
    hist_reel: jnp.ndarray
    hist_weight: jnp.ndarray
    hist_day: jnp.ndarray
    hist_mask: jnp.ndarray
    reel_city: jnp.ndarray
    reel_tokens: jnp.ndarray
    reel_token_mask: jnp.ndarray
    reel_visual: jnp.ndarray
    reel_age_days: jnp.ndarray
    reel_pop: jnp.ndarray
    day_now: float


def build_ctx(world, g_all, cfg) -> Ctx:
    return Ctx(
        jnp.asarray(world.city_dist_km, dtype=jnp.float32),
        jnp.asarray(g_all, dtype=jnp.float32),
        jnp.asarray(world.user_city, dtype=jnp.int32),
        jnp.asarray(world.user_age_norm, dtype=jnp.float32),
        jnp.asarray(world.user_edu, dtype=jnp.int32),
        jnp.asarray(world.user_gender, dtype=jnp.int32),
        jnp.asarray(world.user_occ, dtype=jnp.int32),
        jnp.asarray(world.user_behavior, dtype=jnp.float32),
        jnp.asarray(world.user_s_explicit, dtype=jnp.float32),
        jnp.asarray(world.hist_reel, dtype=jnp.int32),
        jnp.asarray(world.hist_weight, dtype=jnp.float32),
        jnp.asarray(world.hist_day, dtype=jnp.float32),
        jnp.asarray(world.hist_mask),
        jnp.asarray(world.reel_city, dtype=jnp.int32),
        jnp.asarray(world.reel_tokens, dtype=jnp.int32),
        jnp.asarray(world.reel_token_mask),
        jnp.asarray(world.reel_visual, dtype=jnp.float32),
        jnp.asarray(world.reel_age_days, dtype=jnp.float32),
        jnp.asarray(world.reel_pop, dtype=jnp.float32),
        float(cfg.trace_days),
    )


def init_params(key, cfg):
    keys = jax.random.split(key, 11)
    d_demo = 1 + cfg.d_cat * 4
    fair_in = d_demo + 3
    n_sit = len(cfg.situations)
    return {
        "text_encoder": init_text_encoder(keys[0], cfg),
        # Optional warm-item residual; content encoder remains the primary item path.
        "reel_id_residual": init_embedding(keys[1], cfg.n_reels, cfg.d_model),
        "reel_music_emb": init_embedding(keys[2], cfg.n_reels, cfg.d_style),
        "edu_emb": init_embedding(keys[3], cfg.n_education, cfg.d_cat),
        "gender_emb": init_embedding(keys[4], cfg.n_gender, cfg.d_cat),
        "occ_emb": init_embedding(keys[5], cfg.n_occupation, cfg.d_cat),
        "city_emb_user": init_embedding(keys[6], cfg.n_cities, cfg.d_cat),
        "fair_gate": init_fair_gate(keys[7], cfg, fair_in),
        "W_g": jax.random.normal(keys[8], (cfg.rappor_bloom_bits, cfg.d_model)) * 0.02,
        "gate_mlp": init_mlp(keys[9], [cfg.d_model + cfg.d_style + cfg.d_fair, cfg.d_hidden, 4]),
        "situation_lambda_raw": jnp.full((n_sit,), -1.0),
    }


def _demographic_vector(params, ctx: Ctx, uid):
    age = ctx.user_age_norm[uid][:, None]
    edu = embedding_lookup(params["edu_emb"], ctx.user_edu[uid])
    gender = embedding_lookup(params["gender_emb"], ctx.user_gender[uid])
    occ = embedding_lookup(params["occ_emb"], ctx.user_occ[uid])
    city = embedding_lookup(params["city_emb_user"], ctx.user_city[uid])
    return jnp.concatenate([age, edu, gender, occ, city], axis=-1)


def encode_all_reels(params, cfg, ctx: Ctx):
    """Encode all reels from content features; no interaction history is required.

    This is the item-cold-start path: a new reel can be represented before it has any
    engagement simply from its caption/OCR/ASR-style token stream and visual features.
    """
    return encode_reels(
        params["text_encoder"], cfg, ctx.reel_tokens, ctx.reel_token_mask, ctx.reel_visual
    )


def user_representation(params, cfg, ctx: Ctx, uid, all_situation_probs, all_content_reprs):
    d_i = _demographic_vector(params, ctx, uid)
    p_i = ctx.user_behavior[uid]
    b_i = fair_gate_forward(params["fair_gate"], jnp.concatenate([d_i, p_i], axis=-1))

    h_reel = ctx.hist_reel[uid]
    h_w = ctx.hist_weight[uid]
    h_day = ctx.hist_day[uid]
    h_mask = ctx.hist_mask[uid]
    hist_v = all_content_reprs[h_reel] + params["reel_id_residual"][h_reel]
    hist_music = embedding_lookup(params["reel_music_emb"], h_reel)
    hist_situation = jnp.take(all_situation_probs, h_reel, axis=0)

    day_now = jnp.full((uid.shape[0],), ctx.day_now)
    e_interest = temporal_interest_vector(hist_v, h_w, day_now, h_day, h_mask, cfg.lambda_interest)
    e_music = temporal_interest_vector(hist_music, h_w, day_now, h_day, h_mask, cfg.lambda_interest)

    lam_q = jax.nn.softplus(params["situation_lambda_raw"]) + 1e-4
    s_i = temporal_situation_vector(hist_situation, h_w, day_now, h_day, h_mask, lam_q)
    s_i = apply_explicit_situation_floor(s_i, ctx.user_s_explicit[uid], cfg.kappa_explicit)

    geo_proj = ctx.g_all[uid] @ params["W_g"]
    x_i = jnp.concatenate([e_interest, e_music, b_i], axis=-1)
    gate = jax.nn.softmax(mlp_forward(params["gate_mlp"], x_i), axis=-1)
    return {"b_i": b_i, "e_interest": e_interest, "e_music": e_music, "s_i": s_i, "geo_proj": geo_proj, "gate": gate}


def score_candidates(
    params, cfg, ctx, uid, rid, user_repr, all_situation_probs, content_reprs, gate_override=None
):
    content_v = content_reprs[rid]
    # Warm-item residual improves personalization when the reel has history, while
    # content_v alone remains a valid representation for a brand-new reel.
    v_r = content_v + params["reel_id_residual"][rid]
    situation_r = jnp.take(all_situation_probs, rid, axis=0)
    s_situation = cosine_sim(user_repr["s_i"], situation_r)
    s_city = city_match_score(ctx.user_city[uid], ctx.reel_city[rid], ctx.city_dist_km, cfg.city_eta, cfg.city_tau_km)
    s_interest = cosine_sim(user_repr["e_interest"], v_r)
    s_geo = jax.nn.sigmoid(jnp.sum(user_repr["geo_proj"] * v_r, axis=-1))
    gate = user_repr["gate"] if gate_override is None else gate_override
    core = (
        gate[:, 0] * s_situation
        + gate[:, 1] * s_city
        + gate[:, 2] * s_interest
        + gate[:, 3] * s_geo
    )
    trend = recency_trending_boost(ctx.reel_age_days[rid], ctx.reel_pop[rid], cfg.trend_lambda)
    return core + cfg.trend_mix * trend, {
        "S_situation": s_situation,
        "S_city": s_city,
        "S_interest": s_interest,
        "S_geo": s_geo,
        "S_trend": trend,
    }


def forward_bpr_batch(params, cfg, ctx, uid, pos_rid, neg_rid):
    text_reprs, situation_probs = encode_all_reels(params, cfg, ctx)
    user_repr = user_representation(params, cfg, ctx, uid, situation_probs, text_reprs)
    y_pos, pos_components = score_candidates(
        params, cfg, ctx, uid, pos_rid, user_repr, situation_probs, text_reprs
    )
    y_neg, neg_components = score_candidates(
        params, cfg, ctx, uid, neg_rid, user_repr, situation_probs, text_reprs
    )
    adversary_logits = adversary_forward(params["fair_gate"], user_repr["b_i"], y_pos)
    return {
        "y_pos": y_pos,
        "y_neg": y_neg,
        "gate": user_repr["gate"],
        "adversary_logits": adversary_logits,
        "all_situation_probs": situation_probs,
        "pos_components": pos_components,
        "neg_components": neg_components,
        "s_i": user_repr["s_i"],
    }


def score_user_candidates(params, cfg, ctx, uid: int, candidate_rids):
    """Deterministic serving score for a single user; no exploration side effects."""
    uid_arr = jnp.full((len(candidate_rids),), uid, dtype=jnp.int32)
    rid = jnp.asarray(candidate_rids, dtype=jnp.int32)
    text_reprs, probs = encode_all_reels(params, cfg, ctx)
    one_user = user_representation(
        params, cfg, ctx, jnp.asarray([uid], dtype=jnp.int32), probs, text_reprs
    )
    expanded = {k: jnp.repeat(v, len(candidate_rids), axis=0) for k, v in one_user.items()}
    scores, components = score_candidates(
        params, cfg, ctx, uid_arr, rid, expanded, probs, text_reprs
    )
    return scores, components, one_user["gate"][0]
