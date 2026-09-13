"""Non-learnable scoring primitives and serving-time ranking helpers."""

from __future__ import annotations

import jax.numpy as jnp


def city_match_score(user_city, reel_city, city_dist_km, eta: float, tau_km: float):
    exact = (user_city == reel_city).astype(jnp.float32)
    dist = city_dist_km[user_city, reel_city]
    soft = eta * jnp.exp(-dist / jnp.maximum(tau_km, 1e-6)) * (1.0 - exact)
    return exact + soft


def temporal_decay_weights(day_now, hist_day, hist_mask, lam):
    delta = jnp.clip(day_now[:, None] - hist_day, 0.0, None)
    mask = hist_mask.astype(jnp.float32)
    if jnp.ndim(lam) == 0:
        return jnp.exp(-lam * delta) * mask
    return jnp.exp(-lam[None, None, :] * delta[:, :, None]) * mask[:, :, None]


def temporal_interest_vector(hist_item_emb, hist_weight, day_now, hist_day, hist_mask, lam_t):
    decay = temporal_decay_weights(day_now, hist_day, hist_mask, lam_t)
    return jnp.sum((hist_weight * decay)[:, :, None] * hist_item_emb, axis=1)


def temporal_situation_vector(hist_situation_probs, hist_weight, day_now, hist_day, hist_mask, lam_q):
    decay = temporal_decay_weights(day_now, hist_day, hist_mask, lam_q)
    pos_weight = jnp.clip(hist_weight, 0.0, None)
    return jnp.sum((pos_weight[:, :, None] * decay) * hist_situation_probs, axis=1)


def apply_explicit_situation_floor(s_i, s_explicit, kappa):
    return jnp.maximum(s_i, kappa * s_explicit)


def cosine_sim(a, b, eps=1e-8):
    # sqrt(sum(x^2)+eps) has a finite derivative at the exact zero vector,
    # which is essential for cold-start users with empty histories.
    a_den = jnp.sqrt(jnp.sum(a * a, axis=-1, keepdims=True) + eps)
    b_den = jnp.sqrt(jnp.sum(b * b, axis=-1, keepdims=True) + eps)
    return jnp.sum((a / a_den) * (b / b_den), axis=-1)


def peer_similarity(s_i, s_j, city_i, city_j, city_dist_km, delta_local_km):
    return cosine_sim(s_i, s_j) * (city_dist_km[city_i, city_j] < delta_local_km).astype(jnp.float32)


def recency_trending_boost(reel_age_days, reel_pop, lam_trend):
    """Bounded personalization-independent trend feature in [0, 1]."""
    raw = jnp.exp(-lam_trend * reel_age_days) * jnp.log1p(reel_pop)
    return raw / (1.0 + raw)
