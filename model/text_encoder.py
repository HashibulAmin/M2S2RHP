"""Tiny transformer over caption/OCR/ASR-like tokens plus fixed visual features."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.layers import init_embedding


def init_text_encoder(key, cfg):
    k_tok, k_pos, k_layers, k_cls, k_vis = jax.random.split(key, 5)
    if cfg.d_model % cfg.n_text_heads != 0:
        raise ValueError("d_model must be divisible by n_text_heads")
    layers = [
        init_transformer_layer(lk, cfg.d_model, cfg.n_text_heads)
        for lk in jax.random.split(k_layers, cfg.n_text_layers)
    ]
    return {
        "token_emb": init_embedding(k_tok, cfg.vocab_size, cfg.d_model),
        "pos_emb": init_embedding(k_pos, cfg.max_seq_len, cfg.d_model),
        "layers": layers,
        "cls_w": jax.random.normal(k_cls, (cfg.d_model, len(cfg.situations))) * 0.02,
        "cls_b": jnp.zeros((len(cfg.situations),)),
        "visual_w": jax.random.normal(k_vis, (cfg.d_model, cfg.d_model)) * 0.05,
    }


def init_transformer_layer(key, d_model, n_heads):
    keys = jax.random.split(key, 6)
    scale = 0.05
    return {
        "wq": jax.random.normal(keys[0], (d_model, d_model)) * scale,
        "wk": jax.random.normal(keys[1], (d_model, d_model)) * scale,
        "wv": jax.random.normal(keys[2], (d_model, d_model)) * scale,
        "wo": jax.random.normal(keys[3], (d_model, d_model)) * scale,
        "ln1_g": jnp.ones((d_model,)),
        "ln1_b": jnp.zeros((d_model,)),
        "ln2_g": jnp.ones((d_model,)),
        "ln2_b": jnp.zeros((d_model,)),
        "ff1": jax.random.normal(keys[4], (d_model, 4 * d_model)) * scale,
        "ff1_b": jnp.zeros((4 * d_model,)),
        "ff2": jax.random.normal(keys[5], (4 * d_model, d_model)) * scale,
        "ff2_b": jnp.zeros((d_model,)),
    }


def _layernorm(x, g, b, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return g * (x - mean) / jnp.sqrt(var + eps) + b


def _attention(layer, x, pad_mask, n_heads):
    bsz, seq, d_model = x.shape
    head_dim = d_model // n_heads
    q = jnp.dot(x, layer["wq"]).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = jnp.dot(x, layer["wk"]).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
    v = jnp.dot(x, layer["wv"]).reshape(bsz, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(float(head_dim))
    scores = jnp.where(pad_mask[:, None, None, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.matmul(weights, v).transpose(0, 2, 1, 3).reshape(bsz, seq, d_model)
    return jnp.dot(ctx, layer["wo"])


def transformer_layer_forward(layer, x, pad_mask, n_heads):
    h = _layernorm(x, layer["ln1_g"], layer["ln1_b"])
    x = x + _attention(layer, h, pad_mask, n_heads)
    h = _layernorm(x, layer["ln2_g"], layer["ln2_b"])
    ff = jax.nn.gelu(jnp.dot(h, layer["ff1"]) + layer["ff1_b"])
    return x + jnp.dot(ff, layer["ff2"]) + layer["ff2_b"]


def encode_reels(params, cfg, tokens, mask, visual_features=None):
    seq = tokens.shape[1]
    if seq > cfg.max_seq_len:
        raise ValueError("token sequence exceeds configured max_seq_len")
    embs = jnp.take(params["token_emb"], tokens, axis=0) + params["pos_emb"][:seq][None, :, :]
    h = embs
    for layer in params["layers"]:
        h = transformer_layer_forward(layer, h, mask, cfg.n_text_heads)
    mask_f = mask.astype(h.dtype)[:, :, None]
    denom = jnp.maximum(jnp.sum(mask_f, axis=1, keepdims=True), 1.0)
    text_repr = jnp.sum(h * mask_f, axis=1) / denom[:, 0, :]
    if visual_features is not None:
        text_repr = text_repr + jnp.dot(visual_features, params["visual_w"])
    logits = jnp.dot(text_repr, params["cls_w"]) + params["cls_b"]
    return text_repr, jax.nn.sigmoid(logits)
