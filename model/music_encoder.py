"""Lyrics + audio encoder used for user-playlist and reel-music representations.

The implementation is intentionally dependency-light: a small Transformer over lyrics-like
 tokens is fused with fixed audio features. In a production system these inputs can be
replaced with pretrained multilingual language/audio encoders.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.layers import init_embedding
from model.text_encoder import init_transformer_layer, transformer_layer_forward


def init_music_encoder(key, cfg):
    k_tok, k_pos, k_layers, k_audio, k_proj = jax.random.split(key, 5)
    if cfg.d_style % cfg.music_text_heads != 0:
        raise ValueError("d_style must be divisible by music_text_heads")
    layers = [
        init_transformer_layer(k, cfg.d_style, cfg.music_text_heads)
        for k in jax.random.split(k_layers, cfg.music_text_layers)
    ]
    return {
        "token_emb": init_embedding(k_tok, cfg.music_vocab_size, cfg.d_style),
        "pos_emb": init_embedding(k_pos, cfg.music_max_seq_len, cfg.d_style),
        "layers": layers,
        "audio_w": jax.random.normal(k_audio, (cfg.music_audio_dim, cfg.d_style)) * 0.03,
        "audio_b": jnp.zeros((cfg.d_style,)),
        "proj_w": jax.random.normal(k_proj, (cfg.d_style, cfg.d_model)) * 0.03,
        "proj_b": jnp.zeros((cfg.d_model,)),
    }


def encode_music(params, cfg, lyrics_tokens, lyrics_mask, audio_features):
    seq = lyrics_tokens.shape[1]
    if seq > cfg.music_max_seq_len:
        raise ValueError("music lyric sequence exceeds configured music_max_seq_len")
    h = jnp.take(params["token_emb"], lyrics_tokens, axis=0) + params["pos_emb"][:seq][None, :, :]
    for layer in params["layers"]:
        h = transformer_layer_forward(layer, h, lyrics_mask, cfg.music_text_heads)
    mask_f = lyrics_mask.astype(h.dtype)[:, :, None]
    denom = jnp.maximum(jnp.sum(mask_f, axis=1), 1.0)
    lyrics_repr = jnp.sum(h * mask_f, axis=1) / denom
    audio_repr = jnp.dot(audio_features, params["audio_w"]) + params["audio_b"]
    fused = jnp.tanh(lyrics_repr + audio_repr)
    return jnp.dot(fused, params["proj_w"]) + params["proj_b"]
