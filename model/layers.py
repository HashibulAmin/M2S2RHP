"""Shared JAX layers: embeddings, MLPs, and a small Adam optimizer."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_embedding(key, num_embeddings, embedding_dim):
    return jax.random.normal(key, (num_embeddings, embedding_dim)) * 0.02


def embedding_lookup(embedding_matrix, indices):
    return jnp.take(embedding_matrix, indices, axis=0)


def init_mlp(key, layer_sizes):
    params = []
    keys = jax.random.split(key, max(len(layer_sizes) - 1, 1))
    for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w = jax.random.normal(k, (in_dim, out_dim)) * jnp.sqrt(2.0 / max(in_dim, 1))
        b = jnp.zeros((out_dim,))
        params.append({"w": w, "b": b})
    return params


def mlp_forward(params, x, final_activation=None):
    out = x
    for i, layer in enumerate(params):
        out = jnp.dot(out, layer["w"]) + layer["b"]
        if i < len(params) - 1:
            out = jax.nn.relu(out)
    return final_activation(out) if final_activation is not None else out


def tree_zeros_like(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_init(params):
    return {"m": tree_zeros_like(params), "v": tree_zeros_like(params), "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(params, grads, state, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, max_grad_norm=1.0):
    t = state["t"] + 1
    leaves = jax.tree_util.tree_leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g * g) for g in leaves))
    scale = jnp.minimum(1.0, max_grad_norm / jnp.maximum(global_norm, eps))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    m = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state["v"], grads)
    b1_corr = 1.0 - beta1**t
    b2_corr = 1.0 - beta2**t
    step = jax.tree_util.tree_map(
        lambda mm, vv: learning_rate * (mm / b1_corr) / (jnp.sqrt(vv / b2_corr) + eps), m, v
    )
    new_params = jax.tree_util.tree_map(lambda p, s: p - s, params, step)
    return new_params, {"m": m, "v": v, "t": t}
