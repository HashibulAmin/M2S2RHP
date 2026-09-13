"""Adversarial fairness gate.

Training uses an explicit minimax update rather than a custom autodiff primitive:
- adversary parameters minimize protected-attribute cross entropy
- the fairness-gate parameters receive the opposite gradient of that loss
This is equivalent in objective to gradient reversal, but is easier to stabilize and
inspect in a small research prototype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.layers import init_mlp, mlp_forward


def init_fair_gate(key, cfg, in_dim):
    k_gate, k_adv = jax.random.split(key)
    return {
        "gate": init_mlp(k_gate, [in_dim, cfg.d_hidden, cfg.d_fair]),
        "adversary": init_mlp(k_adv, [cfg.d_fair + 1, cfg.d_hidden, cfg.n_protected_classes]),
    }


def fair_gate_forward(params, x):
    return mlp_forward(params["gate"], x, jax.nn.tanh)


def adversary_forward(params, b_i, y_hat):
    # Score is diagnostic context only; the adversary must not destabilize the ranking towers.
    adv_in = jnp.concatenate([b_i, jax.lax.stop_gradient(y_hat[:, None])], axis=-1)
    return mlp_forward(params["adversary"], adv_in)
