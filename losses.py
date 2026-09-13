"""M²S²-Rec losses and explicit adversarial training objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from model.m2s2rec import forward_bpr_batch


def bpr_loss(y_pos, y_neg):
    return -jnp.mean(jax.nn.log_sigmoid(y_pos - y_neg))


def situation_bce_loss(situation_probs, weak_labels):
    p = jnp.clip(situation_probs, 1e-7, 1.0 - 1e-7)
    return -jnp.mean(weak_labels * jnp.log(p) + (1.0 - weak_labels) * jnp.log(1.0 - p))


def adversary_ce_loss(adversary_logits, protected_attr):
    logp = jax.nn.log_softmax(adversary_logits, axis=-1)
    onehot = jax.nn.one_hot(protected_attr, adversary_logits.shape[-1])
    return -jnp.mean(jnp.sum(onehot * logp, axis=-1))


def l2_penalty(params):
    return sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params) if jnp.ndim(p) > 0)


def forward_objectives(params, cfg, ctx, batch, reel_weak_situation):
    out = forward_bpr_batch(params, cfg, ctx, batch["uid"], batch["pos_rid"], batch["neg_rid"])
    l_bpr = bpr_loss(out["y_pos"], out["y_neg"])
    l_cls = situation_bce_loss(out["all_situation_probs"], reel_weak_situation)
    l_adv = adversary_ce_loss(out["adversary_logits"], batch["protected_attr"])
    l_reg = l2_penalty(params)
    task_loss = l_bpr + cfg.lambda_cls * l_cls + cfg.l2_reg * l_reg
    total_reported = task_loss + cfg.lambda_adv * l_adv
    return total_reported, task_loss, l_adv, out


def total_loss(params, cfg, ctx, batch, reel_weak_situation):
    total_reported, _, l_adv, out = forward_objectives(params, cfg, ctx, batch, reel_weak_situation)
    l_bpr = bpr_loss(out["y_pos"], out["y_neg"])
    l_cls = situation_bce_loss(out["all_situation_probs"], reel_weak_situation)
    l_reg = l2_penalty(params)
    aux = {
        "loss": total_reported,
        "l_bpr": l_bpr,
        "l_cls": l_cls,
        "l_adv": l_adv,
        "l_reg": l_reg,
        "mean_gate": jnp.mean(out["gate"], axis=0),
        "mean_margin": jnp.mean(out["y_pos"] - out["y_neg"]),
    }
    return total_reported, aux
