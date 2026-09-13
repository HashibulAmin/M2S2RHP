"""Reproducible CPU training entry point for M²S²-Rec.

The fairness objective is optimized explicitly as a minimax game:
  adversary:        minimize CE(protected | b_i)
  fairness gate:   minimize task_loss - lambda_adv * CE(protected | b_i)
All other ranking parameters optimize task_loss only. This avoids injecting adversarial
noise into the interest/content towers while preserving the paper's fairness-gate idea.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from data.dataset import Config, MockWorldData, get_dataloader
from losses import bpr_loss, situation_bce_loss, l2_penalty, forward_objectives
from model.layers import adam_init, adam_update
from model.m2s2rec import build_ctx, init_params


def _stop_adv_params(params):
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x), params["fair_gate"]["adversary"])


def make_train_step(cfg, ctx, reel_weak_situation):
    def objectives(params, batch):
        reported, task_loss, adv_loss, out = forward_objectives(
            params, cfg, ctx, batch, reel_weak_situation
        )
        return reported, task_loss, adv_loss, out

    def task_fn(params, batch):
        return objectives(params, batch)[1]

    def adv_frozen_fn(params, batch):
        frozen = dict(params)
        fair = dict(params["fair_gate"])
        fair["adversary"] = _stop_adv_params(params)
        frozen["fair_gate"] = fair
        return objectives(frozen, batch)[2]

    def adv_full_fn(params, batch):
        return objectives(params, batch)[2]

    @jax.jit
    def train_step(params, opt_state, batch):
        task_loss, task_grads = jax.value_and_grad(task_fn)(params, batch)
        reported, _, adv_loss, out = objectives(params, batch)
        adv_gate_grads = jax.grad(adv_frozen_fn)(params, batch)
        adv_full_grads = jax.grad(adv_full_fn)(params, batch)

        def combine(tg, ag):
            return tg - cfg.lambda_adv * ag

        combined = jax.tree_util.tree_map(combine, task_grads, adv_gate_grads)
        combined = dict(combined)
        combined["fair_gate"] = dict(combined["fair_gate"])
        # Adversary itself minimizes CE, not the reversed objective.
        combined["fair_gate"]["adversary"] = adv_full_grads["fair_gate"]["adversary"]

        new_params, new_state = adam_update(
            params, combined, opt_state, learning_rate=cfg.learning_rate, max_grad_norm=1.0
        )
        aux = {
            "loss": reported,
            "l_bpr": bpr_loss(out["y_pos"], out["y_neg"]),
            "l_cls": situation_bce_loss(out["all_situation_probs"], reel_weak_situation),
            "l_adv": adv_loss,
            "l_reg": l2_penalty(params),
            "mean_gate": jnp.mean(out["gate"], axis=0),
            "mean_margin": jnp.mean(out["y_pos"] - out["y_neg"]),
        }
        return new_params, new_state, aux

    return train_step


def train_loop(world: MockWorldData, cfg: Config, verbose: bool = True):
    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    params = init_params(init_key, cfg)
    ctx = build_ctx(world, world.g_all, cfg)
    labels = jnp.asarray(world.reel_weak_situation, dtype=jnp.float32)
    opt_state = adam_init(params)
    step = make_train_step(cfg, ctx, labels)
    history = []

    for epoch in range(cfg.epochs):
        rows = []
        for batch in get_dataloader(world, cfg):
            jax_batch = {k: jnp.asarray(v) for k, v in batch.items()}
            params, opt_state, aux = step(params, opt_state, jax_batch)
            rows.append(float(aux["loss"]))
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(rows)),
            "bpr": float(aux["l_bpr"]),
            "situation_bce": float(aux["l_cls"]),
            "adversary_ce": float(aux["l_adv"]),
            "margin": float(aux["mean_margin"]),
            "gate": np.asarray(aux["mean_gate"]).round(6).tolist(),
        }
        history.append(row)
        if verbose:
            print(
                f"Epoch {row['epoch']}/{cfg.epochs} | loss={row['loss']:.4f} "
                f"BPR={row['bpr']:.4f} BCE={row['situation_bce']:.4f} "
                f"ADV={row['adversary_ce']:.4f} margin={row['margin']:.4f}"
            )
            print(f"Mean gate [situation, city, interest, geo]: {row['gate']}")
    return params, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=Path, default=Path("artifacts/training_history.json"))
    args = p.parse_args()
    base = Config()
    cfg = base._replace(
        epochs=args.epochs if args.epochs is not None else base.epochs,
        seed=args.seed if args.seed is not None else base.seed,
    )
    world = MockWorldData(cfg)
    print(
        f"World: {cfg.n_users} users, {cfg.n_reels} reels, {cfg.n_cities} cities; "
        f"cold-start={len(world.cold_users)} users"
    )
    _, history = train_loop(world, cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
