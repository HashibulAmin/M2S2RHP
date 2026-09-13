"""Synthetic, leakage-controlled data pipeline for the M²S²-Rec prototype.

The generator intentionally mirrors the paper while making the experimental protocol
reproducible and analyzable:
- city distances come from 2-D centroids (valid metric, zero diagonal)
- users have chronological interaction histories with signed implicit feedback
- the final positive interactions are held out as validation/test targets
- cold-start users have no training history but do have evaluation targets
- reel situation labels are correlated with text markers, not sampled independently
- protected attributes are stable user attributes, never fresh random labels
- city features are privatized with a RAPPOR-style mechanism before K-hop LightGCN
- BPR positives come only from positive engagement events
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from data.graph import lightgcn_propagate, sample_social_graph
from data.privacy import privatize_locations


SITUATIONS = ("job_search", "new_parent", "moving", "graduation")


class Config(NamedTuple):
    d_model: int = 12
    d_style: int = 6
    music_audio_dim: int = 4
    music_vocab_size: int = 64
    music_max_seq_len: int = 6
    music_text_layers: int = 1
    music_text_heads: int = 3
    d_cat: int = 3
    d_fair: int = 6
    d_hidden: int = 12
    n_reels: int = 20
    n_cities: int = 4
    n_education: int = 3
    n_gender: int = 3
    n_occupation: int = 4
    n_users: int = 10
    rappor_bloom_bits: int = 12
    rappor_n_hashes: int = 2
    rappor_f: float = 0.5
    rappor_p: float = 0.5
    rappor_q: float = 0.75
    situations: tuple = SITUATIONS
    lambda_interest: float = 0.05
    playlist_mix: float = 0.75
    city_eta: float = 0.5
    city_tau_km: float = 80.0
    lambda_cls: float = 1.0
    lambda_adv: float = 0.001
    l2_reg: float = 1e-4
    kappa_explicit: float = 0.4
    trace_days: float = 90.0
    learning_rate: float = 3e-4
    epochs: int = 1
    seed: int = 42
    batch_size: int = 4
    hist_len: int = 8
    playlist_len: int = 6
    vocab_size: int = 64
    max_seq_len: int = 6
    n_protected_classes: int = 4
    n_text_layers: int = 1
    n_text_heads: int = 3
    n_gcn_hops: int = 2
    gate_epsilon: float = 0.05
    trend_lambda: float = 0.03
    trend_mix: float = 0.02
    n_batches: int = 1
    cold_start_frac: float = 0.10
    val_holdout: int = 1
    test_holdout: int = 1
    min_train_events: int = 4
    peer_delta_km: float = 120.0
    eval_k: int = 5


class MockWorldData:
    """Generates a reproducible synthetic recommendation world."""

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng or np.random.default_rng(cfg.seed)
        n_sit = len(cfg.situations)
        if cfg.vocab_size < n_sit * 8:
            raise ValueError("vocab_size must reserve enough ids for situation markers")

        # --- Cities and users -------------------------------------------------
        coords = rng.uniform(0.0, 800.0, size=(cfg.n_cities, 2)).astype(np.float32)
        diff = coords[:, None, :] - coords[None, :, :]
        self.city_coords = coords
        self.city_dist_km = np.sqrt(np.sum(diff**2, axis=-1)).astype(np.float32)
        np.fill_diagonal(self.city_dist_km, 0.0)

        self.user_city = rng.integers(0, cfg.n_cities, cfg.n_users, dtype=np.int32)
        self.user_age_norm = rng.uniform(0.15, 0.85, cfg.n_users).astype(np.float32)
        self.user_edu = rng.integers(0, cfg.n_education, cfg.n_users, dtype=np.int32)
        self.user_gender = rng.integers(0, cfg.n_gender, cfg.n_users, dtype=np.int32)
        self.user_occ = rng.integers(0, cfg.n_occupation, cfg.n_users, dtype=np.int32)
        self.user_behavior = rng.uniform(0.0, 1.0, (cfg.n_users, 3)).astype(np.float32)

        self.user_s_explicit = np.zeros((cfg.n_users, n_sit), dtype=np.float32)
        declare = rng.random(cfg.n_users) < 0.35
        sit_ids = rng.integers(0, n_sit, cfg.n_users)
        self.user_s_explicit[declare, sit_ids[declare]] = 1.0

        # --- Reels ------------------------------------------------------------
        self.reel_situation = np.zeros((cfg.n_reels, n_sit), dtype=np.float32)
        self.reel_tokens = np.zeros((cfg.n_reels, cfg.max_seq_len), dtype=np.int32)
        self.reel_token_mask = np.ones((cfg.n_reels, cfg.max_seq_len), dtype=bool)
        self.reel_city = rng.integers(0, cfg.n_cities, cfg.n_reels, dtype=np.int32)
        self.reel_age_days = rng.uniform(0.0, cfg.trace_days, cfg.n_reels).astype(np.float32)
        self.reel_pop = rng.integers(1, 500, cfg.n_reels).astype(np.float32)

        # Fixed synthetic visual features: situation-correlated, but not item-id learnable.
        visual_basis = rng.normal(0.0, 0.4, size=(n_sit, cfg.d_model)).astype(np.float32)
        self.reel_visual = np.zeros((cfg.n_reels, cfg.d_model), dtype=np.float32)
        self.n_tracks = cfg.n_reels
        self.track_lyrics = np.zeros((self.n_tracks, cfg.music_max_seq_len), dtype=np.int32)
        self.track_lyrics_mask = np.ones((self.n_tracks, cfg.music_max_seq_len), dtype=bool)
        self.track_audio = rng.normal(0.0, 1.0, (self.n_tracks, cfg.music_audio_dim)).astype(np.float32)
        self.reel_track = np.arange(cfg.n_reels, dtype=np.int32)

        marker_block = max(8, cfg.vocab_size // n_sit)
        for k in range(cfg.n_reels):
            n_lab = 1 + int(rng.random() < 0.30)
            labels = rng.choice(n_sit, size=n_lab, replace=False)
            self.reel_situation[k, labels] = 1.0
            self.reel_visual[k] = visual_basis[labels].mean(axis=0) + rng.normal(0, 0.12, cfg.d_model)

            # Marker IDs and lexical-ish tokens are correlated with the weak labels.
            ids: list[int] = []
            for label in labels.tolist():
                lo = int(label * marker_block)
                ids.append(lo)  # reserved hashtag-like marker
            while len(ids) < cfg.max_seq_len:
                primary = int(labels[0])
                lo = int(primary * marker_block + 1)
                hi = min(int((primary + 1) * marker_block), cfg.vocab_size)
                ids.append(int(rng.integers(lo, max(hi, lo + 1))))
            self.reel_tokens[k] = np.asarray(ids[: cfg.max_seq_len], dtype=np.int32)

        # Music tracks have lyrics-like tokens correlated with the reel's primary situation,
        # plus fixed acoustic descriptors. No copyrighted lyric text is embedded in the repo.
        music_marker_block = max(8, cfg.music_vocab_size // n_sit)
        for m in range(self.n_tracks):
            primary = int(np.argmax(self.reel_situation[m]))
            lo = int(primary * music_marker_block)
            hi = min(int((primary + 1) * music_marker_block), cfg.music_vocab_size)
            ids = [lo]
            while len(ids) < cfg.music_max_seq_len:
                ids.append(int(rng.integers(lo, max(hi, lo + 1))))
            self.track_lyrics[m] = np.asarray(ids[: cfg.music_max_seq_len], dtype=np.int32)

        # The synthetic weak labels intentionally approximate hashtag bootstrapping.
        # In a real system these would be derived from observed hashtags, then audited.
        self.reel_weak_situation = self.reel_situation.copy()

        # --- User playlists ---------------------------------------------------
        # Playlists are sampled independently but from each user's learned situation/city
        # priors; only playlist evidence available at training time is used for the user's
        # music representation, preventing test-event leakage.
        self.playlist_track = np.zeros((cfg.n_users, cfg.playlist_len), dtype=np.int32)
        self.playlist_weight = np.zeros((cfg.n_users, cfg.playlist_len), dtype=np.float32)
        for uid in range(cfg.n_users):
            pref = self.user_situation_prior(uid)
            choices = []
            for _ in range(cfg.playlist_len):
                sit = int(rng.choice(n_sit, p=pref))
                pool = np.where(self.reel_situation[:, sit] > 0)[0]
                if pool.size == 0:
                    pool = np.arange(self.n_tracks)
                choices.append(int(rng.choice(pool)))
            self.playlist_track[uid] = np.asarray(choices, dtype=np.int32)
            self.playlist_weight[uid] = rng.uniform(0.4, 1.0, cfg.playlist_len).astype(np.float32)

        # --- Full chronological event log -----------------------------------
        cold_n = max(1, int(round(cfg.cold_start_frac * cfg.n_users)))
        self.cold_users = np.asarray(
            sorted(rng.choice(cfg.n_users, cold_n, replace=False).tolist()), dtype=np.int32
        )
        cold_set = set(self.cold_users.tolist())

        self.hist_reel = np.zeros((cfg.n_users, cfg.hist_len), dtype=np.int32)
        self.hist_weight = np.zeros((cfg.n_users, cfg.hist_len), dtype=np.float32)
        self.hist_day = np.zeros((cfg.n_users, cfg.hist_len), dtype=np.float32)
        self.hist_mask = np.zeros((cfg.n_users, cfg.hist_len), dtype=bool)
        self.val_reel = np.full(cfg.n_users, -1, dtype=np.int32)
        self.test_reel = np.full(cfg.n_users, -1, dtype=np.int32)

        for uid in range(cfg.n_users):
            city = int(self.user_city[uid])
            pref = self.user_situation_prior(uid)
            if uid in cold_set:
                # Cold-start evaluation targets are independently sampled positives.
                same = np.where(self.reel_city == city)[0]
                pool = same if same.size else np.arange(cfg.n_reels)
                self.test_reel[uid] = int(rng.choice(pool))
                continue

            event_count = int(rng.integers(cfg.min_train_events + cfg.val_holdout + cfg.test_holdout, cfg.hist_len + 1))
            days = np.sort(rng.uniform(0.0, cfg.trace_days, event_count)).astype(np.float32)
            reels: list[int] = []
            for _ in range(event_count):
                if rng.random() < 0.50:
                    pool = np.where(self.reel_city == city)[0]
                else:
                    sit = int(rng.choice(n_sit, p=pref))
                    pool = np.where(self.reel_situation[:, sit] > 0)[0]
                if pool.size == 0:
                    pool = np.arange(cfg.n_reels)
                reels.append(int(rng.choice(pool)))

            # Positive engagement dominates; skips are explicit negatives.
            weights = rng.uniform(0.25, 1.0, event_count).astype(np.float32)
            skip = rng.random(event_count) < 0.22
            weights[skip] *= -1.0

            positive_idx = np.flatnonzero(weights > 0)
            holdout_needed = cfg.val_holdout + cfg.test_holdout
            holdable = positive_idx.size >= holdout_needed + cfg.min_train_events
            if holdable:
                test_idx = int(positive_idx[-1])
                val_idx = int(positive_idx[-2]) if cfg.val_holdout else None
                train_count = event_count
                if val_idx is not None:
                    self.val_reel[uid] = reels[val_idx]
                self.test_reel[uid] = reels[test_idx]
                keep = np.ones(event_count, dtype=bool)
                keep[test_idx] = False
                if val_idx is not None:
                    keep[val_idx] = False
                train_indices = np.flatnonzero(keep)[: cfg.hist_len]
                train_count = train_indices.size
            else:
                # No valid chronological holdout is available; keep all observed events
                # for training and create evaluation targets from unseen catalog items.
                train_indices = np.arange(event_count)
                train_count = event_count

            self.hist_mask[uid, :train_count] = True
            self.hist_reel[uid, :train_count] = np.asarray(reels, dtype=np.int32)[train_indices]
            self.hist_weight[uid, :train_count] = weights[train_indices]
            self.hist_day[uid, :train_count] = days[train_indices]

            # Guarantee evaluation targets that are NOT in the training positives.
            train_pos = set(self.hist_reel[uid, (self.hist_mask[uid] & (self.hist_weight[uid] > 0))].tolist())
            unseen_same = np.asarray(
                [r for r in np.where(self.reel_city == city)[0].tolist() if r not in train_pos],
                dtype=np.int32,
            )
            fallback_pool = unseen_same if unseen_same.size else np.asarray(
                [r for r in np.arange(cfg.n_reels).tolist() if r not in train_pos], dtype=np.int32
            )
            if self.test_reel[uid] < 0:
                self.test_reel[uid] = int(rng.choice(fallback_pool))
            if self.val_reel[uid] < 0:
                remaining = fallback_pool[fallback_pool != self.test_reel[uid]]
                self.val_reel[uid] = int(rng.choice(remaining if remaining.size else fallback_pool))

        # --- Social/spatial signal ------------------------------------------
        self.neighbors = sample_social_graph(rng, self.user_city, mean_degree=4.0)
        g_private = privatize_locations(
            self.user_city,
            n_bits=cfg.rappor_bloom_bits,
            n_hashes=cfg.rappor_n_hashes,
            rng=rng,
            f=cfg.rappor_f,
            p=cfg.rappor_p,
            q=cfg.rappor_q,
            memo_seed=np.arange(cfg.n_users, dtype=np.int64),
        )
        self.g_all = lightgcn_propagate(g_private, self.neighbors, cfg.n_gcn_hops)
        self._rng = rng

    def user_situation_prior(self, uid: int) -> np.ndarray:
        prior = self.user_s_explicit[uid].copy()
        if prior.sum() <= 0:
            prior[:] = 1.0
        return prior / prior.sum()

    def positive_history(self, uid: int) -> np.ndarray:
        mask = self.hist_mask[uid] & (self.hist_weight[uid] > 0)
        return self.hist_reel[uid, mask]

    def known_positive_reels(self, uid: int) -> np.ndarray:
        vals = list(self.positive_history(uid))
        for rid in (self.val_reel[uid], self.test_reel[uid]):
            if rid >= 0:
                vals.append(int(rid))
        return np.unique(np.asarray(vals, dtype=np.int32)) if vals else np.empty(0, dtype=np.int32)

    def sample_bpr_batch(self, batch_size: int) -> dict:
        """Sample a pairwise batch without using skipped or held-out positives as negatives."""
        cfg = self.cfg
        rng = self._rng
        uids = rng.integers(0, cfg.n_users, batch_size, dtype=np.int32)
        pos = np.zeros(batch_size, dtype=np.int32)
        neg = np.zeros(batch_size, dtype=np.int32)

        all_reels = np.arange(cfg.n_reels, dtype=np.int32)
        for b, uid_raw in enumerate(uids.tolist()):
            uid = int(uid_raw)
            hist_pos = self.positive_history(uid)
            if hist_pos.size:
                pos[b] = int(rng.choice(hist_pos))
            else:
                same = np.where(self.reel_city == self.user_city[uid])[0]
                pos[b] = int(rng.choice(same if same.size else all_reels))

            known = self.known_positive_reels(uid)
            cand = np.setdiff1d(all_reels, known, assume_unique=False)
            if cand.size == 0:
                cand = all_reels
            far = cand[self.reel_city[cand] != self.user_city[uid]]
            pool = far if far.size else cand
            overlap = self.reel_situation[pool] @ self.reel_situation[pos[b]]
            mismatch = pool[overlap <= 0.5]
            neg[b] = int(rng.choice(mismatch if mismatch.size else pool))

        return {
            "uid": uids,
            "pos_rid": pos,
            "neg_rid": neg,
            "protected_attr": self.user_gender[uids].astype(np.int32),
        }


def get_dataloader(world: MockWorldData, cfg: Config, num_batches: int | None = None):
    for _ in range(num_batches if num_batches is not None else cfg.n_batches):
        yield world.sample_bpr_batch(cfg.batch_size)
