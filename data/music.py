"""Utilities for building compact lyrics/music and playlist features.

The real-data helpers are dependency-free. They intentionally keep the repository free of
licensed lyrics: callers may tokenize lawfully obtained lyrics text or, preferably, supply
pre-computed lyric embeddings from an approved encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MusicTrack:
    track_id: int
    lyrics: str
    audio_features: np.ndarray


@dataclass(frozen=True)
class PlaylistEvent:
    user_id: int
    track_id: int
    timestamp: float
    weight: float = 1.0


def stable_token_id(token: str, vocab_size: int) -> int:
    """Deterministically hash a token into a bounded vocabulary id."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    digest = hashlib.blake2b(token.strip().lower().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % vocab_size


def tokenize_lyrics(text: str, vocab_size: int, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert lawfully obtained lyric text into fixed-size token ids and a mask."""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    tokens = [t for t in text.split() if t.strip()]
    ids = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=bool)
    for i, token in enumerate(tokens[:max_len]):
        ids[i] = stable_token_id(token, vocab_size)
        mask[i] = True
    if not mask.any():
        mask[0] = True
        ids[0] = stable_token_id("<empty>", vocab_size)
    return ids, mask


def build_playlist_arrays(
    events: Iterable[PlaylistEvent],
    n_users: int,
    playlist_len: int,
    as_of_timestamp: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-length, time-safe playlist matrix from timestamped events.

    Only events at or before ``as_of_timestamp`` are used. For each user, the most recent
    unique track occurrences are retained, making the function suitable for chronological
    evaluation without future playlist leakage.
    """
    if n_users <= 0 or playlist_len <= 0:
        raise ValueError("n_users and playlist_len must be positive")
    grouped: dict[int, dict[int, PlaylistEvent]] = {u: {} for u in range(n_users)}
    for event in events:
        if not 0 <= event.user_id < n_users:
            continue
        if as_of_timestamp is not None and event.timestamp > as_of_timestamp:
            continue
        prior = grouped[event.user_id].get(event.track_id)
        if prior is None or event.timestamp >= prior.timestamp:
            grouped[event.user_id][event.track_id] = event

    track_ids = np.zeros((n_users, playlist_len), dtype=np.int32)
    weights = np.zeros((n_users, playlist_len), dtype=np.float32)
    for user_id, by_track in grouped.items():
        rows = sorted(by_track.values(), key=lambda e: e.timestamp, reverse=True)[:playlist_len]
        # Reverse once more so older retained events precede newer ones inside the profile;
        # the model itself uses the weights, not positional order.
        rows = list(reversed(rows))
        for j, event in enumerate(rows):
            track_ids[user_id, j] = int(event.track_id)
            weights[user_id, j] = max(float(event.weight), 0.0)
    return track_ids, weights
