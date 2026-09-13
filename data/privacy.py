"""RAPPOR-style local randomization for categorical city values.

This is a research prototype, not a production privacy implementation. We expose the
instantaneous bit-channel epsilon and a conservative n-bit upper bound rather than
claiming a single tight vector-level epsilon. Permanent memoization is deterministic
for reproducibility; a production implementation must use a secret PRF/hash.
"""

from __future__ import annotations

import math
import numpy as np


def _bloom_hash(value: int, bit_idx: int, n_bits: int) -> int:
    return int((value * 1_000_003 + bit_idx * 97_411 + 17) % n_bits)


def bloom_encode_cities(city_ids: np.ndarray, n_bits: int, n_hashes: int) -> np.ndarray:
    bits = np.zeros((len(city_ids), n_bits), dtype=np.int8)
    for row, city in enumerate(np.asarray(city_ids).tolist()):
        for h in range(n_hashes):
            bits[row, _bloom_hash(int(city), h, n_bits)] = 1
    return bits


def rappor_epsilon_per_bit(f: float, p: float, q: float) -> float:
    """Exact epsilon of the composed permanent+instantaneous binary channel per bit."""
    if not (0 <= f <= 1 and 0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError("f, p, q must be in [0, 1]")
    if p == 0 or q == 1:
        return math.inf
    # P(B'=1 | B=1) = 1-f/2, P(B'=1 | B=0) = f/2.
    t = 1.0 - f / 2.0
    b = f / 2.0
    r1 = q * t + p * b
    r0 = q * b + p * t
    return float(math.log(max(r1 / r0, r0 / r1)))


def rappor_vector_epsilon_upper_bound(n_bits: int, f: float, p: float, q: float) -> float:
    """Conservative basic-composition bound across independently reported Bloom bits."""
    return float(n_bits * rappor_epsilon_per_bit(f, p, q))


def rappor_report(
    true_bits: np.ndarray,
    rng: np.random.Generator,
    f: float,
    p: float,
    q: float,
    memo_seed: np.ndarray | None = None,
) -> np.ndarray:
    """Two-stage randomized response: memoized permanent RR + instantaneous RR."""
    true_bits = np.asarray(true_bits, dtype=np.int8)
    n, m = true_bits.shape
    if memo_seed is None:
        memo_seed = np.arange(n, dtype=np.int64)
    memo_seed = np.asarray(memo_seed, dtype=np.int64)

    # Reproducible memoization surrogate. Replace with a secret keyed PRF in production.
    idx = np.arange(m, dtype=np.int64)[None, :]
    seed = memo_seed[:, None]
    raw = (seed * 1_103_515_245 + idx * 12_345 + 678_679) % 2_147_483_647
    u1 = (raw % 10_000) / 10_000.0
    u2 = ((raw * 7 + 13) % 10_000) / 10_000.0

    forced = (u2 >= 0.5).astype(np.int8)
    permanent = np.where(u1 < f, forced, true_bits)
    inst_u = rng.random((n, m))
    threshold = np.where(permanent == 1, q, p)
    return (inst_u < threshold).astype(np.float32)


def privatize_locations(
    user_city: np.ndarray,
    n_bits: int,
    n_hashes: int,
    rng: np.random.Generator,
    f: float = 0.5,
    p: float = 0.5,
    q: float = 0.75,
    memo_seed: np.ndarray | None = None,
) -> np.ndarray:
    bloom = bloom_encode_cities(user_city, n_bits=n_bits, n_hashes=n_hashes)
    return rappor_report(bloom, rng, f=f, p=p, q=q, memo_seed=memo_seed)
