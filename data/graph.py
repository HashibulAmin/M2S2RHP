"""K-hop LightGCN-style propagation over privatized user location features."""

from __future__ import annotations

import numpy as np


def lightgcn_propagate(h0: np.ndarray, neighbors: list[list[int]], n_hops: int) -> np.ndarray:
    """Return the mean of layers 0..K using symmetric degree normalization."""
    h0 = np.asarray(h0, dtype=np.float32)
    n = h0.shape[0]
    deg = np.asarray([max(len(nbrs), 1) for nbrs in neighbors], dtype=np.float32)
    inv_sqrt = 1.0 / np.sqrt(deg)

    layers = [h0]
    prev = h0
    for _ in range(n_hops):
        nxt = np.zeros_like(prev)
        for i, nbrs in enumerate(neighbors):
            if not nbrs:
                nxt[i] = prev[i]
                continue
            for j in nbrs:
                nxt[i] += inv_sqrt[i] * inv_sqrt[j] * prev[j]
        layers.append(nxt)
        prev = nxt
    return np.mean(np.stack(layers, axis=0), axis=0)


def sample_social_graph(
    rng: np.random.Generator,
    user_city: np.ndarray,
    mean_degree: float = 4.0,
    same_city_prob: float = 0.7,
) -> list[list[int]]:
    """Create a simple unique undirected graph with city-homophily and isolated nodes allowed."""
    n = int(user_city.shape[0])
    target_edges = int(round(n * mean_degree / 2.0))
    groups: dict[int, list[int]] = {}
    for i, city in enumerate(user_city.tolist()):
        groups.setdefault(int(city), []).append(i)

    neighbors = [set() for _ in range(n)]
    attempts = 0
    max_attempts = max(20 * target_edges, 100)
    while sum(len(s) for s in neighbors) // 2 < target_edges and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, n))
        same = groups[int(user_city[i])]
        if rng.random() < same_city_prob and len(same) > 1:
            j = int(same[int(rng.integers(0, len(same)))])
        else:
            j = int(rng.integers(0, n))
        if i == j or j in neighbors[i]:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)
    return [sorted(s) for s in neighbors]
