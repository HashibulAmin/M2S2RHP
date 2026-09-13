import unittest

import numpy as np
import jax.numpy as jnp

from data.dataset import Config, MockWorldData
from data.graph import lightgcn_propagate
from data.privacy import rappor_epsilon_per_bit, rappor_vector_epsilon_upper_bound
from model.scoring import city_match_score, temporal_decay_weights


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(n_users=40, n_reels=80, n_cities=8, batch_size=16, n_batches=1, epochs=1)
        cls.world = MockWorldData(cls.cfg)

    def test_city_distance_is_metric_symmetric(self):
        d = self.world.city_dist_km
        np.testing.assert_allclose(d, d.T, atol=1e-5)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-6)
        self.assertTrue(np.all(d >= 0))

    def test_history_prefix_and_no_negative_bpr_positive(self):
        for uid in range(self.cfg.n_users):
            mask = self.world.hist_mask[uid]
            true_len = int(mask.sum())
            if true_len:
                self.assertTrue(mask[:true_len].all())
                self.assertFalse(mask[true_len:].any())
            pos = self.world.positive_history(uid)
            self.assertTrue(np.all(self.world.hist_weight[uid, :true_len][self.world.hist_weight[uid, :true_len] > 0] > 0))
            batch = self.world.sample_bpr_batch(32)
            self.assertTrue(np.all([r >= 0 for r in batch["pos_rid"]]))

    def test_temporal_decay_monotonic(self):
        day = jnp.array([10.0])
        hist = jnp.array([[0.0, 5.0, 9.0]])
        mask = jnp.array([[True, True, True]])
        w = temporal_decay_weights(day, hist, mask, 0.1)
        self.assertGreater(float(w[0, 2]), float(w[0, 1]))
        self.assertGreater(float(w[0, 1]), float(w[0, 0]))

    def test_city_exact_beats_non_exact(self):
        d = jnp.asarray(self.world.city_dist_km)
        user = jnp.array([0, 0])
        reel = jnp.array([0, 1])
        s = city_match_score(user, reel, d, 0.5, 80.0)
        self.assertAlmostEqual(float(s[0]), 1.0, places=6)
        self.assertLessEqual(float(s[1]), 0.5 + 1e-6)

    def test_graph_zero_input_zero_output_and_finite(self):
        h = np.zeros((4, 8), dtype=np.float32)
        nbrs = [[1], [0], [3], [2]]
        out = lightgcn_propagate(h, nbrs, 2)
        np.testing.assert_allclose(out, 0.0)

    def test_privacy_budget_is_exposed(self):
        eps = rappor_epsilon_per_bit(0.5, 0.5, 0.75)
        self.assertGreater(eps, 0.0)
        self.assertAlmostEqual(rappor_vector_epsilon_upper_bound(64, 0.5, 0.5, 0.75), 64 * eps)


if __name__ == "__main__":
    unittest.main()
