import unittest

import numpy as np
import jax
import jax.numpy as jnp

from data.dataset import Config, MockWorldData
from data.graph import lightgcn_propagate
from data.privacy import rappor_epsilon_per_bit, rappor_vector_epsilon_upper_bound
from model.m2s2rec import build_ctx, encode_all_reels, init_params, score_user_candidates, user_representation
from model.scoring import city_match_score, cosine_sim, temporal_decay_weights


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(n_users=20, n_reels=40, n_cities=6, batch_size=8, n_batches=1, epochs=1)
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
        batch = self.world.sample_bpr_batch(32)
        for uid, pos, neg in zip(batch["uid"], batch["pos_rid"], batch["neg_rid"]):
            self.assertIn(int(pos), set(self.world.positive_history(int(uid)).tolist()) or {int(pos)})
            self.assertNotIn(int(neg), set(self.world.known_positive_reels(int(uid)).tolist()))

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

    def test_zero_cosine_has_finite_gradient(self):
        a = jnp.zeros((1, 8), dtype=jnp.float32)
        b = jnp.ones((1, 8), dtype=jnp.float32)
        value = cosine_sim(a, b)
        grad = jax.grad(lambda x: jnp.sum(cosine_sim(x, b)))(a)
        self.assertTrue(np.isfinite(np.asarray(value)).all())
        self.assertTrue(np.isfinite(np.asarray(grad)).all())

    def test_graph_zero_input_zero_output_and_finite(self):
        h = np.zeros((4, 8), dtype=np.float32)
        nbrs = [[1], [0], [3], [2]]
        out = lightgcn_propagate(h, nbrs, 2)
        np.testing.assert_allclose(out, 0.0)
        self.assertTrue(np.isfinite(out).all())

    def test_privacy_budget_is_exposed(self):
        eps = rappor_epsilon_per_bit(0.5, 0.5, 0.75)
        self.assertGreater(eps, 0.0)
        self.assertAlmostEqual(rappor_vector_epsilon_upper_bound(64, 0.5, 0.5, 0.75), 64 * eps)

    def test_item_cold_start_uses_content_path(self):
        params = init_params(jax.random.PRNGKey(self.cfg.seed), self.cfg)
        ctx = build_ctx(self.world, self.world.g_all, self.cfg)
        content, probs = encode_all_reels(params, self.cfg, ctx)
        rid = 0
        # Zero the warm-item residual: content representation must remain non-zero and usable.
        params = dict(params)
        residual = params["reel_id_residual"].at[rid].set(0.0)
        params["reel_id_residual"] = residual
        scores, components, gate = score_user_candidates(params, self.cfg, ctx, 0, [rid])
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(np.isfinite(np.asarray(scores)).all())
        self.assertTrue(np.linalg.norm(np.asarray(content[rid])) > 0.0)
        self.assertTrue(np.isfinite(np.asarray(probs)).all())
        self.assertAlmostEqual(float(jnp.sum(gate)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
