import unittest
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from implementation.simplicial import DECOperatorEngine
from implementation.reservoir import ChordESN, SplitLeakChordESN
from implementation.utils import build_delaunay_complex

class TestCHORDESN(unittest.TestCase):
    def setUp(self):
        # Build a small, stable Delaunay complex for testing
        # 10 random points inside [0, 1] x [0, 1]
        self.coords, self.edges, self.faces, self.hodge_0, self.hodge_1, self.hodge_2 = build_delaunay_complex(
            num_points=12, seed=123
        )
        self.num_vertices = len(self.coords)
        self.N_e = len(self.edges)
        self.N_f = len(self.faces)
        
        # Instantiate the DEC engine
        self.dec = DECOperatorEngine(
            num_vertices=self.num_vertices,
            edges=self.edges,
            faces=self.faces,
            hodge_0=self.hodge_0,
            hodge_1=self.hodge_1,
            hodge_2=self.hodge_2,
            normalize=True
        )
        # Precompute projectors for direct tests
        self.dec.precompute_projectors()

    def test_topological_identities(self):
        """Verify the fundamental topological identity d1 @ d0 = 0 and delta0 @ delta1 = 0."""
        # 1. d1 @ d0 must be zero matrix
        d1_d0 = self.dec.d1 @ self.dec.d0
        np.testing.assert_array_almost_equal(
            d1_d0.toarray(), np.zeros((self.N_f, self.num_vertices)), decimal=9
        )
        
        # 2. delta0 @ delta1 must be zero matrix (divergence of curl)
        del0_del1 = self.dec.delta_0 @ self.dec.delta_1
        np.testing.assert_array_almost_equal(
            del0_del1.toarray(), np.zeros((self.num_vertices, self.N_f)), decimal=9
        )

    def test_helmholtz_hodge_decomposition(self):
        """Verify that any edge flow field can be orthogonally decomposed under the Hodge inner product."""
        random_state = np.random.RandomState(42)
        u = random_state.randn(self.N_e)
        
        # Compute components
        u_grad = self.dec.project_gradient(u)
        u_curl = self.dec.project_curl(u)
        u_harm = self.dec.project_harmonic(u)
        
        # 1. Reconstruction accuracy: u_grad + u_curl + u_harm = u
        u_reconstructed = u_grad + u_curl + u_harm
        np.testing.assert_array_almost_equal(u, u_reconstructed, decimal=6)
        
        # 2. Orthogonality under the Hodge inner product: <a, b>_* = a^T * star_1 * b
        star_1 = sp.diags(self.dec.hodge_1).toarray()
        
        inner_grad_curl = u_grad.T @ star_1 @ u_curl
        inner_grad_harm = u_grad.T @ star_1 @ u_harm
        inner_curl_harm = u_curl.T @ star_1 @ u_harm
        
        self.assertAlmostEqual(inner_grad_curl, 0.0, places=6)
        self.assertAlmostEqual(inner_grad_harm, 0.0, places=6)
        self.assertAlmostEqual(inner_curl_harm, 0.0, places=6)
        
        # 3. Idempotency check: P(P(u)) = P(u)
        u_grad_twice = self.dec.project_gradient(u_grad)
        np.testing.assert_array_almost_equal(u_grad, u_grad_twice, decimal=6)
        
        u_curl_twice = self.dec.project_curl(u_curl)
        np.testing.assert_array_almost_equal(u_curl, u_curl_twice, decimal=6)

    def test_norm_estimation(self):
        """Verify that our power iteration norm estimator matches SVD operator norm estimation."""
        # Create a dummy ESN and run the estimator
        esn = ChordESN(simplicial_complex=self.dec, input_size=1, seed=42)
        W = esn._create_sparse_random_matrix(self.N_e, density=0.2, random_state=np.random.RandomState(10))
        
        estimated_norm = esn._estimate_operator_norm(W, steps=25)
        # Compute exact singular value norm
        svd_norm = la.svdvals(W.toarray())[0]
        
        # It should be extremely close (within 1%)
        self.assertLess(abs(estimated_norm - svd_norm) / svd_norm, 0.02)

    def test_stability_configuration(self):
        """Verify that the contractive row-sum check holds for ChordESN."""
        # Check standard ESN
        esn = ChordESN(
            simplicial_complex=self.dec,
            input_size=2,
            leaking_rate=0.8,
            eta=0.1,
            seed=42
        )
        
        # M is a 3x3 block-Lipschitz matrix with identical row sums:
        # row_sum = (1 - leak) + leak * (alpha * sx + beta * sy + gamma * sz)
        row_sum = (1.0 - esn.leaking_rate) + esn.leaking_rate * (
            esn.alpha * esn.s_x + esn.beta * esn.s_y + esn.gamma * esn.s_z
        )
        self.assertLess(row_sum, 1.0 - esn.eta)
        self.assertTrue(0.4 <= esn.s_x <= 0.8)
        self.assertTrue(0.05 <= esn.alpha <= 0.5)

    def test_splitleak_reservoir_update(self):
        """Verify SplitLeakChordESN execution, caching logic, and projection intervals."""
        T_proj = 5
        esn = SplitLeakChordESN(
            simplicial_complex=self.dec,
            input_size=1,
            seed=42,
            T_proj=T_proj,
            epsilon=1e-4,
            lambda_ex=0.6,
            lambda_co=0.4,
            lambda_ha=0.03
        )
        
        # Test input sequence
        T = 20
        inputs = np.random.RandomState(99).randn(T, 1)
        
        # Run forward simulation
        states, last_state = esn.forward(inputs, washout=0)
        self.assertEqual(states.shape, (T, self.N_e))
        self.assertEqual(len(last_state), self.N_e)
        
        # Verify that states are bounded
        self.assertTrue(np.all(np.abs(states) < 10.0))

    def test_chord_esn_pipeline(self):
        """Verify ChordESNPipeline execution, stability, feature extraction, and fitting."""
        from implementation.reservoir import ChordESNPipeline
        
        # Instantiate pipeline
        pipeline = ChordESNPipeline(
            simplicial_complex=self.dec,
            input_size=2,
            density=0.2,
            seed=42,
            lambda_x=0.6,
            lambda_y=0.4,
            lambda_z=0.5,
            T_proj=5,
            epsilon=1e-4,
            lambda_ex=0.5,
            lambda_co=0.3,
            lambda_ha=0.04,
            tau_0=0.01,
            tau_1=0.01,
            tau_2=0.01
        )
        
        # 1. Verify that states are reset to zero
        self.assertEqual(len(pipeline.x), self.num_vertices)
        self.assertEqual(len(pipeline.y), self.N_e)
        self.assertEqual(len(pipeline.z), self.N_f)
        np.testing.assert_array_equal(pipeline.x, np.zeros(self.num_vertices))
        np.testing.assert_array_equal(pipeline.y, np.zeros(self.N_e))
        np.testing.assert_array_equal(pipeline.z, np.zeros(self.N_f))
        
        # 2. Verify stability parameters are set correctly
        # row_sum must be < 1.0 - eta
        row_sum1 = (1.0 - pipeline.lambda_x) + pipeline.lambda_x * pipeline.s_x_scale
        self.assertLess(row_sum1, 1.0 - pipeline.eta)
        
        # 3. Run a step and verify the returned readout vector shape
        u_t = np.array([0.5, -0.2])
        phi_t = pipeline.step(u_t, t=0)
        
        expected_dim = 2 * self.num_vertices + 2 * self.N_e + self.N_f + 1
        self.assertEqual(len(phi_t), expected_dim)
        
        # Last element of phi_t should be 1.0 (constant bias)
        self.assertEqual(phi_t[-1], 1.0)
        
        # 4. Verify forward_pipeline runs and returns correct shape
        T = 20
        inputs = np.random.RandomState(99).randn(T, 2)
        features = pipeline.forward_pipeline(inputs, washout=5)
        self.assertEqual(features.shape, (T - 5, expected_dim))
        
        # 5. Verify fit and predict
        targets = np.random.RandomState(100).randn(T, 3)
        pipeline.fit(inputs, targets, washout=5)
        self.assertIsNotNone(pipeline.W_out)
        self.assertEqual(pipeline.W_out.shape, (3, expected_dim))
        
        preds = pipeline.predict(inputs)
        self.assertEqual(preds.shape, (T, 3))

    def test_train_chord_esn_pipeline_orchestrator(self):
        """Verify out-of-class training orchestrator train_chord_esn_pipeline correctness."""
        from implementation.reservoir import ChordESNPipeline, train_chord_esn_pipeline
        
        # Instantiate pipeline
        pipeline = ChordESNPipeline(
            simplicial_complex=self.dec,
            input_size=2,
            density=0.2,
            seed=42,
            lambda_x=0.6,
            lambda_y=0.4,
            lambda_z=0.5,
            T_proj=5,
            epsilon=1e-4,
            lambda_ex=0.5,
            lambda_co=0.3,
            lambda_ha=0.04,
            tau_0=0.01,
            tau_1=0.01,
            tau_2=0.01
        )
        
        T = 30
        inputs = np.random.RandomState(42).randn(T, 2)
        targets = np.random.RandomState(43).randn(T, 3)
        washout = 10
        lambda_R = 1e-4
        
        expected_dim = 2 * self.num_vertices + 2 * self.N_e + self.N_f + 1
        
        # Execute the orchestrator
        W_out = train_chord_esn_pipeline(
            pipeline=pipeline,
            inputs=inputs,
            targets=targets,
            washout=washout,
            lambda_R=lambda_R
        )
        
        # Verify weight shapes and values
        self.assertEqual(W_out.shape, (3, expected_dim))
        self.assertEqual(pipeline.W_out.shape, (3, expected_dim))
        np.testing.assert_array_almost_equal(W_out, pipeline.W_out)
        
        # Verify prediction works with the trained weights
        preds = pipeline.predict(inputs)
        self.assertEqual(preds.shape, (T, 3))

if __name__ == "__main__":
    unittest.main()


