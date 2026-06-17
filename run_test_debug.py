import numpy as np
from implementation.utils import build_delaunay_complex
from implementation.simplicial import DECOperatorEngine
from implementation.reservoir import SplitLeakChordESN

coords, edges, faces, hodge_0, hodge_1, hodge_2 = build_delaunay_complex(num_points=12, seed=123)
dec = DECOperatorEngine(len(coords), edges, faces, hodge_0, hodge_1, hodge_2, normalize=True)
dec.precompute_projectors()

esn = SplitLeakChordESN(
    simplicial_complex=dec,
    input_size=1,
    seed=42,
    T_proj=5,
    epsilon=1e-4,
    lambda_ex=0.6,
    lambda_co=0.4,
    lambda_ha=0.03
)

inputs = np.random.RandomState(99).randn(200, 1)

# We will test the two update equations directly in the script to see their norm behavior
# Equation 1: s_y = r + raw_update (additive)
# Equation 2: s_y = raw_update (direct)

W_res = esn.alpha * esn.W_x + esn.beta * esn.W_y + esn.gamma * esn.W_z

print("--- Testing Equation 1 (Additive s_y = r + raw_update) ---")
r = np.zeros(esn.N_e)
cached_y_ex = np.zeros(esn.N_e)
cached_y_co = np.zeros(esn.N_e)
cached_y_ha = np.zeros(esn.N_e)
dec = esn.simplicial_complex

for t in range(50):
    u_t = inputs[t]
    raw_update = np.tanh(W_res.dot(r) + esn.W_in @ u_t + esn.b)
    s_y = r + raw_update
    if t % esn.T_proj == 0:
        b_ex = dec.d0.T @ (dec.star_1 @ s_y)
        p, _ = spla.cg(esn.A_ex, b_ex, **CG_KWARGS)
        y_ex = dec.d0 @ p
        if esn.A_co is not None:
            b_co = dec.d1 @ s_y
            q, _ = spla.cg(esn.A_co, b_co, **CG_KWARGS)
            y_co = dec.star_1_inv @ (dec.d1.T @ q)
        else:
            y_co = np.zeros(esn.N_e)
        y_ha = s_y - y_ex - y_co
        cached_y_ex = y_ex.copy()
        cached_y_co = y_co.copy()
        cached_y_ha = y_ha.copy()
    else:
        y_ex = cached_y_ex
        y_co = cached_y_co
        y_ha = cached_y_ha
    r = s_y - esn.lambda_ex * y_ex - esn.lambda_co * y_co - esn.lambda_ha * y_ha
    if t % 10 == 0 or t < 5:
        print(f"t={t}: norm(r)={np.linalg.norm(r):.4f}")

print("\n--- Testing Equation 2 (Direct s_y = raw_update) ---")
r = np.zeros(esn.N_e)
cached_y_ex = np.zeros(esn.N_e)
cached_y_co = np.zeros(esn.N_e)
cached_y_ha = np.zeros(esn.N_e)

for t in range(50):
    u_t = inputs[t]
    s_y = np.tanh(W_res.dot(r) + esn.W_in @ u_t + esn.b)
    if t % esn.T_proj == 0:
        b_ex = dec.d0.T @ (dec.star_1 @ s_y)
        p, _ = spla.cg(esn.A_ex, b_ex, **CG_KWARGS)
        y_ex = dec.d0 @ p
        if esn.A_co is not None:
            b_co = dec.d1 @ s_y
            q, _ = spla.cg(esn.A_co, b_co, **CG_KWARGS)
            y_co = dec.star_1_inv @ (dec.d1.T @ q)
        else:
            y_co = np.zeros(esn.N_e)
        y_ha = s_y - y_ex - y_co
        cached_y_ex = y_ex.copy()
        cached_y_co = y_co.copy()
        cached_y_ha = y_ha.copy()
    else:
        y_ex = cached_y_ex
        y_co = cached_y_co
        y_ha = cached_y_ha
    # Apply split-leak and update state r as the leaked state
    # Wait, if s_y = raw_update, the state is leaky-integrated: r = (1 - lambda) * r + lambda * s_y?
    # Or r = s_y - lambda * y?
    r = s_y - esn.lambda_ex * y_ex - esn.lambda_co * y_co - esn.lambda_ha * y_ha
    if t % 10 == 0 or t < 5:
        print(f"t={t}: norm(r)={np.linalg.norm(r):.4f}")


