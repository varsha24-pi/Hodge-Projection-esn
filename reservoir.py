import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import inspect

# Solve SciPy version compatibility for cg() tol/rtol parameter
_cg_sig = inspect.signature(spla.cg)
CG_KWARGS = {'rtol': 1e-6} if 'rtol' in _cg_sig.parameters else {'tol': 1e-6}

class ChordESN:
    def __init__(
        self,
        simplicial_complex,
        input_size: int,
        density: float = 0.1,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        projection_type: str = "solenoidal",  # 'gradient', 'curl', 'harmonic', 'solenoidal', 'none'
        projection_mode: str = "recurrent",    # 'recurrent', 'state', 'readout'
        regularization: float = 1e-6,
        include_input: bool = True,
        eta: float = 0.1,
        seed: int = 42
    ):
        """
        Hodge-Projected Echo-State Network (CHORD-ESN) Reservoir.
        
        Args:
            simplicial_complex: An instance of DECOperatorEngine.
            input_size: Dimension of the input signal.
            density: Network density for sparse random internal mixing loops.
            leaking_rate: Leaky integration rate (1.0 = no leakage).
            input_scaling: Scaling factor for input weights.
            projection_type: Type of Hodge projection to apply ('gradient', 'curl', 'harmonic', 'solenoidal', 'none').
            projection_mode: How/where the projection is applied ('recurrent', 'state', 'readout').
            regularization: Ridge regression regularization parameter.
            include_input: If True, concatenates input to states for readout training.
            eta: Protective margin for Gershgorin-friendly sufficient contractive bound.
            seed: Random seed for reproducibility.
        """
        self.simplicial_complex = simplicial_complex
        self.input_size = input_size
        self.density = density
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.projection_type = projection_type
        self.projection_mode = projection_mode
        self.regularization = regularization
        self.include_input = include_input
        self.eta = eta
        self.seed = seed
        
        # Dimensions
        self.N_e = self.simplicial_complex.N_e
        
        if self.N_e == 0:
            raise ValueError("The simplicial complex must contain at least one edge.")
            
        # Select projection function
        if self.projection_type == "gradient":
            self.project_fn = self.simplicial_complex.project_gradient
        elif self.projection_type == "curl":
            self.project_fn = self.simplicial_complex.project_curl
        elif self.projection_type == "harmonic":
            self.project_fn = self.simplicial_complex.project_harmonic
        elif self.projection_type == "solenoidal":
            self.project_fn = self.simplicial_complex.project_solenoidal
        else:
            self.project_fn = lambda u, **kwargs: u
            
        # 1. Instantiation and Stability Analysis
        self.configure_stability(eta=self.eta)
        
        # Initialize input weight matrix W_in (dense, uniform)
        random_state = np.random.RandomState(self.seed + 1)
        self.W_in = random_state.uniform(
            -self.input_scaling, self.input_scaling, size=(self.N_e, self.input_size)
        )
        # Initialize bias vector b (dense, uniform)
        self.b = random_state.uniform(-0.1, 0.1, size=(self.N_e,))
        
        # Readout weights (trained later)
        self.W_out = None

    def _create_sparse_random_matrix(self, N: int, density: float, random_state: np.random.RandomState) -> sp.csr_matrix:
        """Creates a sparse random matrix of shape N x N with standard normal elements."""
        if N == 0:
            return sp.csr_matrix((0, 0))
        # Use scipy.sparse.random with normal distribution
        res = sp.random(
            N, N, density=density, format='csr', random_state=random_state, data_rvs=random_state.randn
        )
        return res

    def _estimate_operator_norm(self, W: sp.csr_matrix, steps: int = 15) -> float:
        """Estimates the operator norm (largest singular value) of sparse matrix W using power iteration."""
        N = W.shape[1]
        WTW = W.T @ W
        
        # Use a fixed seed for norm estimation to ensure deterministic updates during parameter tuning
        random_state = np.random.RandomState(1337)
        v = random_state.randn(N)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return 0.0
        v = v / v_norm
        
        try:
            # Attempt to use scipy.sparse.linalg.eigs
            eigenvalues, _ = spla.eigs(WTW, k=1, which='LM', maxiter=steps, tol=1e-5)
            return np.sqrt(np.abs(eigenvalues[0]))
        except Exception:
            # Fallback to manual power iteration
            for _ in range(steps):
                v_next = WTW.dot(v)
                norm = np.linalg.norm(v_next)
                if norm == 0:
                    break
                v = v_next / norm
            val = np.dot(v, WTW.dot(v))
            return np.sqrt(max(0.0, val))

    def configure_stability(self, eta: float = 0.1):
        """
        Enforces block-Lipschitz matrix inequality check:
        max(RowSums(M)) < 1.0 - eta
        Rescales recurrent mixing scales and coupling gains programmatically.
        """
        N_e = self.N_e
        random_state = np.random.RandomState(self.seed)
        
        # 1. Pull sparse random internal mixing loops
        Wx_raw = self._create_sparse_random_matrix(N_e, self.density, random_state)
        Wy_raw = self._create_sparse_random_matrix(N_e, self.density, random_state)
        Wz_raw = self._create_sparse_random_matrix(N_e, self.density, random_state)
        
        # 2. Use 15 steps of power iteration to estimate operator norms
        nx = self._estimate_operator_norm(Wx_raw, steps=15)
        ny = self._estimate_operator_norm(Wy_raw, steps=15)
        nz = self._estimate_operator_norm(Wz_raw, steps=15)
        
        # Normalize matrices to unit norm
        self.Wx = Wx_raw / nx if nx > 0 else Wx_raw
        self.Wy = Wy_raw / ny if ny > 0 else Wy_raw
        self.Wz = Wz_raw / nz if nz > 0 else Wz_raw
        
        # Programmatically tune:
        # recurrent mixing scales (s_x, s_y, s_z in [0.4, 0.8])
        # coupling gains (alpha, beta, gamma in [0.05, 0.5])
        s_x = 0.8
        s_y = 0.8
        s_z = 0.8
        
        alpha = 0.3
        beta = 0.3
        gamma = 0.3
        
        # Adjust eta if leaking_rate <= eta, ensuring convergence
        adjusted_eta = eta
        if self.leaking_rate <= eta:
            adjusted_eta = self.leaking_rate / 2.0
            
        decay = 0.95
        # Grid search / decay loop until row-sum check passes
        for _ in range(200):
            # RowSum = (1 - leak) + leak * (alpha * s_x + beta * s_y + gamma * s_z)
            row_sum = (1.0 - self.leaking_rate) + self.leaking_rate * (alpha * s_x + beta * s_y + gamma * s_z)
            if row_sum < 1.0 - adjusted_eta:
                break
                
            # Decay values
            s_x = max(0.4, s_x * decay)
            s_y = max(0.4, s_y * decay)
            s_z = max(0.4, s_z * decay)
            
            alpha = max(0.05, alpha * decay)
            beta = max(0.05, beta * decay)
            gamma = max(0.05, gamma * decay)
            
            # If we reached minimum and it still fails, adjust eta further to prevent infinite loop
            if (s_x == 0.4 and s_y == 0.4 and s_z == 0.4 and
                alpha == 0.05 and beta == 0.05 and gamma == 0.05):
                adjusted_eta = adjusted_eta * 0.5
                
        # Store scaled values
        self.s_x = s_x
        self.s_y = s_y
        self.s_z = s_z
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Store scaled matrices
        self.W_x = self.Wx * self.s_x
        self.W_y = self.Wy * self.s_y
        self.W_z = self.Wz * self.s_z
        
        # Precompute effective recurrent matrix W_res_effective for Recurrent Mode
        # W_res = alpha * W_x * P_grad + beta * W_y * P_curl + gamma * W_z * P_harm
        if hasattr(self.simplicial_complex, 'P_grad_mat'):
            self.W_res_effective = (
                self.alpha * self.W_x.toarray() @ self.simplicial_complex.P_grad_mat +
                self.beta * self.W_y.toarray() @ self.simplicial_complex.P_curl_mat +
                self.gamma * self.W_z.toarray() @ self.simplicial_complex.P_harm_mat
            )
        else:
            self.W_res_effective = None

    def forward(self, inputs: np.ndarray, initial_state: np.ndarray = None, washout: int = 0) -> tuple:
        """
        Runs the reservoir simulation over the input sequence.
        
        Args:
            inputs: input array of shape (T, D_in)
            initial_state: optional array of shape (N_e,) representing initial state
            washout: number of initial steps to discard
            
        Returns:
            states: array of shape (T - washout, N_e) representing reservoir states
            last_state: final state of shape (N_e,)
        """
        T, D_in = inputs.shape
        if D_in != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {D_in}.")
            
        # Initialize reservoir state
        if initial_state is None:
            r = np.zeros(self.N_e)
        else:
            r = np.asarray(initial_state, dtype=np.float64)
            if len(r) != self.N_e:
                raise ValueError(f"initial_state must have length equal to N_e ({self.N_e}).")
                
        all_states = []
        
        # Choose update mechanism based on projection mode
        a = self.leaking_rate
        
        if self.projection_mode == "recurrent":
            # Recurrent Projection Mode
            if self.W_res_effective is not None:
                for t in range(T):
                    u_t = inputs[t]
                    # Update state using precomputed W_res_effective
                    r = (1.0 - a) * r + a * np.tanh(self.W_res_effective @ r + self.W_in @ u_t + self.b)
                    if t >= washout:
                        all_states.append(r.copy())
            else:
                for t in range(T):
                    u_t = inputs[t]
                    # Compute projections iteratively
                    u_grad = self.simplicial_complex.project_gradient(r, use_iterative=True)
                    u_curl = self.simplicial_complex.project_curl(r, use_iterative=True)
                    u_harm = self.simplicial_complex.project_harmonic(r, use_iterative=True)
                    
                    recurrent_term = (
                        self.alpha * self.W_x.dot(u_grad) +
                        self.beta * self.W_y.dot(u_curl) +
                        self.gamma * self.W_z.dot(u_harm)
                    )
                    r = (1.0 - a) * r + a * np.tanh(recurrent_term + self.W_in @ u_t + self.b)
                    if t >= washout:
                        all_states.append(r.copy())
                        
        elif self.projection_mode == "state":
            # State Projection Mode: project updated state at each step
            # Recurrent weights are a sum of the normalized mixing matrices
            W_res = self.alpha * self.W_x + self.beta * self.W_y + self.gamma * self.W_z
            for t in range(T):
                u_t = inputs[t]
                r_tilde = (1.0 - a) * r + a * np.tanh(W_res.dot(r) + self.W_in @ u_t + self.b)
                r = self.project_fn(r_tilde)
                if t >= washout:
                    all_states.append(r.copy())
                    
        elif self.projection_mode == "readout":
            # Readout Projection Mode: normal recurrent update, project state for output only
            W_res = self.alpha * self.W_x + self.beta * self.W_y + self.gamma * self.W_z
            for t in range(T):
                u_t = inputs[t]
                r = (1.0 - a) * r + a * np.tanh(W_res.dot(r) + self.W_in @ u_t + self.b)
                if t >= washout:
                    # Apply projection only to the collected readout state
                    r_readout = self.project_fn(r)
                    all_states.append(r_readout.copy())
                    
        return np.array(all_states), r

    def fit(self, inputs: np.ndarray, targets: np.ndarray, washout: int = 0):
        """
        Trains the readout weights W_out using Ridge Regression.
        
        Args:
            inputs: input sequence of shape (T, D_in)
            targets: target sequence of shape (T, D_out)
            washout: number of initial steps to discard
        """
        states, _ = self.forward(inputs, washout=washout)
        targets_cut = targets[washout:]
        
        if self.include_input:
            inputs_cut = inputs[washout:]
            R = np.hstack([states, inputs_cut, np.ones((len(states), 1))])
        else:
            R = np.hstack([states, np.ones((len(states), 1))])
            
        N_features = R.shape[1]
        I_reg = np.eye(N_features)
        I_reg[-1, -1] = 0.0  # Do not regularize the bias term
        
        # W_out = Y^T @ R @ (R^T @ R + regularization * I)^-1
        self.W_out = targets_cut.T @ R @ la.pinv(R.T @ R + self.regularization * I_reg)

    def predict(self, inputs: np.ndarray, initial_state: np.ndarray = None) -> np.ndarray:
        """
        Predicts the outputs for the given inputs using the trained W_out.
        
        Args:
            inputs: input sequence of shape (T, D_in)
            initial_state: optional initial state of shape (N_e,)
            
        Returns:
            predictions: predicted outputs of shape (T, D_out)
        """
        if self.W_out is None:
            raise ValueError("Model must be fitted before making predictions.")
            
        states, _ = self.forward(inputs, initial_state=initial_state, washout=0)
        
        if self.include_input:
            R = np.hstack([states, inputs, np.ones((len(states), 1))])
        else:
            R = np.hstack([states, np.ones((len(states), 1))])
            
        return R @ self.W_out.T

class SplitLeakChordESN(ChordESN):
    def __init__(
        self,
        simplicial_complex,
        input_size: int,
        density: float = 0.1,
        input_scaling: float = 1.0,
        regularization: float = 1e-6,
        include_input: bool = True,
        eta: float = 0.1,
        seed: int = 42,
        T_proj: int = 10,
        epsilon: float = 1e-4,
        lambda_ex: float = 0.5,
        lambda_co: float = 0.3,
        lambda_ha: float = 0.05
    ):
        """
        Split-Leak CHORD-ESN with intermittent Hodge projections.
        
        Args:
            simplicial_complex: An instance of DECOperatorEngine.
            input_size: Dimension of the input signal.
            density: Network density for sparse random internal mixing loops.
            input_scaling: Scaling factor for input weights.
            regularization: Ridge regression regularization parameter.
            include_input: If True, concatenates input to states for readout training.
            eta: Protective margin for Gershgorin-friendly sufficient contractive bound.
            seed: Random seed for reproducibility.
            T_proj: Cadence/window size for projection steps (evaluate projection every T_proj steps).
            epsilon: Tikhonov conditioning stabilizer for solver.
            lambda_ex: Decay rate for exact (gradient) flow component.
            lambda_co: Decay rate for coexact (curl) flow component.
            lambda_ha: Decay rate for harmonic cycle residue component (topology-anchored channel).
        """
        self.T_proj = T_proj
        self.epsilon = epsilon
        self.lambda_ex = lambda_ex
        self.lambda_co = lambda_co
        self.lambda_ha = lambda_ha
        
        # Call parent constructor (leaking_rate is set to 1.0, handled by split-leak instead)
        super().__init__(
            simplicial_complex=simplicial_complex,
            input_size=input_size,
            density=density,
            leaking_rate=1.0,
            input_scaling=input_scaling,
            projection_type="none",  # custom projection handled inside forward loop
            projection_mode="recurrent",
            regularization=regularization,
            include_input=include_input,
            eta=eta,
            seed=seed
        )
        
        # Precompute the time-invariant SPD matrices for the projection solvers
        dec = self.simplicial_complex
        
        # A_ex = d0.T @ star_1 @ d0 + epsilon * star_0
        self.A_ex = (dec.d0.T @ dec.star_1 @ dec.d0 + self.epsilon * dec.star_0).tocsr()
        
        # A_co = d1 @ star_1_inv @ d1.T + epsilon * star_2
        if dec.N_f > 0:
            self.A_co = (dec.d1 @ dec.star_1_inv @ dec.d1.T + self.epsilon * dec.star_2).tocsr()
        else:
            self.A_co = None

    def forward(self, inputs: np.ndarray, initial_state: np.ndarray = None, washout: int = 0) -> tuple:
        """
        Runs the reservoir simulation over the input sequence using the Split-Leak mechanical core.
        
        Args:
            inputs: input array of shape (T, D_in)
            initial_state: optional array of shape (N_e,) representing initial state
            washout: number of initial steps to discard
            
        Returns:
            states: array of shape (T - washout, N_e) representing reservoir states
            last_state: final state of shape (N_e,)
        """
        T, D_in = inputs.shape
        if D_in != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {D_in}.")
            
        if initial_state is None:
            r = np.zeros(self.N_e)
        else:
            r = np.asarray(initial_state, dtype=np.float64)
            if len(r) != self.N_e:
                raise ValueError(f"initial_state must have length equal to N_e ({self.N_e}).")
                
        all_states = []
        
        # Recurrent weight matrix (combination of scaled mixing loops)
        W_res = self.alpha * self.W_x + self.beta * self.W_y + self.gamma * self.W_z
        
        # Caching variables for the projection step
        cached_y_ex = np.zeros(self.N_e)
        cached_y_co = np.zeros(self.N_e)
        cached_y_ha = np.zeros(self.N_e)
        
        dec = self.simplicial_complex
        
        for t in range(T):
            u_t = inputs[t]
            
            # 1. State update transition (mechanical core)
            raw_update = np.tanh(W_res.dot(r) + self.W_in @ u_t + self.b)
            s_y = r + raw_update
            
            # 2. Intermittent Hodge decomposition
            if t % self.T_proj == 0:
                # Solve exact projection problem: A_ex p = b_ex
                b_ex = dec.d0.T @ (dec.star_1 @ s_y)
                p, _ = spla.cg(self.A_ex, b_ex, **CG_KWARGS)
                y_ex = dec.d0 @ p
                
                # Solve dual coexact projection problem: A_co q = b_co
                if self.A_co is not None and dec.N_f > 0:
                    b_co = dec.d1 @ s_y
                    q, _ = spla.cg(self.A_co, b_co, **CG_KWARGS)
                    y_co = dec.star_1_inv @ (dec.d1.T @ q)
                else:
                    y_co = np.zeros(self.N_e)
                    
                # Extract harmonic cycle residue
                y_ha = s_y - y_ex - y_co
                
                # Cache components
                cached_y_ex = y_ex.copy()
                cached_y_co = y_co.copy()
                cached_y_ha = y_ha.copy()
            else:
                # Bypass solver on intermediate steps, reuse cached components
                y_ex = cached_y_ex
                y_co = cached_y_co
                y_ha = cached_y_ha
                
            # 4. Apply split-leak equation
            r = s_y - self.lambda_ex * y_ex - self.lambda_co * y_co - self.lambda_ha * y_ha
            
            if t >= washout:
                all_states.append(r.copy())
                
        return np.array(all_states), r

