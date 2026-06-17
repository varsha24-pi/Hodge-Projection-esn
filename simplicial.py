import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import inspect

# Solve SciPy version compatibility for cg() tol/rtol parameter
_cg_sig = inspect.signature(spla.cg)
CG_KWARGS = {'rtol': 1e-6} if 'rtol' in _cg_sig.parameters else {'tol': 1e-6}

class DECOperatorEngine:
    def __init__(
        self,
        num_vertices: int,
        edges: np.ndarray,
        faces: np.ndarray,
        hodge_0: np.ndarray = None,
        hodge_1: np.ndarray = None,
        hodge_2: np.ndarray = None,
        normalize: bool = True
    ):
        """
        Initializes the structure-preserving discrete differential operators on an oriented 2-simplicial complex.
        
        Args:
            num_vertices: Number of vertices (0-simplices)
            edges: numpy array of shape (N_e, 2) representing oriented edges (1-simplices)
            faces: numpy array of shape (N_f, 3) representing oriented faces (2-simplices, triangles)
            hodge_0: optional 1D array of length N_v representing primal/dual volume ratios for vertices
            hodge_1: optional 1D array of length N_e representing primal/dual volume ratios for edges
            hodge_2: optional 1D array of length N_f representing primal/dual volume ratios for faces
            normalize: if True, normalizes the Hodge star diagonals so operator norms match O(1) scaling
        """
        self.num_vertices = num_vertices
        self.edges = np.asarray(edges, dtype=np.int32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.N_e = len(self.edges)
        self.N_f = len(self.faces)
        
        # Inputs validation
        if self.N_e > 0 and (self.edges.ndim != 2 or self.edges.shape[1] != 2):
            raise ValueError("edges must be a numpy array of shape (N_e, 2).")
        if self.N_f > 0 and (self.faces.ndim != 2 or self.faces.shape[1] != 3):
            raise ValueError("faces must be a numpy array of shape (N_f, 3).")
            
        # Default Hodge star diagonals if not provided
        if hodge_0 is None:
            hodge_0 = np.ones(self.num_vertices, dtype=np.float64)
        else:
            hodge_0 = np.asarray(hodge_0, dtype=np.float64)
            if len(hodge_0) != self.num_vertices:
                raise ValueError(f"hodge_0 must have length equal to num_vertices ({self.num_vertices}).")
            
        if hodge_1 is None:
            hodge_1 = np.ones(self.N_e, dtype=np.float64)
        else:
            hodge_1 = np.asarray(hodge_1, dtype=np.float64)
            if len(hodge_1) != self.N_e:
                raise ValueError(f"hodge_1 must have length equal to the number of edges ({self.N_e}).")
            
        if hodge_2 is None:
            hodge_2 = np.ones(self.N_f, dtype=np.float64)
        else:
            hodge_2 = np.asarray(hodge_2, dtype=np.float64)
            if len(hodge_2) != self.N_f:
                raise ValueError(f"hodge_2 must have length equal to the number of faces ({self.N_f}).")
                
        # Validate that all Hodge diagonal elements are strictly positive
        if np.any(hodge_0 <= 0):
            raise ValueError("hodge_0 must contain strictly positive values.")
        if np.any(hodge_1 <= 0):
            raise ValueError("hodge_1 must contain strictly positive values.")
        if np.any(hodge_2 <= 0):
            raise ValueError("hodge_2 must contain strictly positive values.")
            
        # 2. Diagonal Hodge Stars Normalization
        if normalize:
            hodge_0 = hodge_0 / np.mean(hodge_0)
            hodge_1 = hodge_1 / np.mean(hodge_1)
            if self.N_f > 0:
                hodge_2 = hodge_2 / np.mean(hodge_2)
            
        self.hodge_0 = hodge_0
        self.hodge_1 = hodge_1
        self.hodge_2 = hodge_2
        
        # 1. Coboundary Matrices (d_0, d_1)
        # Construct d0 (exterior derivative on 0-forms, shape (N_e, N_v))
        if self.N_e > 0:
            row0 = np.repeat(np.arange(self.N_e), 2)
            col0 = self.edges.ravel()
            data0 = np.tile([-1.0, 1.0], self.N_e)
            self.d0 = sp.coo_matrix((data0, (row0, col0)), shape=(self.N_e, self.num_vertices)).tocsr()
        else:
            self.d0 = sp.csr_matrix((0, self.num_vertices))
            
        # Construct d1 (exterior derivative on 1-forms, shape (N_f, N_e))
        if self.N_f > 0 and self.N_e > 0:
            max_v = max(self.num_vertices - 1, self.edges.max(), self.faces.max())
            multiplier = int(max_v + 1)
            # Create unique hash keys for oriented edges to look up their index
            edge_hash = self.edges[:, 0] * multiplier + self.edges[:, 1]
            edge_to_idx = {h: idx for idx, h in enumerate(edge_hash)}
            
            row1 = []
            col1 = []
            data1 = []
            
            v0, v1, v2 = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
            segments = [(v0, v1), (v1, v2), (v2, v0)]
            
            for u, v in segments:
                fwd_hashes = u * multiplier + v
                rev_hashes = v * multiplier + u
                
                for f_idx, (fwd, rev) in enumerate(zip(fwd_hashes, rev_hashes)):
                    if fwd in edge_to_idx:
                        row1.append(f_idx)
                        col1.append(edge_to_idx[fwd])
                        data1.append(1.0)
                    elif rev in edge_to_idx:
                        row1.append(f_idx)
                        col1.append(edge_to_idx[rev])
                        data1.append(-1.0)
                    else:
                        raise ValueError(
                            f"Boundary edge ({u[f_idx]} -> {v[f_idx]}) of face {f_idx} is not in the edges list."
                        )
            self.d1 = sp.coo_matrix((data1, (row1, col1)), shape=(self.N_f, self.N_e)).tocsr()
        else:
            self.d1 = sp.csr_matrix((self.N_f, self.N_e))
            
        # Materialize Hodge star matrices
        self.star_0 = sp.diags(self.hodge_0).tocsr()
        self.star_1 = sp.diags(self.hodge_1).tocsr()
        self.star_2 = sp.diags(self.hodge_2).tocsr()
        
        self.star_0_inv = sp.diags(1.0 / self.hodge_0).tocsr()
        self.star_1_inv = sp.diags(1.0 / self.hodge_1).tocsr()
        self.star_2_inv = sp.diags(1.0 / self.hodge_2).tocsr()
        
        # 3. Codifferentials (delta_0, delta_1)
        self.delta_0 = (self.star_0_inv @ self.d0.T @ self.star_1).tocsr()
        self.delta_1 = (self.star_1_inv @ self.d1.T @ self.star_2).tocsr()
        
        # 4. Laplace-de Rham Operators (L_0, L_1, L_2)
        self.L0 = (self.delta_0 @ self.d0).tocsr()
        self.L1 = (self.d0 @ self.delta_0 + self.delta_1 @ self.d1).tocsr()
        self.L2 = (self.d1 @ self.delta_1).tocsr()
        
    def precompute_projectors(self):
        """
        Precomputes the dense projection matrices for gradient, curl, and harmonic spaces.
        This is extremely fast for simulations since it avoids solving linear systems at each time step.
        """
        import scipy.linalg as la
        
        N_e = self.N_e
        N_v = self.num_vertices
        N_f = self.N_f
        
        I_e = np.eye(N_e)
        
        # P_grad = d0 @ L0^+ @ delta_0
        if N_v > 0 and self.L0.nnz > 0:
            L0_dense = self.L0.toarray()
            L0_pinv = la.pinv(L0_dense)
            self.P_grad_mat = self.d0.toarray() @ L0_pinv @ self.delta_0.toarray()
        else:
            self.P_grad_mat = np.zeros((N_e, N_e))
            
        # P_curl = delta_1 @ L2^+ @ d1
        if N_f > 0 and self.L2.nnz > 0:
            L2_dense = self.L2.toarray()
            L2_pinv = la.pinv(L2_dense)
            self.P_curl_mat = self.delta_1.toarray() @ L2_pinv @ self.d1.toarray()
        else:
            self.P_curl_mat = np.zeros((N_e, N_e))
            
        # P_harm = I - P_grad - P_curl
        self.P_harm_mat = I_e - self.P_grad_mat - self.P_curl_mat
        self.P_sol_mat = self.P_curl_mat + self.P_harm_mat

    def project_gradient(self, u: np.ndarray, use_iterative: bool = False) -> np.ndarray:
        """
        Projects an edge flow u onto the gradient (curl-free) space.
        u can be a 1D array of shape (N_e,) or a 2D array of shape (N_e, B).
        """
        if self.N_e == 0:
            return u
        if not use_iterative and hasattr(self, 'P_grad_mat'):
            return self.P_grad_mat @ u
            
        u_arr = np.asarray(u)
        if u_arr.ndim == 1:
            rhs = self.delta_0 @ u_arr
            phi, _ = spla.cg(self.L0, rhs, **CG_KWARGS)
            return self.d0 @ phi
        else:
            res = np.zeros_like(u_arr)
            for b in range(u_arr.shape[1]):
                rhs = self.delta_0 @ u_arr[:, b]
                phi, _ = spla.cg(self.L0, rhs, **CG_KWARGS)
                res[:, b] = self.d0 @ phi
            return res
            
    def project_curl(self, u: np.ndarray, use_iterative: bool = False) -> np.ndarray:
        """
        Projects an edge flow u onto the curl (divergence-free) space.
        """
        if self.N_e == 0 or self.N_f == 0:
            return np.zeros_like(u)
        if not use_iterative and hasattr(self, 'P_curl_mat'):
            return self.P_curl_mat @ u
            
        u_arr = np.asarray(u)
        if u_arr.ndim == 1:
            rhs = self.d1 @ u_arr
            psi, _ = spla.cg(self.L2, rhs, **CG_KWARGS)
            return self.delta_1 @ psi
        else:
            res = np.zeros_like(u_arr)
            for b in range(u_arr.shape[1]):
                rhs = self.d1 @ u_arr[:, b]
                psi, _ = spla.cg(self.L2, rhs, **CG_KWARGS)
                res[:, b] = self.delta_1 @ psi
            return res
            
    def project_harmonic(self, u: np.ndarray, use_iterative: bool = False) -> np.ndarray:
        """
        Projects an edge flow u onto the harmonic space (both curl-free and divergence-free).
        """
        if self.N_e == 0:
            return u
        if not use_iterative and hasattr(self, 'P_harm_mat'):
            return self.P_harm_mat @ u
            
        u_grad = self.project_gradient(u, use_iterative=use_iterative)
        u_curl = self.project_curl(u, use_iterative=use_iterative)
        return u - u_grad - u_curl
        
    def project_solenoidal(self, u: np.ndarray, use_iterative: bool = False) -> np.ndarray:
        """
        Projects an edge flow u onto the solenoidal space (curl + harmonic = divergence-free).
        """
        if self.N_e == 0:
            return u
        if not use_iterative and hasattr(self, 'P_sol_mat'):
            return self.P_sol_mat @ u
            
        u_grad = self.project_gradient(u, use_iterative=use_iterative)
        return u - u_grad

