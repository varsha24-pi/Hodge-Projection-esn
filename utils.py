import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from implementation.simplicial import DECOperatorEngine

def build_delaunay_complex(num_points: int, coords: np.ndarray = None, seed: int = 42) -> tuple:
    """
    Builds a 2-simplicial complex from a 2D Delaunay triangulation.
    Computes exact barycentric Hodge star diagonals.
    
    Args:
        num_points: Number of random points to generate if coords is None.
        coords: Optional array of shape (num_points, 2) defining coordinates.
        seed: Random seed for point generation.
        
    Returns:
        coords: Coordinates array of shape (num_points, 2).
        edges: Oriented edges array of shape (N_e, 2).
        faces: Oriented faces array of shape (N_f, 3) (CCW oriented).
        hodge_0, hodge_1, hodge_2: Barycentric Hodge star diagonals.
    """
    if coords is None:
        random_state = np.random.RandomState(seed)
        coords = random_state.uniform(0.0, 1.0, size=(num_points, 2))
    else:
        coords = np.asarray(coords, dtype=np.float64)
        num_points = len(coords)
        
    # Perform Delaunay Triangulation
    tri = Delaunay(coords)
    simplices = tri.simplices.copy()
    
    # Orient faces counter-clockwise
    oriented_faces = []
    for f in simplices:
        p0, p1, p2 = coords[f[0]], coords[f[1]], coords[f[2]]
        # Signed area / cross product
        val = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
        if val < 0:
            # Swap to CCW orientation
            oriented_faces.append([f[0], f[2], f[1]])
        else:
            oriented_faces.append([f[0], f[1], f[2]])
    faces = np.array(oriented_faces, dtype=np.int32)
    N_f = len(faces)
    
    # Extract unique edges and build maps
    # Canonical edge key (u, v) where u < v
    canonical_edges = set()
    for face in faces:
        v0, v1, v2 = face
        canonical_edges.add((min(v0, v1), max(v0, v1)))
        canonical_edges.add((min(v1, v2), max(v1, v2)))
        canonical_edges.add((min(v2, v0), max(v2, v0)))
        
    edges = np.array(sorted(list(canonical_edges)), dtype=np.int32)
    N_e = len(edges)
    
    # Pre-map edges to index and find face-to-edge adjacencies
    edge_to_idx = {tuple(e): idx for idx, e in enumerate(edges)}
    
    # 1. Compute face areas and centroids
    face_areas = np.zeros(N_f)
    face_centroids = np.zeros((N_f, 2))
    for f_idx, face in enumerate(faces):
        p0, p1, p2 = coords[face[0]], coords[face[1]], coords[face[2]]
        area = 0.5 * np.abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        face_areas[f_idx] = max(area, 1e-12)
        face_centroids[f_idx] = (p0 + p1 + p2) / 3.0
        
    # 2. Compute hodge_0: Primal vertex volume is 1. Dual vertex volume is barycentric area.
    # Area of dual cell for vertex i is 1/3 of the sum of areas of all adjacent triangles
    vertex_dual_areas = np.zeros(num_points)
    for f_idx, face in enumerate(faces):
        for v in face:
            vertex_dual_areas[v] += face_areas[f_idx] / 3.0
    hodge_0 = vertex_dual_areas  # primal volume is 1
    
    # 3. Compute hodge_2: Primal face area, Dual face volume is 1
    hodge_2 = 1.0 / face_areas
    
    # 4. Compute hodge_1: Primal edge length, Dual edge length is centroid-to-centroid distance
    # For each edge, find the adjacent faces
    edge_adj_faces = [[] for _ in range(N_e)]
    for f_idx, face in enumerate(faces):
        v0, v1, v2 = face
        e1 = (min(v0, v1), max(v0, v1))
        e2 = (min(v1, v2), max(v1, v2))
        e3 = (min(v2, v0), max(v2, v0))
        edge_adj_faces[edge_to_idx[e1]].append(f_idx)
        edge_adj_faces[edge_to_idx[e2]].append(f_idx)
        edge_adj_faces[edge_to_idx[e3]].append(f_idx)
        
    hodge_1 = np.zeros(N_e)
    for e_idx, edge in enumerate(edges):
        u, v = edge
        primal_length = max(np.linalg.norm(coords[u] - coords[v]), 1e-12)
        midpoint = 0.5 * (coords[u] + coords[v])
        
        adj = edge_adj_faces[e_idx]
        if len(adj) == 2:
            # Shared edge: dual edge is the distance between the two face centroids
            c1 = face_centroids[adj[0]]
            c2 = face_centroids[adj[1]]
            dual_length = np.linalg.norm(c1 - c2)
        elif len(adj) == 1:
            # Boundary edge: dual edge is distance from centroid to midpoint of the edge
            c1 = face_centroids[adj[0]]
            dual_length = np.linalg.norm(c1 - midpoint)
        else:
            dual_length = 0.0
            
        hodge_1[e_idx] = dual_length / primal_length
        
    return coords, edges, faces, hodge_0, hodge_1, hodge_2

def generate_lorenz63(T: int, dt: float = 0.01, sigma: float = 10.0, beta: float = 8.0/3.0, rho: float = 28.0, seed: int = 42) -> np.ndarray:
    """Generates a trajectory of the chaotic Lorenz-63 attractor."""
    random_state = np.random.RandomState(seed)
    state = random_state.uniform(-10.0, 10.0, size=3)
    
    trajectory = np.zeros((T, 3))
    for t in range(T):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        state = state + dt * np.array([dx, dy, dz])
        trajectory[t] = state
        
    return trajectory

def generate_mackey_glass(T: int, tau: int = 17, beta: float = 0.2, gamma: float = 0.1, n: int = 10, dt: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generates the Mackey-Glass chaotic time series."""
    # Warmup length
    warmup = 1000
    total_len = T + warmup
    history_len = int(tau / dt)
    
    random_state = np.random.RandomState(seed)
    # Initialize history queue
    history = list(random_state.uniform(0.8, 1.2, size=history_len))
    
    x = np.zeros(total_len)
    x[:history_len] = history
    
    for t in range(history_len, total_len):
        x_tau = x[t - history_len]
        curr_x = x[t - 1]
        
        # Differential equation using Euler method
        dx = (beta * x_tau) / (1.0 + x_tau**n) - gamma * curr_x
        x[t] = curr_x + dt * dx
        
    return x[warmup:].reshape(-1, 1)
