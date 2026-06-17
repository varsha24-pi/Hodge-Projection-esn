import numpy as np
import time
from implementation.utils import build_delaunay_complex, generate_lorenz63
from implementation.simplicial import DECOperatorEngine
from implementation.reservoir import ChordESN, SplitLeakChordESN, ChordESNPipeline

def run_benchmarks():
    print("======================================================================")
    # 1. Setup Simplical Complex
    print("Initializing oriented 2-simplicial complex...")
    coords, edges, faces, hodge_0, hodge_1, hodge_2 = build_delaunay_complex(
        num_points=100, seed=42
    )
    N_v = len(coords)
    N_e = len(edges)
    N_f = len(faces)
    print(f"Mesh stats: Vertices={N_v}, Edges={N_e}, Faces={N_f}")
    
    # Instantiate the DEC Operator Engine
    dec = DECOperatorEngine(N_v, edges, faces, hodge_0, hodge_1, hodge_2, normalize=True)
    dec.precompute_projectors()
    print("Topological structures and projection operators precomputed successfully.")
    
    # 2. Generate chaotic dataset: Lorenz-63 Attractor
    print("\nGenerating Lorenz-63 chaotic trajectory...")
    T_total = 3000
    data = generate_lorenz63(T_total, dt=0.01, seed=42)
    # We will predict the next step in the trajectory
    # Inputs: data[t], Targets: data[t+1]
    inputs = data[:-1]
    targets = data[1:]
    
    # Split into train/validation/test sets
    washout = 200
    train_len = 1800
    test_len = 900
    
    tr_inputs = inputs[:train_len]
    tr_targets = targets[:train_len]
    
    ts_inputs = inputs[train_len:train_len + test_len]
    ts_targets = targets[train_len:train_len + test_len]
    
    print(f"Dataset split: Train length={train_len} (washout={washout}), Test length={test_len}")
    
    # 3. Model configurations to evaluate
    models = {
        "Standard ESN (No Projection)": ChordESN(
            simplicial_complex=dec,
            input_size=3,
            leaking_rate=0.8,
            projection_type="none",
            projection_mode="recurrent",
            regularization=1e-6,
            seed=100
        ),
        "CHORD-ESN (Solenoidal Projection)": ChordESN(
            simplicial_complex=dec,
            input_size=3,
            leaking_rate=0.8,
            projection_type="solenoidal",
            projection_mode="recurrent",
            regularization=1e-6,
            seed=100
        ),
        "Split-Leak CHORD-ESN (T_proj=5)": SplitLeakChordESN(
            simplicial_complex=dec,
            input_size=3,
            seed=100,
            T_proj=5,
            epsilon=1e-4,
            lambda_ex=0.5,
            lambda_co=0.3,
            lambda_ha=0.04,
            regularization=1e-6
        ),
        "ChordESN execution pipeline": ChordESNPipeline(
            simplicial_complex=dec,
            input_size=3,
            seed=100,
            lambda_x=0.5,
            lambda_y=0.5,
            lambda_z=0.5,
            T_proj=5,
            epsilon=1e-4,
            lambda_ex=0.5,
            lambda_co=0.3,
            lambda_ha=0.04,
            tau_0=0.01,
            tau_1=0.01,
            tau_2=0.01,
            regularization=1e-6
        )
    }
    
    print("\nStarting model evaluation...")
    print("----------------------------------------------------------------------")
    
    for name, model in models.items():
        # Train the model
        start_time = time.time()
        model.fit(tr_inputs, tr_targets, washout=washout)
        train_time = time.time() - start_time
        
        # Predict on training data (after washout)
        tr_pred = model.predict(tr_inputs)[washout:]
        tr_mse = np.mean((tr_pred - tr_targets[washout:]) ** 2)
        
        # Predict on test data
        # We pass the final state from training as the initial state for testing
        if isinstance(model, ChordESNPipeline):
            last_state = model.get_state()
        else:
            _, last_state = model.forward(tr_inputs, washout=0)
            
        start_time = time.time()
        ts_pred = model.predict(ts_inputs, initial_state=last_state)
        eval_time = time.time() - start_time
        ts_mse = np.mean((ts_pred - ts_targets) ** 2)
        
        print(f"Model: {name}")
        print(f"  Training MSE:   {tr_mse:.6f}")
        print(f"  Test MSE:       {ts_mse:.6f}")
        print(f"  Train Time:     {train_time * 1000:.2f} ms")
        print(f"  Inference Time: {eval_time * 1000:.2f} ms")
        print("----------------------------------------------------------------------")

if __name__ == "__main__":
    run_benchmarks()
