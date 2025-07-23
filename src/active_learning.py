"""
Active Learning Framework for Materials Discovery

Bayesian optimization with CGCNN surrogate models for efficient
materials design and property optimization. Combines uncertainty
quantification with adaptive sampling strategies.

Author: lunazhang
Date: 2023
"""

import os
import random
import time
from typing import List, Tuple

import numpy as np

try:
    from skopt import Optimizer
    from skopt.space import Real
except ImportError as e:
    raise ImportError('Please install scikit-optimize to use this script: pip install scikit-optimize') from e

from predict import predict  # import the function we just refactored


# ----------------------  CONFIG  ----------------------
MODEL_CKPT = 'model_best.pth.tar'  # path to your trained CGCNN model
CIF_TEMPLATE_DIR = 'vacancy_data'  # directory with base .cif files to mutate
GENERATED_DIR = 'generated_cifs'   # directory to dump generated candidates
os.makedirs(GENERATED_DIR, exist_ok=True)

N_INITIAL = 10     # initial random candidates
N_ITER = 100       # total BO iterations
K_AL_EPOCH = 20    # active-learning cycle: every K iterations retrain model
N_MCDO = 25        # Monte-Carlo dropout passes for uncertainty
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# Define a simple 2-D search space: Li vacancy fraction [0,1], Ni-Co-Mn content Ni [0.6,0.9]
# (This is toy—real design would need compositional constraints.)
space = [Real(0.0, 1.0, name='li_vac'), Real(0.6, 0.9, name='ni_fraction')]

# Pretend we have some DFT oracle (placeholder)
def run_dft_simulation(cif_path: str) -> float:
    """Mock DFT – returns noisy true value for demonstration."""
    time.sleep(0.1)  # pretend it is expensive
    # Here we just return a random value; in practice call VASP / QE etc.
    return random.uniform(-5, 0)


def mutate_cif(base_cif: str, li_vac: float, ni_frac: float, idx: int) -> str:
    """Produce a mutated cif filename; real implementation should edit structure."""
    new_name = f'cand_{idx:04d}.cif'
    new_path = os.path.join(GENERATED_DIR, new_name)
    # In reality you would edit the file; we simply copy the base file for demo.
    from shutil import copyfile
    copyfile(base_cif, new_path)
    return new_path


# ----------------------  MAIN LOOP  ----------------------

def main():
    base_cifs = [os.path.join(CIF_TEMPLATE_DIR, f) for f in os.listdir(CIF_TEMPLATE_DIR) if f.endswith('.cif')]
    opt = Optimizer(space, random_state=SEED)

    all_cif_paths: List[str] = []
    all_scores: List[float] = []

    for step in range(N_ITER):
        # 1. Suggest next design parameters
        if step < N_INITIAL:
            params = [s.rvs(random_state=SEED + step)[0] for s in space]
        else:
            params = opt.ask()

        li_vac, ni_frac = params
        base_cif = random.choice(base_cifs)
        cif_path = mutate_cif(base_cif, li_vac, ni_frac, step)

        # 2. Fast CGCNN estimate
        preds = predict(MODEL_CKPT, cif_dir=os.path.dirname(cif_path), batch_size=1,
                        workers=0, device=None, print_freq=0, n_dropout=N_MCDO)
        pred_val, pred_std = list(preds.values())[0]

        # 3. Store & update BO
        opt.tell(params, pred_val)
        all_cif_paths.append(cif_path)
        all_scores.append(pred_val)

        print(f"[BO] Iter {step:03d}  params={params}  CGCNN={pred_val:.3f} ± {pred_std:.3f}")

        # 4. Active-learning cycle: every K steps, run expensive DFT on top-K uncertain samples
        if (step + 1) % K_AL_EPOCH == 0:
            top_indices = np.argsort(-np.array(all_scores))[:3]  # pick 3 best
            for idx in top_indices:
                true_val = run_dft_simulation(all_cif_paths[idx])
                print(f"    [DFT]  {all_cif_paths[idx]}  true={true_val:.3f}")
                # TODO: append to training CSV & retrain CGCNN (fine-tune). This part
                # is highly project-specific and left as an exercise.

if __name__ == '__main__':
    main() 