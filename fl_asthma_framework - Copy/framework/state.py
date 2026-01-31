import os
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_weights(path, coef, intercept):
    ensure_dir(os.path.dirname(path))
    np.savez(path, coef=coef.astype(np.float32), intercept=intercept.astype(np.float32))

def load_weights(path):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["coef"].astype(np.float32), data["intercept"].astype(np.float32)
