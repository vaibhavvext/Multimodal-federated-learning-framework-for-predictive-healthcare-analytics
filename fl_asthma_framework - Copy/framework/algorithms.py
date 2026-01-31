import numpy as np

def fedavg(weights_list):
    coefs = np.mean([w[0] for w in weights_list], axis=0).astype(np.float32)
    intercepts = np.mean([w[1] for w in weights_list], axis=0).astype(np.float32)
    return coefs, intercepts

def weighted_fedavg(weights_list, sizes):
    total = float(np.sum(sizes))
    coefs = np.zeros_like(weights_list[0][0], dtype=np.float64)
    intercepts = np.zeros_like(weights_list[0][1], dtype=np.float64)
    for (coef, intercept), n in zip(weights_list, sizes):
        w = n / total
        coefs += coef.astype(np.float64) * w
        intercepts += intercept.astype(np.float64) * w
    return coefs.astype(np.float32), intercepts.astype(np.float32)
