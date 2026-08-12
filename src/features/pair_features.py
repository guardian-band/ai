import numpy as np

def compute_tanimoto_similarity(vec_a, vec_b):
    intersection = np.dot(vec_a, vec_b)
    denom = np.dot(vec_a, vec_a) + np.dot(vec_b, vec_b) - intersection
    if denom == 0:
        return 0.0
    return intersection / denom

def build_symmetric_pair_features(vec_a, vec_b):
    sum_feat = vec_a + vec_b
    diff_feat = np.abs(vec_a - vec_b)
    mult_feat = vec_a * vec_b
    tanimoto_sim = np.array([compute_tanimoto_similarity(vec_a[:512], vec_b[:512])], dtype=np.float32)
    
    return np.concatenate([sum_feat, diff_feat, mult_feat, tanimoto_sim])
