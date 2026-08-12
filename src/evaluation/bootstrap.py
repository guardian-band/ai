import numpy as np
import pandas as pd
from typing import Dict, Callable

def calculate_bootstrap_ci(
    data: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    resamples: int = 2000,
    seed: int = 8675309,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Calculates bootstrap confidence intervals by resampling pair rows.
    """
    rng = np.random.default_rng(seed)
    
    # Identify unique pair rows to resample. Usually data is flat (pairs x labels)
    # We want to resample pairs, not individual label predictions independently
    pair_ids = data['pair_id'].unique()
    n_pairs = len(pair_ids)
    
    results = []
    failures = 0
    
    for _ in range(resamples):
        # Sample with replacement
        sample_ids = rng.choice(pair_ids, size=n_pairs, replace=True)
        
        # Build the sampled dataframe. This is slow if done naively.
        # Faster approach: merge or use index
        # For evaluation, we can assume 'metric_fn' takes the dataframe of predictions
        # A more optimal way is to work with indices, but for correctness let's reconstruct
        
        # We can map pair_id to groups of indices
        # Since data is likely pair_id oriented, let's do a join
        sample_df = pd.DataFrame({'pair_id': sample_ids})
        resampled_data = sample_df.merge(data, on='pair_id', how='left')
        
        try:
            val = metric_fn(resampled_data)
            if not np.isnan(val):
                results.append(val)
            else:
                failures += 1
        except Exception:
            failures += 1
            
    if not results:
        return {
            "mean": float('nan'),
            "lower_ci": float('nan'),
            "upper_ci": float('nan'),
            "failures": failures
        }
        
    alpha = 1.0 - confidence_level
    lower_p = alpha / 2.0 * 100
    upper_p = (1.0 - alpha / 2.0) * 100
    
    lower_ci = np.percentile(results, lower_p)
    upper_ci = np.percentile(results, upper_p)
    
    return {
        "mean": float(np.mean(results)),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
        "failures": failures
    }
