import numpy as np
import pandas as pd
from typing import Set, Dict, List, Tuple
from src.data.benchmark_builder import canonicalize_drug_pair

def calculate_similarity_bin(sim: float, n_bins: int = 5) -> int:
    """Bin similarity [0, 1] into n_bins. 0 is lowest, n_bins-1 is highest."""
    if sim >= 1.0:
        return n_bins - 1
    return int(sim * n_bins)

def generate_negative_samples(
    positive_pairs: pd.DataFrame,
    known_positives: Set[Tuple[str, str]],
    allowed_drugs: List[str],
    drug_degree_bins: Dict[str, int],
    drug_similarities: Dict[Tuple[str, str], float], # Precomputed or lazy dict
    seed: int = 42,
    max_attempts: int = 10000,
    sim_bins: int = 5
) -> pd.DataFrame:
    """
    Generates exactly one degree-matched, similarity-matched control per positive pair.
    """
    rng = np.random.default_rng(seed)
    
    controls = []
    relaxations = 0
    
    # Group allowed drugs by degree bin for fast sampling
    drugs_by_bin = {}
    for d in allowed_drugs:
        b = drug_degree_bins.get(d, 0)
        if b not in drugs_by_bin:
            drugs_by_bin[b] = []
        drugs_by_bin[b].append(d)
        
    for _, row in positive_pairs.iterrows():
        da, db = row['drug_a'], row['drug_b']
        original_sim = drug_similarities.get((da, db), drug_similarities.get((db, da), 0.0))
        target_sim_bin = calculate_similarity_bin(original_sim, sim_bins)
        
        # 3. Randomly choose one endpoint to corrupt
        if rng.random() < 0.5:
            retained, corrupted = da, db
        else:
            retained, corrupted = db, da
            
        target_degree_bin = drug_degree_bins.get(corrupted, 0)
        candidate_pool = drugs_by_bin.get(target_degree_bin, [])
        
        if not candidate_pool:
            raise RuntimeError(f"No candidates in degree bin {target_degree_bin} for drug {corrupted}")
            
        success = False
        current_sim_tolerance = 0
        
        for attempt in range(max_attempts):
            replacement = rng.choice(candidate_pool)
            
            # Reject self-pairs
            if replacement == retained:
                continue
                
            can_a, can_b = canonicalize_drug_pair(retained, replacement)
            
            # Reject known positives
            if (can_a, can_b) in known_positives:
                continue
                
            # Check similarity bin
            cand_sim = drug_similarities.get((can_a, can_b), drug_similarities.get((can_b, can_a), 0.0))
            cand_sim_bin = calculate_similarity_bin(cand_sim, sim_bins)
            
            if abs(cand_sim_bin - target_sim_bin) <= current_sim_tolerance:
                success = True
                controls.append({
                    'pair_id': row['pair_id'], # Keep original pair_id but it will be recalculated later if needed, actually the plan says "source_positive_pair_id"
                    'source_positive_pair_id': row['pair_id'],
                    'drug_a': can_a,
                    'drug_b': can_b,
                    'label_cui': row['label_cui'],
                    'observation_status': 'sampled_unlabeled',
                    'split': row['split'],
                    'scenario': row['scenario']
                })
                break
                
            # Relax similarity bin if nearing max attempts
            if attempt > 0 and attempt % 1000 == 0:
                current_sim_tolerance += 1
                relaxations += 1
                
        if not success:
            raise RuntimeError(f"Failed to find negative sample for {da}-{db} after {max_attempts} attempts.")
            
    # print(f"Generated {len(controls)} controls with {relaxations} similarity relaxations.")
    return pd.DataFrame(controls)
