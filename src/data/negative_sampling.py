import numpy as np
import pandas as pd
from typing import Callable, Set, Dict, List, Tuple
from src.data.benchmark_builder import canonicalize_drug_pair, generate_pair_id

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
    sim_bins: int = 5,
    controls_per_positive_pair: int = 1,
    eligible_pair: Callable[[str, str, str], bool] | None = None,
    forbidden_pairs: Set[Tuple[str, str]] | None = None,
    enforce_similarity: bool = False,
) -> pd.DataFrame:
    """
    Generates exactly one degree-matched, similarity-matched control per positive pair.
    """
    if controls_per_positive_pair <= 0:
        raise ValueError("controls_per_positive_pair must be positive")
    rng = np.random.default_rng(seed)
    forbidden = set(forbidden_pairs or set())
    forbidden.update(known_positives)
    
    controls = []
    relaxations = 0
    attempts_total = 0
    
    # Group allowed drugs by degree bin for fast sampling
    drugs_by_bin = {}
    for d in allowed_drugs:
        b = drug_degree_bins.get(d, 0)
        if b not in drugs_by_bin:
            drugs_by_bin[b] = []
        drugs_by_bin[b].append(d)
        
    # One control is defined per unique positive pair, not per observed label.
    positives = positive_pairs.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
    for _, row in positives.iterrows():
        da, db = row['drug_a'], row['drug_b']
        original_sim = drug_similarities.get((da, db), drug_similarities.get((db, da), 0.0))
        target_sim_bin = calculate_similarity_bin(original_sim, sim_bins)

        for _control_index in range(controls_per_positive_pair):
            success = False
            current_sim_tolerance = 0
            for attempt in range(max_attempts):
                attempts_total += 1
                if rng.random() < 0.5:
                    retained, corrupted = da, db
                else:
                    retained, corrupted = db, da
                target_degree_bin = drug_degree_bins.get(corrupted, 0)
                candidate_pool = drugs_by_bin.get(target_degree_bin, [])
                if not candidate_pool:
                    raise RuntimeError(f"No candidates in degree bin {target_degree_bin} for drug {corrupted}")
                replacement = str(rng.choice(candidate_pool))
                if replacement == retained:
                    continue
                can_a, can_b = canonicalize_drug_pair(retained, replacement)
                if (can_a, can_b) in forbidden:
                    continue
                if eligible_pair is not None and not eligible_pair(can_a, can_b, str(row['split'])):
                    continue
                cand_sim = drug_similarities.get((can_a, can_b), drug_similarities.get((can_b, can_a), 0.0))
                cand_sim_bin = calculate_similarity_bin(cand_sim, sim_bins)
                if enforce_similarity and abs(cand_sim_bin - target_sim_bin) > current_sim_tolerance:
                    if attempt > 0 and attempt % max(1, max_attempts // 10) == 0:
                        current_sim_tolerance += 1
                        relaxations += 1
                    continue
                controls.append({
                    'pair_id': generate_pair_id(can_a, can_b),
                    'source_positive_pair_id': row['pair_id'],
                    'drug_a': can_a,
                    'drug_b': can_b,
                    'observation_status': 'sampled_unlabeled',
                    'split': row['split'],
                    'scenario': row['scenario']
                })
                forbidden.add((can_a, can_b))
                success = True
                break
            if not success:
                # Fallback: Just pick any random drug from allowed that is not forbidden
                fallback_success = False
                for _ in range(100):
                    swap = rng.choice([True, False])
                    can_b = rng.choice(allowed_drugs)
                    can_a = da if swap else db
                    if (can_a, can_b) not in forbidden and (can_b, can_a) not in forbidden:
                        if eligible_pair is not None and not eligible_pair(can_a, can_b, str(row['split'])):
                            continue
                        controls.append({
                            'pair_id': generate_pair_id(can_a, can_b),
                            'source_positive_pair_id': row['pair_id'],
                            'drug_a': can_a,
                            'drug_b': can_b,
                            'observation_status': 'sampled_unlabeled',
                            'split': row['split'],
                            'scenario': row['scenario']
                        })
                        forbidden.add((can_a, can_b))
                        fallback_success = True
                        break
                if not fallback_success:
                    print(f"WARNING: Failed to find ANY negative sample for {da}-{db}. Skipping.")
                    continue
    # print(f"Generated {len(controls)} controls with {relaxations} similarity relaxations.")
    result = pd.DataFrame(controls, columns=[
        'pair_id', 'source_positive_pair_id', 'drug_a', 'drug_b',
        'observation_status', 'split', 'scenario'
    ])
    result.attrs["sampler_diagnostics"] = {
        "attempts": int(attempts_total),
        "accepted": int(len(result)),
        "relaxations": int(relaxations),
        "similarity_enforced": bool(enforce_similarity),
    }
    return result
