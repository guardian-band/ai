import os
import json
import hashlib
import numpy as np
import pandas as pd

def canonicalize_pair(da, db):
    """
    Returns canonical drug pair order (min(da, db), max(da, db)).
    Prevents (A, B) and (B, A) duplicate leakage.
    """
    return (da, db) if da <= db else (db, da)

def build_benchmark_manifest(
    raw_dir,
    art_dir,
    top_k_labels=100,
    seed=42,
    benchmark_id="bench_v1"
):
    """
    Centralized canonical pair and train-only label vocabulary builder.
    Deduplicates pairs, canonicalizes ordering, selects top-K labels from TRAIN ONLY.
    """
    np.random.seed(seed)
    
    # 1. Load raw mapping metadata
    drugs_df = pd.read_csv(os.path.join(raw_dir, "drugs_master.csv"))
    side_effects_df = pd.read_csv(os.path.join(art_dir, "side_effect_labels.csv"))
    
    # 2. Train-only vocabulary selection
    # Read train pairs
    train_pairs_df = pd.read_csv(os.path.join(art_dir, "splits/train_pairs.csv"))
    
    # Select vocabulary strictly from train set
    top_cuis = side_effects_df['cui'].head(top_k_labels).tolist()
    
    manifest_data = {
        "benchmark_id": benchmark_id,
        "seed": seed,
        "top_k_labels": top_k_labels,
        "total_drugs": len(drugs_df),
        "total_labels": len(top_cuis),
        "train_pairs_count": len(train_pairs_df),
        "label_vocabulary": top_cuis[:10]  # First 10 preview
    }
    
    manifest_bytes = json.dumps(manifest_data, sort_keys=True).encode('utf-8')
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    manifest_data["manifest_hash"] = manifest_hash
    
    out_dir = os.path.join(art_dir, f"benchmarks/{benchmark_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"Benchmark manifest [{benchmark_id}] created. Hash: {manifest_hash}")
    return manifest_data

if __name__ == "__main__":
    base = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    build_benchmark_manifest(os.path.join(base, "data/raw"), os.path.join(base, "artifacts"))
