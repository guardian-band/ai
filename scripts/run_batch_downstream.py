import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import subprocess

def combine_features(encoder_name):
    # Load Morgan features
    morgan_df = pd.read_parquet("data/processed/colab_morgan_features.parquet")
    
    # Load GraphSAGE embeddings
    emb_path = f"artifacts/embeddings/{encoder_name}.npy"
    map_path = f"artifacts/embeddings/{encoder_name}_drug_mapping.parquet"
    
    graphsage_embs = np.load(emb_path)
    mapping_df = pd.read_parquet(map_path)
    
    combined_features = []
    drugbank_ids = []
    
    # Pre-index morgan for faster lookup
    morgan_dict = dict(zip(morgan_df['drugbank_id'], morgan_df['morgan_2048']))
    
    for i, row in mapping_df.iterrows():
        drug_id = row['drugbank_id']
        idx = row['node_index']
        
        morgan_fp = morgan_dict[drug_id]
        graphsage_fp = graphsage_embs[idx]
        
        combined = np.concatenate([morgan_fp, graphsage_fp])
        combined_features.append(combined)
        drugbank_ids.append(drug_id)
        
    df_combined = pd.DataFrame({
        'drugbank_id': drugbank_ids,
        'morgan_fingerprint': combined_features
    })
    
    out_path = "artifacts/morgan_graphsage_features.parquet"
    df_combined.to_parquet(out_path, index=False)
    print(f"Features combined for {encoder_name} and saved to {out_path}")

def run_experiment(scenario, seed):
    manifest_path = f"artifacts/benchmarks/polypharmacy_v1/{scenario}/seed_{seed}/manifest.json"
    config_path = "configs/model_morgan_graphsage_mlp.yaml"
    
    cmd = [
        sys.executable, "run_experiment.py",
        "--experiment", config_path,
        "--benchmark", manifest_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    runs = []
    seeds = [42, 101, 2024, 27182, 31415]
    for s in seeds:
        runs.append(("warm_pair", s))
    for s in seeds:
        runs.append(("cold_1", s))
    for s in seeds:
        runs.append(("cold_2", s))
        
    completed_runs = 0
    total_runs = len(runs)
    
    for scenario, seed in runs:
        # Check if already completed
        run_dirs = glob.glob(f"artifacts/runs/morgan_graphsage_mlp__{scenario}__seed_{seed}__*")
        is_completed = False
        for rd in run_dirs:
            if os.path.exists(os.path.join(rd, "metrics.json")):
                is_completed = True
                break
                
        if is_completed:
            print(f"[{completed_runs+1}/{total_runs}] SKIP: {scenario}/seed_{seed} already completed.")
            completed_runs += 1
            continue
            
        print(f"\n[{completed_runs+1}/{total_runs}] STARTING: {scenario}/seed_{seed}")
        
        if scenario == "warm_pair":
            encoder_name = "GraphSAGE_universal_encoder"
        else:
            encoder_name = f"GraphSAGE_{scenario}_seed_{seed}"
            
        combine_features(encoder_name)
        run_experiment(scenario, seed)
        completed_runs += 1

    print("\nAll 15 runs processed. Generating final benchmark results...")

    # Aggregation logic
    results = []
    for scenario in ["warm_pair", "cold_1", "cold_2"]:
        macro_aps = []
        micro_aps = []
        macro_aurocs = []
        prevalences = []
        ap_lifts = []
        
        run_dirs = glob.glob(f"artifacts/runs/morgan_graphsage_mlp__{scenario}__seed_*")
        # Filter strictly those with metrics.json
        valid_dirs = [rd for rd in run_dirs if os.path.exists(os.path.join(rd, "metrics.json"))]
        
        for run in valid_dirs:
            with open(os.path.join(run, "metrics.json")) as f:
                m = json.load(f)
                macro_aps.append(m["macro_ap"])
                micro_aps.append(m["micro_ap"])
                macro_aurocs.append(m["macro_auroc"])
                
                label_prevalences = []
                label_lifts = []
                for label, lm in m["per_label_metrics"].items():
                    if "ap_lift" in lm and lm["ap_lift"] > 0:
                        prev = lm["ap"] / lm["ap_lift"]
                        label_prevalences.append(prev)
                        label_lifts.append(lm["ap_lift"])
                
                prevalences.append(np.mean(label_prevalences))
                ap_lifts.append(np.mean(label_lifts))
                
        results.append({
            "Scenario": scenario,
            "Macro AP (Mean)": np.mean(macro_aps),
            "Macro AP (SD)": np.std(macro_aps),
            "Micro AP (Mean)": np.mean(micro_aps),
            "Micro AP (SD)": np.std(micro_aps),
            "Macro AUROC (Mean)": np.mean(macro_aurocs),
            "Macro AUROC (SD)": np.std(macro_aurocs),
            "Label Prevalence (Macro)": np.mean(prevalences),
            "AP Lift (Macro)": np.mean(ap_lifts),
            "Valid Runs": len(valid_dirs)
        })
        
    df = pd.DataFrame(results)
    df.to_csv("artifacts/final_benchmark_results.csv", index=False)
    print("Saved final results to artifacts/final_benchmark_results.csv")
    
if __name__ == "__main__":
    main()
