import os
import json
import pandas as pd
from pathlib import Path

def aggregate_runs(runs_dir: str, output_file: str):
    """
    Scans runs_dir for completed runs and aggregates their metrics.
    Only considers runs with a completion.json file.
    """
    records = []
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"Directory {runs_dir} does not exist.")
        return
        
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        completion_file = run_dir / "completion.json"
        metrics_file = run_dir / "metrics.json"
        config_file = run_dir / "config.resolved.json"
        
        if not completion_file.exists() or not metrics_file.exists() or not config_file.exists():
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
            # Flatten or extract necessary keys
            record = {
                "run_id": run_dir.name,
                "model_config": config.get("experiment", "unknown"),
                "scenario": config.get("scenario", "unknown"),
                "seed": config.get("seed", -1),
                "benchmark_id": config.get("benchmark_id", "unknown"),
                
                # Metrics (extracting top-level scalars)
                "macro_ap": metrics.get("macro_ap", float('nan')),
                "macro_auroc": metrics.get("macro_auroc", float('nan')),
                "micro_ap": metrics.get("micro_ap", float('nan')),
                "micro_auroc": metrics.get("micro_auroc", float('nan')),
                "ece_15": metrics.get("ece_15", float('nan')),
                "brier_score": metrics.get("brier_score", float('nan')),
                "precision_at_5": metrics.get("precision_at_k", float('nan')),
                "recall_at_5": metrics.get("recall_at_k", float('nan')),
                "ndcg_at_5": metrics.get("ndcg_at_k", float('nan'))
            }
            
            records.append(record)
        except Exception as e:
            print(f"Error parsing run {run_dir.name}: {e}")
            
    if not records:
        print("No completed runs found.")
        return
        
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"Aggregated {len(df)} runs to {output_file}")

if __name__ == "__main__":
    aggregate_runs("artifacts/runs", "artifacts/aggregated_results.csv")
