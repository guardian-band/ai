import subprocess
import sys
import os
import shutil

def run_command(cmd, desc):
    print(f"\n{'='*80}\nStarting: {desc}\nCommand: {cmd}\n{'='*80}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}: {desc}")
        sys.exit(1)
    print(f"\n[SUCCESS] Completed: {desc}")

def main():
    runs = [
        # warm_pair only requires a single universal encoder, using seed 42 as the reference to extract all seen drugs
        ("warm_pair", 42),
        
        # cold_1 (4 remaining seeds, 42 already done)
        ("cold_1", 101),
        ("cold_1", 2024),
        ("cold_1", 27182),
        ("cold_1", 31415),
        
        # cold_2 (all 5 seeds)
        ("cold_2", 42),
        ("cold_2", 101),
        ("cold_2", 2024),
        ("cold_2", 27182),
        ("cold_2", 31415),
    ]
    
    total_runs = len(runs)
    print(f"Starting GraphSAGE Orchestration Sweep for {total_runs} runs...")
    
    # 0. Pre-flight Assertion Checks
    print("\n--- Running Pre-flight Asset Checks ---")
    essential_files = [
        "data/processed/colab_primekg_safe.parquet",
        "data/processed/colab_morgan_features.parquet",
        "scripts/colab_graphsage_full_training_sweep.py"
    ]
    for scenario, seed in runs:
        essential_files.append(f"artifacts/benchmarks/polypharmacy_v1/{scenario}/seed_{seed}/manifest.json")
        essential_files.append(f"artifacts/benchmarks/polypharmacy_v1/{scenario}/seed_{seed}/pairs.parquet")
        
    for filepath in essential_files:
        assert os.path.exists(filepath), f"CRITICAL ERROR: Missing required file for sweep: {filepath}"
        
    print("All required benchmark splits, manifests, and data files are present!\n")
    
    for i, (scenario, seed) in enumerate(runs, 1):
        print(f"\n>>> RUN {i}/{total_runs}: Scenario={scenario}, Seed={seed} <<<")
        
        run_name = f"GraphSAGE_universal_encoder" if scenario == "warm_pair" else f"GraphSAGE_{scenario}_seed_{seed}"
        zip_name = f"{run_name}_artifacts.zip"
        drive_path = "/content/drive/MyDrive/polypharmacy_ai_artifacts"
        drive_zip_path = os.path.join(drive_path, zip_name)
        
        # 1. Resume Check
        if os.path.exists(drive_zip_path):
            print(f"[SKIP] Run {run_name} is already completed and backed up in Google Drive at {drive_zip_path}.")
            continue
        elif os.path.exists(zip_name):
            print(f"[SKIP] Run {run_name} is already completed locally (found {zip_name}). Moving to next run...")
            if os.path.exists("/content/drive/MyDrive"):
                os.makedirs(drive_path, exist_ok=True)
                run_command(f"cp {zip_name} {drive_zip_path}", f"Copying {zip_name} to Google Drive")
            continue
        
        # 2. Run the training script
        train_cmd = f"python scripts/colab_graphsage_full_training_sweep.py --scenario {scenario} --seed {seed}"
        run_command(train_cmd, f"Training GraphSAGE ({scenario}/seed_{seed})")
        
        # 3. Verify Run Completion & Package the artifacts
        config_path = f"artifacts/runs/{run_name}/config.resolved.json"
        if not os.path.exists(config_path):
            print(f"\n[ERROR] Run {run_name} did not complete successfully (missing config.resolved.json). Halting sweep.")
            sys.exit(1)
            
        zip_cmd = (
            f"zip -r {zip_name} "
            f"artifacts/runs/{run_name}/ "
            f"artifacts/embeddings/{run_name}.npy "
            f"artifacts/embeddings/{run_name}_drug_mapping.parquet"
        )
        run_command(zip_cmd, f"Zipping artifacts for {run_name}")
        
        # 4. Backup to Google Drive immediately
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(drive_path, exist_ok=True)
            copy_cmd = f"cp {zip_name} {drive_zip_path}"
            run_command(copy_cmd, f"Copying {zip_name} to Google Drive")
            print(f"Successfully backed up {zip_name} to Google Drive!")
        else:
            print(f"[INFO] Google Drive not mounted at /content/drive/MyDrive. Kept {zip_name} locally.")
            
    print(f"\n{'='*80}\nALL {total_runs} GRAPHSAGE RUNS COMPLETED SUCCESSFULLY!\n{'='*80}")

if __name__ == "__main__":
    main()
