import pandas as pd
from pathlib import Path
import os

def filter_primekg(input_csv: str, output_csv: str):
    print(f"Reading PrimeKG from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    initial_count = len(df)
    print(f"Initial edges: {initial_count}")
    
    relations_to_remove = {
        "drug_drug",
        "drug_effect",
        "contraindication",
        "indication"
    }
    
    # Check what relations are present
    print("\nRelations in dataset:")
    counts = df['relation'].value_counts()
    for rel, count in counts.items():
        if rel in relations_to_remove:
            print(f" - {rel}: {count} (WILL BE REMOVED)")
        else:
            print(f" - {rel}: {count}")
            
    filtered_df = df[~df['relation'].isin(relations_to_remove)]
    final_count = len(filtered_df)
    
    print(f"\nFinal edges: {final_count}")
    print(f"Removed {initial_count - final_count} leakage edges.")
    
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving filtered graph to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    input_file = "data/external/primekg_kg.csv"
    output_file = "artifacts/filtered_primekg.csv"
    filter_primekg(input_file, output_file)
