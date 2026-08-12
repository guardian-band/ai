import os
import json
import pandas as pd
import numpy as np

def integrate_primekg(primekg_path, output_dir):
    print("Parsing and indexing Harvard PrimeKG knowledge graph...")
    os.makedirs(output_dir, exist_ok=True)
    
    entity_counts = {}
    relation_counts = {}
    total_edges = 0
    
    # Process PrimeKG in chunks
    for chunk in pd.read_csv(primekg_path, chunksize=1000000, usecols=['relation', 'x_type', 'y_type', 'x_name', 'y_name']):
        total_edges += len(chunk)
        for r, count in chunk['relation'].value_counts().items():
            relation_counts[r] = relation_counts.get(r, 0) + count
        for xt in chunk[['x_type', 'x_name']].drop_duplicates().itertuples():
            entity_counts[xt.x_type] = entity_counts.get(xt.x_type, set())
            entity_counts[xt.x_type].add(xt.x_name)
        for yt in chunk[['y_type', 'y_name']].drop_duplicates().itertuples():
            entity_counts[yt.y_type] = entity_counts.get(yt.y_type, set())
            entity_counts[yt.y_type].add(yt.y_name)
            
    summary = {
        "dataset_name": "Harvard PrimeKG (Precision Medicine Knowledge Graph)",
        "total_relationships_edges": total_edges,
        "unique_entity_types_count": len(entity_counts),
        "entities_breakdown": {k: len(v) for k, v in sorted(entity_counts.items(), key=lambda x: len(x[1]), reverse=True)},
        "top_relations": {k: int(v) for k, v in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:15]}
    }
    
    summary_path = os.path.join(output_dir, "primekg_metadata_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"Saved PrimeKG metadata summary to {summary_path}")
    return summary

if __name__ == "__main__":
    base_dir = "/Users/acelyayildiz/.gemini/antigravity/scratch/polypharmacy_ai"
    primekg_file = os.path.join(base_dir, "data/external/primekg_kg.csv")
    out_dir = os.path.join(base_dir, "artifacts")
    integrate_primekg(primekg_file, out_dir)
