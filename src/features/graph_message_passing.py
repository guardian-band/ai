"""
from src.training.reproducibility import repository_root

Pre-Computed Graph Message Passing: Drug → Protein → Drug & Drug → Disease → Drug
Enriches each drug's feature vector with aggregated neighborhood information
from the PrimeKG biological knowledge graph WITHOUT loading the graph into GPU.
"""
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

def build_graph_enriched_features(
    base_dir=str(repository_root()) + "",
    n_hops=2
):
    print("="*65)
    print("PRE-COMPUTED GRAPH MESSAGE PASSING (CPU-Only, RAM-Efficient)")
    print("Drug → Protein → Drug  &  Drug → Disease → Drug")
    print("="*65)
    start_time = time.time()
    
    raw_dir = os.path.join(base_dir, "data/raw")
    art_dir = os.path.join(base_dir, "artifacts")
    
    # 1. Load original drug features (2,864 dims - proven effective)
    drug_feat_df = pd.read_parquet(os.path.join(art_dir, "drug_features.parquet"))
    feat_cols = [c for c in drug_feat_df.columns if c.startswith('f_')]
    drug_ids = drug_feat_df['drugbank_id'].tolist()
    drug_feature_dict = {row['drugbank_id']: row[feat_cols].values.astype(np.float32) for _, row in drug_feat_df.iterrows()}
    
    drugs_df = pd.read_csv(os.path.join(raw_dir, "drugs_master.csv"))
    drug_names = dict(zip(drugs_df['name'].str.lower().str.strip(), drugs_df['drugbank_id']))
    
    print(f"Loaded {len(drug_feature_dict)} drug feature vectors ({len(feat_cols)} dims each)")
    
    # 2. Build Drug-Protein bipartite graph from PrimeKG
    print("\nBuilding Drug-Protein bipartite graph from PrimeKG...")
    pkg = pd.read_csv(
        os.path.join(base_dir, "data/external/primekg_kg.csv"),
        usecols=['x_type', 'y_type', 'x_name', 'y_name', 'display_relation']
    )
    
    drug_to_proteins = defaultdict(set)
    protein_to_drugs = defaultdict(set)
    
    # PrimeKG Drug-Protein edges
    dp_edges = pkg[((pkg['x_type']=='drug') & (pkg['y_type']=='gene/protein'))]
    for _, row in dp_edges.iterrows():
        d_name = row['x_name'].lower().strip()
        p_name = row['y_name']
        if d_name in drug_names:
            db_id = drug_names[d_name]
            drug_to_proteins[db_id].add(p_name)
            protein_to_drugs[p_name].add(db_id)
    
    pd_edges = pkg[((pkg['x_type']=='gene/protein') & (pkg['y_type']=='drug'))]
    for _, row in pd_edges.iterrows():
        d_name = row['y_name'].lower().strip()
        p_name = row['x_name']
        if d_name in drug_names:
            db_id = drug_names[d_name]
            drug_to_proteins[db_id].add(p_name)
            protein_to_drugs[p_name].add(db_id)
    
    print(f"  Drug-Protein graph: {len(drug_to_proteins)} drugs connected to {len(protein_to_drugs)} proteins")
    
    # 3. Build Drug-Disease bipartite graph from PrimeKG
    print("Building Drug-Disease bipartite graph from PrimeKG...")
    drug_to_diseases = defaultdict(set)
    disease_to_drugs = defaultdict(set)
    
    dd_edges = pkg[((pkg['x_type']=='drug') & (pkg['y_type']=='disease'))]
    for _, row in dd_edges.iterrows():
        d_name = row['x_name'].lower().strip()
        dis_name = row['y_name']
        if d_name in drug_names:
            db_id = drug_names[d_name]
            drug_to_diseases[db_id].add(dis_name)
            disease_to_drugs[dis_name].add(db_id)
    
    dd_edges2 = pkg[((pkg['x_type']=='disease') & (pkg['y_type']=='drug'))]
    for _, row in dd_edges2.iterrows():
        d_name = row['y_name'].lower().strip()
        dis_name = row['x_name']
        if d_name in drug_names:
            db_id = drug_names[d_name]
            drug_to_diseases[db_id].add(dis_name)
            disease_to_drugs[dis_name].add(db_id)
    
    print(f"  Drug-Disease graph: {len(drug_to_diseases)} drugs connected to {len(disease_to_drugs)} diseases")
    
    del pkg  # Free memory
    
    # 4. Pre-compute 1-hop and 2-hop neighborhood aggregations
    print(f"\nPre-computing {n_hops}-hop graph message passing...")
    in_dim = len(feat_cols)
    
    # Initialize: h^(0) = original features
    h_current = {d: drug_feature_dict[d].copy() for d in drug_ids}
    
    for hop in range(1, n_hops + 1):
        print(f"  Computing hop {hop} message aggregation...")
        h_next = {}
        
        for d in drug_ids:
            # Collect neighbor drug IDs through shared proteins
            protein_neighbors = set()
            for prot in drug_to_proteins.get(d, []):
                for neighbor_d in protein_to_drugs.get(prot, []):
                    if neighbor_d != d and neighbor_d in h_current:
                        protein_neighbors.add(neighbor_d)
            
            # Collect neighbor drug IDs through shared diseases
            disease_neighbors = set()
            for dis in drug_to_diseases.get(d, []):
                for neighbor_d in disease_to_drugs.get(dis, []):
                    if neighbor_d != d and neighbor_d in h_current:
                        disease_neighbors.add(neighbor_d)
            
            # Aggregate neighbor features (weighted mean)
            all_neighbors = protein_neighbors | disease_neighbors
            
            if len(all_neighbors) > 0:
                neighbor_vecs = np.stack([h_current[n] for n in all_neighbors])
                
                # Weight protein neighbors 2x higher (more biologically relevant for DDI)
                weights = np.ones(len(all_neighbors))
                for i, n in enumerate(all_neighbors):
                    if n in protein_neighbors:
                        weights[i] = 2.0
                
                weights /= weights.sum()
                agg_vec = np.average(neighbor_vecs, axis=0, weights=weights)
                
                # GCN-style update: h^(k) = normalize(h^(k-1) + agg_neighbors)
                h_next[d] = h_current[d] + agg_vec
                norm = np.linalg.norm(h_next[d])
                if norm > 0:
                    h_next[d] = h_next[d] / norm * np.linalg.norm(h_current[d])
            else:
                h_next[d] = h_current[d].copy()
        
        h_current = h_next
        
        # Count how many drugs have graph neighbors
        connected = sum(1 for d in drug_ids if len(drug_to_proteins.get(d, set())) > 0 or len(drug_to_diseases.get(d, set())) > 0)
        print(f"    Hop {hop} complete: {connected}/{len(drug_ids)} drugs have graph neighbors")
    
    # 5. Build pair-level graph interaction features
    print("\nComputing pair-level graph interaction features...")
    
    # For each drug, also compute summary statistics of its graph neighborhood
    graph_summary_dim = 8
    drug_graph_summary = {}
    
    for d in drug_ids:
        prot_count = len(drug_to_proteins.get(d, set()))
        dis_count = len(drug_to_diseases.get(d, set()))
        
        # Count protein neighbors' drugs (2nd order connectivity)
        second_order_drugs = set()
        for prot in drug_to_proteins.get(d, []):
            second_order_drugs.update(protein_to_drugs.get(prot, set()))
        second_order_drugs.discard(d)
        
        # Count disease neighbors' drugs
        disease_co_drugs = set()
        for dis in drug_to_diseases.get(d, []):
            disease_co_drugs.update(disease_to_drugs.get(dis, set()))
        disease_co_drugs.discard(d)
        
        summary = np.array([
            np.log1p(prot_count),                    # Log protein target count
            np.log1p(dis_count),                     # Log disease indication count
            np.log1p(len(second_order_drugs)),        # Log 2nd-order drug connectivity
            np.log1p(len(disease_co_drugs)),           # Log disease-co-treated drugs
            prot_count / max(1, prot_count + dis_count),  # Protein ratio
            1.0 if prot_count > 5 else 0.0,           # High protein connectivity flag
            1.0 if len(second_order_drugs) > 20 else 0.0,  # Hub drug flag
            min(prot_count, dis_count) / max(1, max(prot_count, dis_count))  # Balance ratio
        ], dtype=np.float32)
        
        drug_graph_summary[d] = summary
    
    # 6. Combine: [graph-updated features || original features || graph summary]
    print("Assembling final graph-enriched feature matrix...")
    
    enriched_data = []
    for d in drug_ids:
        original = drug_feature_dict[d]
        graph_updated = h_current[d]
        graph_sum = drug_graph_summary.get(d, np.zeros(graph_summary_dim, dtype=np.float32))
        
        # Concatenate: original features + graph-propagated features + graph summary
        enriched = np.concatenate([original, graph_updated, graph_sum])
        enriched_data.append(enriched)
    
    enriched_matrix = np.array(enriched_data)
    total_dim = enriched_matrix.shape[1]
    
    # Create column names
    orig_cols = feat_cols
    graph_cols = [f"g_{i}" for i in range(len(feat_cols))]
    summary_cols = [f"gs_{i}" for i in range(graph_summary_dim)]
    all_cols = orig_cols + graph_cols + summary_cols
    
    enriched_df = pd.DataFrame(enriched_matrix, columns=all_cols)
    enriched_df.insert(0, 'drugbank_id', drug_ids)
    
    # Restore drugbank_name if exists
    if 'drugbank_name' in drug_feat_df.columns:
        enriched_df.insert(1, 'drugbank_name', drug_feat_df['drugbank_name'].tolist())
    
    output_path = os.path.join(art_dir, "drug_features_graph_enriched.parquet")
    enriched_df.to_parquet(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"GRAPH MESSAGE PASSING COMPLETE")
    print(f"{'='*65}")
    print(f"Original Feature Dimension:      {len(feat_cols)}")
    print(f"Graph-Propagated Feature Dimension: {len(feat_cols)}")
    print(f"Graph Summary Features:          {graph_summary_dim}")
    print(f"TOTAL ENRICHED DIMENSION:        {total_dim}")
    print(f"Drugs with Graph Neighbors:      {connected}/{len(drug_ids)}")
    print(f"Saved to: {output_path}")
    print(f"Execution Time: {elapsed:.1f} seconds")
    
    return output_path

if __name__ == "__main__":
    build_graph_enriched_features()
