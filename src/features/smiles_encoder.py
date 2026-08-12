import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

def smiles_to_rich_molecular_features(smiles_str, n_bits_morgan=512, n_bits_rdk=512):
    total_dim = n_bits_morgan + 167 + n_bits_rdk + 10
    if not isinstance(smiles_str, str) or not smiles_str.strip():
        return np.zeros(total_dim, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return np.zeros(total_dim, dtype=np.float32)
        
        # 1. Morgan Fingerprint (Circular)
        fp_morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits_morgan), dtype=np.float32)
        
        # 2. MACCS Keys (Substructure fragments)
        fp_maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        
        # 3. RDKit Topological Fingerprint (Path-based)
        fp_rdk = np.array(Chem.RDKFingerprint(mol, fpSize=n_bits_rdk), dtype=np.float32)
        
        # 4. Physicochemical Descriptors
        desc = np.array([
            Descriptors.MolWt(mol) / 500.0,
            Descriptors.MolLogP(mol) / 5.0,
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.NumHDonors(mol) / 10.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumRotatableBonds(mol) / 10.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.RingCount(mol) / 10.0,
            Descriptors.NumAromaticRings(mol) / 5.0,
            Descriptors.HeavyAtomCount(mol) / 50.0
        ], dtype=np.float32)
        
        return np.concatenate([fp_morgan, fp_maccs, fp_rdk, desc])
    except Exception:
        return np.zeros(total_dim, dtype=np.float32)

def build_drug_features(drugs_master_path, indications_path, targets_path="data/external/ChG-TargetDecagon_targets.csv", n_bits=512, n_top_targets=300):
    print("Loading drugs_master.csv, indications.csv, and Drug-Target network...")
    drugs_df = pd.read_csv(drugs_master_path)
    ind_df = pd.read_csv(indications_path)
    
    # 1. Molecular Features
    print(f"Generating Rich Hybrid Molecular Features (Morgan + MACCS + Topological + Physicochemical) for {len(drugs_df)} drugs...")
    fp_matrix = []
    for smiles in drugs_df['smiles']:
        fp_matrix.append(smiles_to_rich_molecular_features(smiles, n_bits_morgan=n_bits, n_bits_rdk=n_bits))
    fp_matrix = np.vstack(fp_matrix)
    
    # 2. Indication Vectors
    print("Building indication vector representations...")
    unique_diseases = ind_df['y_name'].dropna().unique()
    disease_to_idx = {disease: idx for idx, disease in enumerate(unique_diseases)}
    
    ind_matrix = np.zeros((len(drugs_df), len(unique_diseases)), dtype=np.float32)
    db_to_row = {db_id: row_idx for row_idx, db_id in enumerate(drugs_df['drugbank_id'])}
    
    for _, row in ind_df.iterrows():
        db_id = row['x_id']
        disease = row['y_name']
        if db_id in db_to_row and disease in disease_to_idx:
            r_idx = db_to_row[db_id]
            c_idx = disease_to_idx[disease]
            ind_matrix[r_idx, c_idx] = 1.0
            
    # 3. Drug-Target Protein Interaction Vector (DTI)
    target_matrix = np.zeros((len(drugs_df), n_top_targets), dtype=np.float32)
    if os.path.exists(targets_path):
        print(f"Integrating BioSNAP Drug-Target (DTI) Protein Network (Top {n_top_targets} Target Genes)...")
        targets_df = pd.read_csv(targets_path, comment='#', names=['Drug', 'Gene'])
        
        # Map STITCH CID -> DrugBank ID
        drugs_df['pubchem_cid_clean'] = pd.to_numeric(drugs_df['pubchem_compound_id'], errors='coerce')
        cid_to_drugbank = dict(zip(drugs_df['pubchem_cid_clean'].dropna().astype(int), drugs_df['drugbank_id']))
        
        def parse_stitch(stitch_str):
            if not isinstance(stitch_str, str):
                return None
            cleaned = stitch_str.replace("CID000", "").replace("CID00", "").replace("CID0", "").replace("CID1", "").replace("CID", "")
            try:
                return int(cleaned)
            except ValueError:
                return None
                
        targets_df['cid'] = targets_df['Drug'].apply(parse_stitch)
        targets_df['db_id'] = targets_df['cid'].map(cid_to_drugbank)
        
        top_genes = targets_df['Gene'].value_counts().head(n_top_targets).index.tolist()
        gene_to_idx = {gene: idx for idx, gene in enumerate(top_genes)}
        
        for _, row in targets_df.dropna(subset=['db_id']).iterrows():
            db_id = row['db_id']
            gene = row['Gene']
            if db_id in db_to_row and gene in gene_to_idx:
                r_idx = db_to_row[db_id]
                c_idx = gene_to_idx[gene]
                target_matrix[r_idx, c_idx] = 1.0
                
    feature_matrix = np.hstack([fp_matrix, ind_matrix, target_matrix])
    print(f"Rich Bio-Molecular Drug Feature Matrix Shape: {feature_matrix.shape} (Molecular: {fp_matrix.shape[1]}, Indication: {len(unique_diseases)}, Targets: {target_matrix.shape[1]})")
    
    drug_feature_dict = {}
    for db_id, feat in zip(drugs_df['drugbank_id'], feature_matrix):
        drug_feature_dict[db_id] = feat
        
    return drug_feature_dict, feature_matrix.shape[1]
