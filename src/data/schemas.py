import pandas as pd
from typing import List, Dict

# Define expected input schemas
INPUT_SCHEMAS = {
    "drugs_master": ["drugbank_id", "pubchem_compound_id"],
    "side_effects": ["umls_cui_from_meddra", "side_effect_name"],
    "biosnap": ["STITCH 1", "STITCH 2", "Polypharmacy Side Effect", "Side Effect Name"]
}

def validate_dataframe(df: pd.DataFrame, schema_name: str) -> bool:
    """
    Validates that a dataframe contains the required columns for a given schema.
    Raises ValueError if required columns are missing.
    """
    if schema_name not in INPUT_SCHEMAS:
        raise ValueError(f"Unknown schema name: {schema_name}")
        
    expected_cols = INPUT_SCHEMAS[schema_name]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns for {schema_name}: {missing_cols}")
        
    return True
