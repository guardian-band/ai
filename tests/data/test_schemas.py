import pytest
import pandas as pd
from src.data.schemas import validate_dataframe

def test_valid_schema():
    df = pd.DataFrame({'drugbank_id': ['DB001'], 'pubchem_compound_id': ['123']})
    assert validate_dataframe(df, 'drugs_master') == True

def test_missing_columns():
    df = pd.DataFrame({'drugbank_id': ['DB001']})
    with pytest.raises(ValueError, match="Missing required columns for drugs_master: \\['pubchem_compound_id'\\]"):
        validate_dataframe(df, 'drugs_master')

def test_unknown_schema():
    df = pd.DataFrame({'drugbank_id': ['DB001']})
    with pytest.raises(ValueError, match="Unknown schema name: invalid_schema"):
        validate_dataframe(df, 'invalid_schema')
