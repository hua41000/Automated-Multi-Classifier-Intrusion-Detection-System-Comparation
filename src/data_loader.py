import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_sample_data(filepath, sample_size=0.1):
    """
    Loads CICIDS2017 dataset and performs stratified sampling.
    Fulfills Project 4 requirement for dataset selection and handling.
    """
    df = pd.read_csv(filepath)
    
    # Standardize column names by removing leading/trailing spaces
    # CICIDS2017 is known for having " Label" or "Label "
    df.columns = df.columns.str.strip()
    
    target_col = "Label"
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {filepath}. Check CSV headers.")

    # Stratified sampling ensures attack categories are preserved in the 10% sample
    _, df_sample = train_test_split(
        df, 
        test_size=sample_size, 
        stratify=df[target_col], 
        random_state=42
    )
    
    print(f"Loaded {filepath}: {len(df_sample)} rows using '{target_col}' as target.")
    return df_sample, target_col