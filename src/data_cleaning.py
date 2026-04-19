import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def drop_redundant_features(df, threshold=0.98):
    """
    Dynamically identifies and drops features with a correlation 
    higher than the specified threshold to prevent multicollinearity.
    """
    print(f"   -> Scanning for features with > {threshold*100}% correlation...")
    
    # 1. Calculate the absolute correlation matrix
    # We use .abs() because a -1.00 correlation is just as redundant as +1.00
    corr_matrix = df.corr().abs()

    # 2. Select the upper triangle of the correlation matrix
    # Why? If Feature A and Feature B are 100% correlated, we only want to drop ONE of them.
    # The upper triangle ensures we don't accidentally drop both.
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 3. Find index of feature columns with correlation greater than the threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= threshold)]

    # 4. Drop the identified features
    if to_drop:
        print(f"   -> Dropped {len(to_drop)} redundant features: {to_drop}")
        df_reduced = df.drop(columns=to_drop)
    else:
        print("   -> No redundant features found.")
        df_reduced = df.copy()

    return df_reduced

def clean_and_prepare_df(df, target_col="Label"):
    """
    1. Removes Inf/NaN values.
    2. Encodes text labels to numbers.
    3. Scales features to [0, 1].
    4. Returns a single combined DataFrame for the Heatmap.
    """
    # --- 1. DATA HYGIENE ---
    # CICIDS datasets often contain Infinity or NaN in packet rate columns
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()

    # --- 2. LABEL ENCODING ---
    # Converts 'BENIGN' -> 0, 'DDoS' -> 1, etc.
    le = LabelEncoder()
    # We must encode before the heatmap so 'Label' can be calculated in the matrix
    df[target_col] = le.fit_transform(df[target_col])

    # --- 3. DYNAMIC DIMENSIONALITY REDUCTION ---
    # Put our new function to work! 
    # We set it to 0.98 to account for tiny floating-point math differences
    df = drop_redundant_features(df, threshold=0.98)
    
    # --- 4. FEATURE SCALING (Normalization) ---
    # Required for 'Correct handling of normalization' 
    scaler = MinMaxScaler()
    
    # Separate features from target for scaling
    X = df.drop(columns=[target_col])
    y = df[target_col].reset_index(drop=True)
    
    # Scale only numerical features to the range [0, 1] 
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # --- 5. REASSEMBLY ---
    # Combine back into one DF so the EDA Engine can see everything together
    processed_df = pd.concat([X_scaled, y], axis=1)
    
    print(f"   -> Data Cleaning Complete: Features normalized and '{target_col}' encoded.")
    return processed_df, le.classes_