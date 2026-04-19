import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def perform_eda(df, output_path, target_col):
    """
    Analyzes processed data to identify behavioral signatures.
    Fulfills Project 4: EDA & Feature Analysis (15% Weight).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_Distribution = f"class_distribution_{timestamp}.png"
    filename_Correlation = f"correlation_matrix_{timestamp}.png"
    # 1. Class Distribution (Now using Numerical Labels 0, 1, etc.)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target_col, data=df)
    plt.title(f"Processed Class Distribution: {target_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_Distribution))
    plt.close()
    
    # 2. Strategic Correlation Heatmap
    plt.figure(figsize=(14, 12))
    
    # Calculate correlation specifically against the Target Label
    corr = df.corr()
    
    # Select the Top 15 features most correlated with the Label
    # This directly identifies the 'behavioral signatures' of attacks
    top_features = corr[target_col].abs().sort_values(ascending=False).head(16).index
    
    sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    
    plt.title("Top Feature Correlations (Behavioral Signatures)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename_Correlation))
    plt.close()
    
    print(f"   -> Analytical artifacts saved to {output_path}")