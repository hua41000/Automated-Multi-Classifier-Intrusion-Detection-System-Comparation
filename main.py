import os
import glob
import pandas as pd
from src.data_loader import load_and_sample_data
from src.eda_engine import perform_eda
from src.data_cleaning import clean_and_prepare_df
from sklearn.model_selection import train_test_split
from src.trainer import run_all_models
from src.evaluator import evaluate_models

# These will be created in the next steps
# from src.processor import preprocess_data
# from src.trainer import train_with_sensitivity
# from src.evaluator import generate_roc_analysis

def main():
    # 1. Project Configuration
    data_folder = "data/"
    output_base_dir = "outputs/"
    
    # Identify all datasets (NSL-KDD, CICIDS, etc.) 
    dataset_paths = glob.glob(os.path.join(data_folder, "*.csv"))
    
    if not dataset_paths:
        print(f"Error: No datasets found in '{data_folder}'.")
        return

    print(f"Found {len(dataset_paths)} datasets. Initializing IDS Pipeline...")

    for path in dataset_paths:
        # Organize outputs by dataset name to keep reports clean
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        dataset_output_dir = os.path.join(output_base_dir, dataset_name)
        
        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        print(f"\n{'='*50}")
        print(f"STARTING PROCESS: {dataset_name}")
        print(f"\n{'='*50}")

        # 2. DATA LOADING & SAMPLING
        # Implements the 10% sampling strategy for performance 
        df, target_col = load_and_sample_data(path, sample_size=0.1)
        
        print("-> Cleaning Dataset...")
        df_clean, label_classes = clean_and_prepare_df(df, target_col=target_col)

        # 3. EXPLORATORY DATA ANALYSIS (EDA)
        # Fulfills Project 4's deliverable for feature correlation studies 
        print("-> Generating EDA visualizations...")
        perform_eda(df_clean, output_path=dataset_output_dir, target_col="Label")

        # ---------------------------------------------------------
        # 4. PREPARATION FOR TRAINING
        # ---------------------------------------------------------
        print("-> Splitting data for model training...")
        
        # Step A: Separate features (X) from the target answer (y)
        X = df_clean.drop(columns=["Label"])
        y = df_clean["Label"]
        
        # --- 1st SAFETY SWITCH (Baseline Days) ---
        if len(y.unique()) < 2:
            print(f"   -> SKIPPING AI TRAINING: '{dataset_name}' contains only 1 class ({label_classes[0]}).")
            continue 
            
        # --- 2nd SAFETY SWITCH (Rare Class Filter) ---
        # Find any classes that have fewer than 2 instances
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index
        
        if len(rare_classes) > 0:
            # Drop the rows belonging to these rare classes
            valid_indices = ~y.isin(rare_classes)
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Print a clean warning using the actual attack names from label_classes
            dropped_names = [label_classes[i] for i in rare_classes]
            print(f"   -> WARNING: Dropped unlearnable classes with < 2 samples: {dropped_names}")
        
        # Step B: Perform the Train-Test Split (70% Training, 30% Testing)
        # 'stratify=y' ensures the exact same ratio of attacks-to-benign traffic 
        # exists in both the training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # ---------------------------------------------------------
        # 5. MODEL TRAINING & SENSITIVITY 
        # ---------------------------------------------------------
        print("-> Training models...")
        # We will build run_all_models in src/trainer.py next
        model_results = run_all_models(X_train, y_train, X_test)

        # ---------------------------------------------------------
        # 6. ROC & PERFORMANCE ANALYSIS
        # ---------------------------------------------------------
        print("-> Generating Evaluation Metrics and ROC Curves...")
        
        # Pass the AI's predictions and the actual answer key (y_test) to the grader
        evaluate_models(
            model_results=model_results, 
            y_test=y_test, 
            target_classes=label_classes, 
            output_path=dataset_output_dir
        )

    print(f"\n{'='*50}")
    print("IDS Pipeline Execution Complete.")
    print(f"All artifacts stored in: {output_base_dir}")

if __name__ == "__main__":
    main()