import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_models(model_results, y_test, target_classes, output_path):
    """
    Evaluates trained models, generates plots, and exports metrics to Excel.
    Fulfills Project 4: Performance Evaluation & ROC Analysis.
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION & METRICS")
    print("="*50)

    # --- NEW: Identify exactly which classes survived the filtering step ---
    # This prevents crashes if a minority class was dropped during preprocessing
    present_labels = np.unique(y_test)
    present_target_names = [target_classes[i] for i in present_labels]

    # 1. Initialize Excel Writer
    excel_file = os.path.join(output_path, "model_evaluation_metrics.xlsx")
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

        plt.figure(figsize=(10, 8)) 

        for model_name, data in model_results.items():
            print(f"\nEvaluating: {model_name}")
            predictions = data['predictions']
            probabilities = data['probabilities']

            # ---------------------------------------------------------
            # 2. Classification Report (Terminal + Excel Export)
            # ---------------------------------------------------------
            print(f"\nClassification Report for {model_name}:")
            # Added labels parameter to map the present classes perfectly
            print(classification_report(
                y_test, predictions, 
                labels=present_labels, target_names=present_target_names, zero_division=0
            ))
            
            report_dict = classification_report(
                y_test, predictions, 
                labels=present_labels, target_names=present_target_names, zero_division=0, output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose().round(4)
            
            safe_sheet_name = model_name[:31].replace(":", "-").replace("/", "-")
            report_df.to_excel(writer, sheet_name=safe_sheet_name)

            # ---------------------------------------------------------
            # 3. Confusion Matrix
            # ---------------------------------------------------------
            # Force the confusion matrix to only plot the labels that exist
            cm = confusion_matrix(y_test, predictions, labels=present_labels)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=present_target_names, yticklabels=present_target_names)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.ylabel('Actual Label (Ground Truth)')
            plt.xlabel('Predicted Label (AI Guess)')
            plt.tight_layout()
            
            cm_filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(os.path.join(output_path, cm_filename))
            plt.close() 

            # ---------------------------------------------------------
            # 4. ROC Curve Preparation
            # ---------------------------------------------------------
            y_test_binary = (y_test > 0).astype(int)
            y_prob_attack = 1.0 - probabilities[:, 0]

            fpr, tpr, _ = roc_curve(y_test_binary, y_prob_attack)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        # ---------------------------------------------------------
        # 5. Finalize Multi-Classifier ROC Overlay
        # ---------------------------------------------------------
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Fall-Out)')
        plt.ylabel('True Positive Rate (Recall / Sensitivity)')
        plt.title('Multi-Classifier ROC Overlay (Benign vs. Malicious)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "multi_classifier_roc_overlay.png"))
        plt.close()

    print(f"\n   -> Metrics exported to Excel: {excel_file}")
    print(f"   -> All evaluation artifacts saved to {output_path}")