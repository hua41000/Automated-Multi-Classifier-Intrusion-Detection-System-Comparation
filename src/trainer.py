import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# --- 1. Specific Model Functions ---

def train_logistic_regression(X_train, y_train):
    """Trains a baseline Logistic Regression model."""
    print("   -> Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_svm(X_train, y_train):
    """
    Trains a Support Vector Machine.
    Uses an RBF kernel to capture non-linear relationships in network traffic.
    """
    print("   -> Training Support Vector Machine (SVM)...")
    # 'probability=True' is MANDATORY here. Without it, SVM cannot output 
    # the probabilities needed for your ROC curves later.
    svm_model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_neural_network(X_train, y_train):
    """
    Trains a Multi-Layer Perceptron (Neural Network).
    """
    print("   -> Training Neural Network (MLP)...")
    # MLPClassifier doesn't have a built-in 'class_weight' parameter.
    # To prevent it from overfitting on the majority BENIGN class, we use 'early_stopping'.
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), # Two hidden layers for feature extraction
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,         # Stops training if validation score stops improving
        random_state=42
    )
    
    # Neural Networks can sometimes throw convergence warnings. 
    # This keeps your console output clean during the pipeline run.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn_model.fit(X_train, y_train)
        
    return nn_model


# --- 2. The Orchestrator Function ---

def run_all_models(X_train, y_train, X_test):
    """
    Runs all required classifiers and collects their probabilities and predictions.
    Outputs a structured dictionary perfect for downstream ROC and F1-score evaluation.
    """
    results = {}
    
    # 1. Train and Evaluate Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    results['Logistic Regression'] = {
        'model': lr_model,
        'probabilities': lr_model.predict_proba(X_test),
        'predictions': lr_model.predict(X_test)
    }
    
    # 2. Train and Evaluate SVM
    svm_model = train_svm(X_train, y_train)
    results['SVM'] = {
        'model': svm_model,
        'probabilities': svm_model.predict_proba(X_test),
        'predictions': svm_model.predict(X_test)
    }

    # 3. Train and Evaluate Neural Network
    nn_model = train_neural_network(X_train, y_train)
    results['Neural Network'] = {
        'model': nn_model,
        'probabilities': nn_model.predict_proba(X_test), # Will be used for ROC/AUC
        'predictions': nn_model.predict(X_test)          # Will be used for Precision/Recall/F1
    }
    
    print("   -> All models trained and predictions generated successfully.")
    return results