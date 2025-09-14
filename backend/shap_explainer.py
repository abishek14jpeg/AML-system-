# backend/shap_explainer.py
import shap
import pandas as pd
import joblib
import os

# Load the model and the EXACT training data
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
X_TRAIN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'X_train.csv')

model = joblib.load(MODEL_PATH)
X_train_background = pd.read_csv(X_TRAIN_PATH)

# --- FIX FOR InvalidMaskerError ---
# 1. Summarize the data using kmeans
background_summary_obj = shap.kmeans(X_train_background, 100)

# 2. Convert the legacy DenseData object back into a proper Pandas DataFrame
background_summary_df = pd.DataFrame(
    background_summary_obj.data,
    columns=X_train_background.columns
)

# 3. Create the explainer using the correctly formatted DataFrame
explainer = shap.TreeExplainer(model, background_summary_df)


def get_shap_explanation(transaction_df):
    """Generates SHAP values for a single transaction."""
    shap_values = explainer.shap_values(transaction_df)
    
    # Handle both binary and multi-class cases
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # Multi-class case: use the fraud class (class 1)
        explanation = dict(zip(transaction_df.columns, shap_values[1][0]))
    else:
        # Binary case: shap_values is a single array
        # For binary classification, shap_values[0] contains values for both classes
        # We want the values for the fraud class (class 1)
        if len(shap_values[0].shape) > 1 and shap_values[0].shape[1] > 1:
            # If it's a 2D array, take the second column (class 1)
            explanation = dict(zip(transaction_df.columns, shap_values[0][:, 1]))
        else:
            # If it's a 1D array, use it directly
            explanation = dict(zip(transaction_df.columns, shap_values[0]))
    
    # Convert numpy arrays to floats for JSON serialization
    converted_explanation = {}
    for k, v in explanation.items():
        if hasattr(v, 'item') and v.size == 1:
            converted_explanation[k] = float(v.item())
        elif hasattr(v, 'tolist'):
            converted_explanation[k] = float(v.tolist()[0]) if len(v.tolist()) == 1 else v.tolist()
        else:
            converted_explanation[k] = float(v) if isinstance(v, (int, float)) else v
    
    explanation = converted_explanation
    
    return explanation