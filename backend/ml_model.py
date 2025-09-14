# backend/ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier # We are replacing this
from xgboost import XGBClassifier # --- NEW: Import XGBoost ---
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    """This function trains the model and saves both the model and its training data."""
    DATA_PATH = os.path.join('..', 'data', 'synthetic_data.csv')
    MODEL_PATH = os.path.join('..', 'models', 'model.pkl')
    X_TRAIN_PATH = os.path.join('..', 'data', 'X_train.csv')
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    features = ['amount', 'sender_account_age', 'receiver_account_age']
    target = 'is_fraud'
    
    X = df[features]
    y = df[target]
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Saving X_train data to {X_TRAIN_PATH}...")
    X_train.to_csv(X_TRAIN_PATH, index=False)
    
    # --- NEW: Use XGBClassifier instead of RandomForestClassifier ---
    print("Training the XGBoost model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model performance...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model training complete and saved successfully!")

if __name__ == '__main__':
    train_model()