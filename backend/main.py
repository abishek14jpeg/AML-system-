# backend/main.py

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import os

# --- IMPORTS for Database and Case Management ---
from backend.shap_explainer import get_shap_explanation
from backend.graph_analysis import analyze_account_graph
from backend.db import create_case, get_all_cases, update_case, CaseSchema

app = FastAPI(title="AML Detection API", version="1.0")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
model = None

@app.on_event("startup")
def load_model():
    """Load the ML model when the server starts up."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        model = None

# --- PYDANTIC MODELS for API requests ---

class Transaction(BaseModel):
    amount: float
    sender_account_age: int
    receiver_account_age: int

# --- NEW PYDANTIC MODELS FOR CASE MANAGEMENT ---

class CreateCasePayload(BaseModel):
    transaction_details: dict
    risk_score: float
    explanation: dict

class UpdateCasePayload(BaseModel):
    notes: str
    status: str

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "AML Detection API is running"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """Receives transaction data and returns a fraud prediction."""
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    input_data = pd.DataFrame([transaction.dict()])
    input_data = input_data[['amount', 'sender_account_age', 'receiver_account_age']]
    
    prediction = model.predict(input_data)[0]
    probability = float(model.predict_proba(input_data)[0][1])  # Convert to Python float
    explanation = get_shap_explanation(input_data)
    
    return {
        "prediction": int(prediction),
        "is_fraud": bool(prediction),
        "fraud_probability": float(round(probability, 4)),  # Ensure it's a Python float
        "explanation": explanation
    }

@app.get("/graph_analysis/{account_id}")
def get_graph_analysis(account_id: str):
    """Performs graph analysis for a given account ID."""
    try:
        acc_id_int = int(account_id)
        graph_data = analyze_account_graph(acc_id_int)
        return graph_data
    except ValueError:
        return {"error": "Invalid account ID format."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# --- NEW ENDPOINTS FOR CASE MANAGEMENT ---

@app.post("/cases", response_model=CaseSchema)
async def add_new_case(payload: CreateCasePayload):
    """Creates a new case from a flagged transaction."""
    new_case = await create_case(
        transaction=payload.transaction_details,
        score=payload.risk_score,
        explanation=payload.explanation
    )
    return new_case

@app.get("/cases", response_model=List[CaseSchema])
async def get_all_existing_cases():
    """Retrieves all cases from the database."""
    cases = await get_all_cases()
    return cases

@app.put("/cases/{case_id}")
async def update_existing_case(case_id: str, payload: UpdateCasePayload):
    """Updates the notes and status of a specific case."""
    success = await update_case(case_id, payload.notes, payload.status)
    if success:
        return {"message": "Case updated successfully"}
    return {"error": "Case not found or not updated"}