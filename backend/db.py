# backend/db.py

import motor.motor_asyncio
from bson import ObjectId
from pydantic import BaseModel, Field, BeforeValidator
from typing import Optional, List, Annotated
import datetime

MONGO_DETAILS = "mongodb://localhost:27017"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
database = client.aml_system
case_collection = database.get_collection("cases")

# --- UPDATED PYDANTIC SCHEMA ---
class CaseSchema(BaseModel):
    # This is the line we are fixing.
    # We use Annotated[str, BeforeValidator(str)] to tell Pydantic:
    # "Before you validate this field as a string, please run the str() function on it first."
    # This cleanly converts the ObjectId to a string.
    id: Annotated[str, BeforeValidator(str)] = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    
    transaction_details: dict
    risk_score: float
    explanation: dict
    status: str = "Open"
    notes: str = ""
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# --- Database Helper Functions (no changes here) ---

async def create_case(transaction: dict, score: float, explanation: dict) -> dict:
    """Create a new case in the database."""
    case_data = {
        "transaction_details": transaction,
        "risk_score": score,
        "explanation": explanation,
        "status": "Open",
        "notes": "",
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow()
    }
    new_case = await case_collection.insert_one(case_data)
    created_case = await case_collection.find_one({"_id": new_case.inserted_id})
    return CaseSchema(**created_case).dict(by_alias=True)

async def get_all_cases() -> List[dict]:
    """Retrieve all cases from the database."""
    cases = []
    async for case in case_collection.find():
        cases.append(CaseSchema(**case).dict(by_alias=True))
    return cases

async def update_case(case_id: str, notes: str, status: str) -> bool:
    """Update a case's notes and status."""
    case_id_obj = ObjectId(case_id)
    result = await case_collection.update_one(
        {"_id": case_id_obj},
        {"$set": {"notes": notes, "status": status, "updated_at": datetime.datetime.utcnow()}}
    )
    return result.modified_count == 1