from enum import Enum
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session, select
import os
import openai

# =====================
# ENUMS
# =====================
class Sex(str, Enum):
    M = "M"
    F = "F"
    X = "X"

class Activity(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"

class Goal(str, Enum):
    loss = "loss"
    maintenance = "maintenance"
    gain = "gain"
    recomposition = "recomposition"

class Diet(str, Enum):
    omnivore = "omnivore"
    vegetarian = "vegetarian"
    vegan = "vegan"
    keto = "keto"
    paleo = "paleo"
    other = "other"

# =====================
# DATABASE MODEL
# =====================
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    age: int
    sex: Sex = Field(default=Sex.M)
    height_cm: int
    weight_kg: float
    activity: Activity = Field(default=Activity.moderate)
    goal: Goal = Field(default=Goal.recomposition)
    diet: Diet = Field(default=Diet.omnivore)
    allergies: Optional[str] = None
    preferences: Optional[str] = None

# =====================
# DATABASE SETUP
# =====================
DATABASE_URL = "sqlite:///./fitness.db"
engine = create_engine(DATABASE_URL, echo=True)
SQLModel.metadata.create_all(engine)

# =====================
# FASTAPI INIT
# =====================
app = FastAPI()

# =====================
# OPENAI CONFIG
# =====================
openai.api_key = os.getenv("OPENAI_API_KEY")

class RecommendationRequest(BaseModel):
    user_id: int

@app.post("/users/", response_model=User)
def create_user(user: User):
    with Session(engine) as session:
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

@app.get("/users/", response_model=List[User])
def list_users():
    with Session(engine) as session:
        users = session.exec(select(User)).all()
        return users

@app.post("/recommendations/")
def get_recommendations(data: RecommendationRequest):
    with Session(engine) as session:
        user = session.get(User, data.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        prompt = f"""
        Eres un asesor de fitness y nutrición.
        Crea un plan de entrenamiento y dieta basado en:
        Edad: {user.age}
        Sexo: {user.sex}
        Altura: {user.height_cm} cm
        Peso: {user.weight_kg} kg
        Nivel de actividad: {user.activity}
        Objetivo: {user.goal}
        Dieta: {user.diet}
        Alergias: {user.allergies}
        Preferencias: {user.preferences}
        Incluye calorías diarias recomendadas, macros y un ejemplo de comidas y ejercicios.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Eres un experto en nutrición y entrenamiento."},
                          {"role": "user", "content": prompt}],
                max_tokens=800
            )
            result = response.choices[0].message["content"]
            return {"plan": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
