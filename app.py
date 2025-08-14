import os
import datetime as dt
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlmodel import SQLModel, Field as SQLField, create_engine, Session, select
from dotenv import load_dotenv
import requests

# =========================
# Enumeraciones para SQLModel
# =========================
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

# =========================
# Carga de entorno
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NUTRITION_API_BASE = os.getenv("NUTRITION_API_BASE")  # opcional
NUTRITION_API_KEY = os.getenv("NUTRITION_API_KEY")    # opcional

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY no configurada. Las rutas de IA fallarán hasta que la configures.")

# =========================
# DB setup (SQLite)
# =========================
engine = create_engine("sqlite:///mvp_coach.db")

class User(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    name: str
    age: int
    sex: Sex = Sex.M
    height_cm: int
    weight_kg: float
    activity: Activity = Activity.moderate
    goal: Goal = Goal.recomposition
    diet: Diet = Diet.omnivore
    allergies: Optional[str] = None
    preferences: Optional[str] = None

class FoodLog(SQLModel, table=True):
    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: int = SQLField(index=True)
    date: str = SQLField(index=True)  # YYYY-MM-DD
    food_name: str
    grams: float
    kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float

def init_db():
    SQLModel.metadata.create_all(engine)

# =========================
# Utilidades
# =========================
def get_session():
    with Session(engine) as session:
        yield session

def mifflin_st_jeor(sex, weight_kg: float, height_cm: int, age: int) -> float:
    sex = str(sex)
    if sex == "M":
        return 10*weight_kg + 6.25*height_cm - 5*age + 5
    elif sex == "F":
        return 10*weight_kg + 6.25*height_cm - 5*age - 161
    else:
        return 10*weight_kg + 6.25*height_cm - 5*age - 78

def activity_factor(level) -> float:
    level = str(level)
    return {"low":1.2, "moderate":1.5, "high":1.75}.get(level, 1.5)

def goal_adjustment(goal) -> int:
    goal = str(goal)
    return {"loss": -400, "maintenance": 0, "gain": 300, "recomposition": 0}.get(goal, 0)

def compute_targets(user: User) -> Dict[str, float]:
    bmr = mifflin_st_jeor(user.sex, user.weight_kg, user.height_cm, user.age)
    tdee = bmr * activity_factor(user.activity)
    kcal_target = max(1200, round(tdee + goal_adjustment(user.goal)))
    protein_g = round(2.0 * user.weight_kg)   # 2 g/kg
    fat_g = round(0.9 * user.weight_kg)       # ~0.9 g/kg
    protein_kcal = protein_g * 4
    fat_kcal = fat_g * 9
    carbs_kcal = max(0, kcal_target - protein_kcal - fat_kcal)
    carbs_g = round(carbs_kcal / 4)
    return {"kcal": kcal_target, "protein_g": protein_g, "fat_g": fat_g, "carbs_g": carbs_g}

# =========================
# Modelos Pydantic (I/O)
# =========================
class CreateUser(BaseModel):
    name: str
    age: int
    sex: Sex = Sex.M
    height_cm: int
    weight_kg: float
    activity: Activity = Activity.moderate
    goal: Goal = Goal.recomposition
    diet: Diet = Diet.omnivore
    allergies: Optional[List[str]] = None
    preferences: Optional[List[str]] = None

class FoodEntry(BaseModel):
    food_name: str
    grams: float

class PlanRequest(BaseModel):
    user_id: int
    date: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    override_targets: Optional[Dict[str,float]] = None

# =========================
# Nutrición: stub/USDA hook
# =========================
def fetch_food_nutrients(food_name: str, grams: float) -> Dict[str, float]:
    db = {
        "pechuga de pollo cocida": {"kcal":165, "protein_g":31, "fat_g":3.6, "carbs_g":0},
        "arroz cocido": {"kcal":130, "protein_g":2.4, "fat_g":0.3, "carbs_g":28},
        "avena": {"kcal":389, "protein_g":16.9, "fat_g":6.9, "carbs_g":66.3},
        "huevo": {"kcal":155, "protein_g":13, "fat_g":11, "carbs_g":1.1},
        "manzana": {"kcal":52, "protein_g":0.3, "fat_g":0.2, "carbs_g":14}
    }
    base = db.get(food_name.lower())
    if not base:
        raise HTTPException(status_code=404, detail=f"No hay datos para '{food_name}'.")
    factor = grams / 100.0
    return {
        "kcal": round(base["kcal"]*factor, 1),
        "protein_g": round(base["protein_g"]*factor, 1),
        "fat_g": round(base["fat_g"]*factor, 1),
        "carbs_g": round(base["carbs_g"]*factor, 1),
    }

# =========================
# Proveedor de IA (OpenAI por defecto)
# =========================
class AIProvider:
    def diet_plan(self, user: User, targets: Dict[str,float], constraints: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
        raise NotImplementedError
    def workout_plan(self, user: User, constraints: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
        raise NotImplementedError

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.base = "https://api.openai.com/v1/chat/completions"

    def _chat(self, system: str, user: str) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":user}
            ],
            "temperature": 0.6,
            "response_format": {"type":"json_object"}
        }
        r = requests.post(self.base, json=payload, headers=headers, timeout=60)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")
        return r.json()

    def diet_plan(self, user: User, targets: Dict[str,float], constraints: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
        system = "Eres un nutriólogo deportivo. Responde SIEMPRE en JSON válido."
        cons = constraints or {}
        prompt = f"""
Genera un plan de comidas de 1 día para el usuario:
Sexo: {user.sex}, Edad: {user.age}, Peso: {user.weight_kg}kg, Altura: {user.height_cm}cm
Objetivo: {user.goal}, Dieta: {user.diet}
Alergias: {user.allergies}, Preferencias: {user.preferences}
Restricciones: {cons}
Objetivo nutricional:
kcal: {targets['kcal']}, proteína: {targets['protein_g']}, grasa: {targets['fat_g']}, carbohidratos: {targets['carbs_g']}
Devuelve SOLO JSON con: {{"meals": [...], "notes": ["", ""]}}
"""
        data = self._chat(system, prompt)
        import json
        return json.loads(data["choices"][0]["message"]["content"])

    def workout_plan(self, user: User, constraints: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
        system = "Eres un entrenador personal certificado. Responde SIEMPRE en JSON válido."
        cons = constraints or {}
        prompt = f"""
Diseña un plan de entrenamiento de 1 semana (5-6 días) para:
Sexo: {user.sex}, Edad: {user.age}, Peso: {user.weight_kg}kg, Altura: {user.height_cm}cm
Objetivo: {user.goal}
Restricciones: {cons}
Devuelve SOLO JSON con: {{"days": [...], "general_tips": ["", ""]}}
"""
        data = self._chat(system, prompt)
        import json
        return json.loads(data["choices"][0]["message"]["content"])

ai: AIProvider = OpenAIProvider(api_key=OPENAI_API_KEY)

# =========================
# FastAPI
# =========================
app = FastAPI(title="Fitness+Nutrition MVP", version="0.1.0")

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/users")
def create_user(payload: CreateUser, session: Session = Depends(get_session)):
    user = User(
        name=payload.name,
        age=payload.age,
        sex=payload.sex,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        activity=payload.activity,
        goal=payload.goal,
        diet=payload.diet,
        allergies=",".join(payload.allergies) if payload.allergies else None,
        preferences=",".join(payload.preferences) if payload.preferences else None
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"user_id": user.id}

@app.get("/users/{user_id}/targets")
def get_targets(user_id: int, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return compute_targets(user)

@app.post("/food/log/{user_id}")
def log_food(user_id: int, item: FoodEntry, date: Optional[str] = None, session: Session = Depends(get_session)):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    date_str = date or dt.date.today().isoformat()
    nutrients = fetch_food_nutrients(item.food_name, item.grams)
    entry = FoodLog(user_id=user_id, date=date_str, food_name=item.food_name, grams=item.grams, **nutrients)
    session.add(entry)
    session.commit()
    return {"log_id": entry.id, "added": nutrients}

@app.get("/diary/{user_id}/{date}")
def day_diary(user_id: int, date: str, session: Session = Depends(get_session)):
    items = session.exec(select(FoodLog).where(FoodLog.user_id==user_id, FoodLog.date==date)).all()
    if not items:
        raise HTTPException(404, "No records for this date")
    totals = {"kcal":0.0, "protein_g":0.0, "fat_g":0.0, "carbs_g":0.0}
    out_items = []
    for it in items:
        for k in totals: totals[k] += getattr(it, k)
        out_items.append(it.dict())
    for k in totals: totals[k] = round(totals[k],1)
    return {"date": date, "items": out_items, "totals": totals}

@app.post("/plan/diet")
def generate_diet_plan(req: PlanRequest, session: Session = Depends(get_session)):
    user = session.get(User, req.user_id)
    if not user:
        raise HTTPException(404, "User not found")
    targets = req.override_targets or compute_targets(user)
    return {"targets": targets, "plan": ai.diet_plan(user, targets, req.constraints)}

@app.post("/plan/workout")
def generate_workout_plan(req: PlanRequest, session: Session = Depends(get_session)):
    user = session.get(User, req.user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return {"plan": ai.workout_plan(user, req.constraints)}

@app.get("/food/search")
def food_search(q: str = Query(..., min_length=2)):
    catalog = ["pechuga de pollo cocida","arroz cocido","avena","huevo","manzana"]
    matches = [f for f in catalog if q.lower() in f.lower()]
    return {"query": q, "results": matches}
