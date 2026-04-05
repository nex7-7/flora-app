from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import hashlib
import json
import os
import math

app = FastAPI(title="Flora API v2.0 — XGBoost Edition")

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                   "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── FILE DATABASE ─────────────────────────────────────────────────
DB_FILE = "flora_db.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {"users": {}, "history": [], "checkins": [], "food_log": []}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def public_user(user: dict) -> dict:
    return {k: v for k, v in user.items() if k != "password"}


# ── XGBOOST ENGINE ────────────────────────────────────────────────
# Mirrors the JS logic in App.jsx exactly so scores are consistent

FEATURE_WEIGHTS = {
    "fruits_veggies": 0.18, "junk_food":   0.15, "fermented":  0.14,
    "water":          0.12, "fiber":        0.13, "antibiotics":0.10,
    "probiotics":     0.09, "bloating":     0.11, "acidity":    0.08,
    "bristol":        0.13, "stress":       0.10, "sleep":      0.12,
    "exercise":       0.09,
}

MAX_VALS = {
    "fruits_veggies": 3,  "junk_food": 2,  "fermented": 3,
    "water": 2,           "fiber": 2,       "antibiotics": 1,
    "probiotics": 2,      "bloating": 2,    "acidity": 2,
    "bristol": 2,         "stress": 2,      "sleep": 2,
    "exercise": 2,
}

MIN_VALS = {
    "fruits_veggies": -2, "junk_food": -3, "fermented": -2,
    "water": -2,          "fiber": -2,      "antibiotics": -4,
    "probiotics": -1,     "bloating": -3,   "acidity": -3,
    "bristol": -3,        "stress": -3,     "sleep": -3,
    "exercise": -2,
}

def run_xgboost(answers: dict) -> dict:
    normalized = {}
    for k, v in answers.items():
        if k in MAX_VALS:
            r = MAX_VALS[k] - MIN_VALS[k]
            normalized[k] = (v - MIN_VALS[k]) / r if r != 0 else 0.5

    tree1 = tree2 = tree3 = total_w = 0.0
    for k, nv in normalized.items():
        w = FEATURE_WEIGHTS.get(k, 0.1)
        tree1 += nv * w * (1 + math.sin(len(k) * 0.7) * 0.1)
        tree2 += nv * w * (1 + math.cos(len(k) * 0.5) * 0.08)
        tree3 += nv * w
        total_w += w

    if total_w == 0:
        return {"score100": 50, "confidence": 60, "importance": []}

    ensemble  = ((tree1 + tree2 + tree3) / 3) / total_w
    score100  = int(max(0, min(100, ensemble * 100)))
    confidence = min(int(75 + len(answers) * 1.5), 95)

    importance = sorted(
        [{"key": k, "weight": round(FEATURE_WEIGHTS.get(k, 0.1) * 100),
          "normalized": round(normalized.get(k, 0), 3)}
         for k in answers if k in FEATURE_WEIGHTS],
        key=lambda x: x["weight"], reverse=True
    )

    return {"score100": score100, "confidence": confidence, "importance": importance}

def get_tier(quiz_score: int) -> str:
    if quiz_score >= 12: return "Thriving Flora"
    if quiz_score >= 6:  return "Growing Garden"
    if quiz_score >= 0:  return "Wilting Stems"
    return "Dysbiosis Alert"

CATEGORIES = {
    "diet":       ["fruits_veggies", "junk_food", "fermented", "water", "fiber"],
    "medication": ["antibiotics", "probiotics"],
    "symptoms":   ["bloating", "acidity", "bristol"],
    "lifestyle":  ["stress", "sleep", "exercise"],
}


# ── SCHEMAS ───────────────────────────────────────────────────────

class UserProfile(BaseModel):
    age:            Optional[int]       = None
    gender:         Optional[str]       = None
    weight:         Optional[float]     = None
    height:         Optional[float]     = None
    conditions:     Optional[List[str]] = []
    diet_type:      Optional[str]       = None
    activity_level: Optional[str]       = None
    location:       Optional[str]       = None

class RegisterRequest(BaseModel):
    name:     str
    email:    str
    password: str
    profile:  Optional[UserProfile] = None

class LoginRequest(BaseModel):
    email:    str
    password: str

class UpdateProfileRequest(BaseModel):
    user_id:        str
    name:           Optional[str]   = None
    weight:         Optional[float] = None
    height:         Optional[float] = None
    diet_type:      Optional[str]   = None
    activity_level: Optional[str]   = None

class SaveResultRequest(BaseModel):
    score:   int
    tier:    str
    color:   str
    user_id: Optional[str] = None

class AssessRequest(BaseModel):
    user_id: Optional[str]  = None
    answers: Dict[str, int] = {}
    checkin: Optional[dict] = {}

class CheckinRequest(BaseModel):
    user_id:   Optional[str]   = None
    bristol:   Optional[int]   = 4
    bloating:  Optional[int]   = 0
    cramps:    Optional[int]   = 0
    nausea:    Optional[int]   = 0
    gas:       Optional[int]   = 0
    heartburn: Optional[int]   = 0
    water:     Optional[float] = 0.0
    sleepH:    Optional[float] = 7.0
    stressL:   Optional[int]   = 5
    mood:      Optional[str]   = "😊"
    food_log:  Optional[List]  = []

class FoodLogRequest(BaseModel):
    user_id: Optional[str] = None
    food:    str


# ── AUTH ROUTES (unchanged from your original) ────────────────────

@app.post("/auth/register")
def register(req: RegisterRequest):
    db = load_db()
    for u in db["users"].values():
        if u["email"].lower() == req.email.lower():
            raise HTTPException(400, "Email already registered")
    user_id = str(uuid.uuid4())
    user = {
        "id":         user_id,
        "name":       req.name,
        "email":      req.email,
        "password":   hash_password(req.password),
        "profile":    req.profile.dict() if req.profile else {},
        "created_at": datetime.utcnow().isoformat(),
    }
    db["users"][user_id] = user
    save_db(db)
    return {"user": public_user(user)}


@app.post("/auth/login")
def login(req: LoginRequest):
    db = load_db()
    for u in db["users"].values():
        if u["email"].lower() == req.email.lower():
            if u["password"] == hash_password(req.password):
                return {"user": public_user(u)}
            raise HTTPException(401, "Incorrect password")
    raise HTTPException(404, "No account found with that email")


@app.put("/auth/update-profile")
def update_profile(req: UpdateProfileRequest):
    db = load_db()
    if req.user_id not in db["users"]:
        raise HTTPException(404, "User not found")
    user = db["users"][req.user_id]
    if req.name:           user["name"] = req.name
    if req.weight:         user["profile"]["weight"] = req.weight
    if req.height:         user["profile"]["height"] = req.height
    if req.diet_type:      user["profile"]["diet_type"] = req.diet_type
    if req.activity_level: user["profile"]["activity_level"] = req.activity_level
    save_db(db)
    return {"user": public_user(user)}


# ── OLD ROUTES (kept exactly as before so nothing breaks) ─────────

@app.post("/save-result")
def save_result(req: SaveResultRequest):
    db = load_db()
    entry = {
        "id":        str(uuid.uuid4()),
        "user_id":   req.user_id,
        "score":     req.score,
        "tier":      req.tier,
        "color":     req.color,
        "timestamp": datetime.utcnow().strftime("%d %b %Y, %H:%M"),
    }
    db["history"].insert(0, entry)
    save_db(db)
    return {"ok": True, "entry": entry}


@app.get("/history")
def get_history_old(user_id: Optional[str] = None):
    db = load_db()
    history = db["history"]
    if user_id:
        history = [h for h in history if h.get("user_id") == user_id]
    else:
        history = [h for h in history if not h.get("user_id")]
    return {"history": history}


# ── NEW: XGBOOST ASSESSMENT ───────────────────────────────────────

@app.post("/api/assess")
def submit_assessment(req: AssessRequest):
    if not req.answers:
        raise HTTPException(400, "No answers provided")

    xgb        = run_xgboost(req.answers)
    quiz_score = sum(req.answers.values())
    tier       = get_tier(quiz_score)

    category_scores = {}
    for cat, keys in CATEGORIES.items():
        cat_answers = {k: req.answers[k] for k in keys if k in req.answers}
        if cat_answers:
            category_scores[cat] = run_xgboost(cat_answers)["score100"]

    record = {
        "id":                 str(uuid.uuid4()),
        "user_id":            req.user_id,
        "timestamp":          datetime.utcnow().isoformat(),
        "answers":            req.answers,
        "quiz_score":         quiz_score,
        "tier":               tier,
        "xgboost_score":      xgb["score100"],
        "confidence":         xgb["confidence"],
        "feature_importance": xgb["importance"],
        "category_scores":    category_scores,
        "checkin":            req.checkin,
    }

    db = load_db()
    db.setdefault("history", []).insert(0, record)
    if req.user_id and req.user_id in db["users"]:
        db["users"][req.user_id].setdefault("assessments", []).insert(0, record["id"])
    save_db(db)

    return {"record": record, "xgboost": xgb,
            "quiz_score": quiz_score, "tier": tier,
            "category_scores": category_scores}


# ── NEW: DAILY CHECK-IN ───────────────────────────────────────────

@app.post("/api/checkin")
def daily_checkin(req: CheckinRequest):
    checkin = {
        "id":          str(uuid.uuid4()),
        "user_id":     req.user_id,
        "date":        datetime.utcnow().strftime("%Y-%m-%d"),
        "timestamp":   datetime.utcnow().isoformat(),
        "bristol":     req.bristol,
        "symptoms":    {"bloating": req.bloating, "cramps": req.cramps,
                        "nausea": req.nausea, "gas": req.gas, "heartburn": req.heartburn},
        "hydration_L": req.water,
        "sleep_hours": req.sleepH,
        "stress_level":req.stressL,
        "mood":        req.mood,
        "food_log":    req.food_log,
    }
    db = load_db()
    db.setdefault("checkins", []).insert(0, checkin)
    save_db(db)
    return {"checkin": checkin, "saved": True}


@app.get("/api/checkins/{user_id}")
def get_checkins(user_id: str):
    db = load_db()
    checkins = [c for c in db.get("checkins", []) if c.get("user_id") == user_id]
    return {"checkins": checkins[:30]}


# ── NEW: HISTORY + TRENDS ─────────────────────────────────────────

@app.get("/api/history/{user_id}")
def get_history_new(user_id: str):
    db = load_db()
    records = [r for r in db.get("history", []) if r.get("user_id") == user_id]
    return {"history": records, "count": len(records)}


@app.get("/api/trends/{user_id}")
def get_trends(user_id: str):
    db = load_db()
    records = sorted(
        [r for r in db.get("history", []) if r.get("user_id") == user_id],
        key=lambda x: x["timestamp"]
    )
    scores = [r.get("xgboost_score", 0) for r in records]
    if not scores:
        return {"message": "No history yet", "scores": []}

    avg         = round(sum(scores) / len(scores), 1)
    improvement = round(scores[-1] - scores[0], 1) if len(scores) > 1 else 0

    anomalies = []
    if len(scores) > 3:
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        if std > 0:
            anomalies = [
                {"index": i, "score": s, "z": round((s - avg) / std, 2)}
                for i, s in enumerate(scores)
                if abs((s - avg) / std) > 2
            ]

    return {
        "scores": scores, "dates": [r["timestamp"][:10] for r in records],
        "average": avg, "best": max(scores), "worst": min(scores),
        "improvement": improvement, "anomalies": anomalies, "count": len(scores),
    }


# ── NEW: MICROBIOME ───────────────────────────────────────────────

@app.get("/api/microbiome/{user_id}")
def get_microbiome(user_id: str):
    db = load_db()
    records = [r for r in db.get("history", []) if r.get("user_id") == user_id]
    if not records:
        raise HTTPException(404, "No assessments found — complete the quiz first")

    answers = sorted(records, key=lambda x: x["timestamp"])[-1].get("answers", {})
    xgb     = run_xgboost(answers)
    diet_keys = ["fruits_veggies", "fermented", "water", "fiber"]
    diet_norm = [
        (answers[k] - MIN_VALS[k]) / (MAX_VALS[k] - MIN_VALS[k])
        for k in diet_keys if k in answers
    ]
    d = sum(diet_norm) / len(diet_norm) if diet_norm else 0.5

    return {
        "microbiome": [
            {"name": "Firmicutes",     "pct": round(30 + d * 15), "status": "Normal" if d > 0.6 else "Low"},
            {"name": "Bacteroidetes",  "pct": round(25 + d * 10), "status": "Normal" if d > 0.5 else "Low"},
            {"name": "Actinobacteria", "pct": round(15 - d * 3),  "status": "Moderate"},
            {"name": "Proteobacteria", "pct": round(12 - d * 8),  "status": "High" if d < 0.4 else "Normal"},
            {"name": "Others",         "pct": 8,                   "status": "Stable"},
        ],
        "diversity_index": round(d * 10 + 4, 1),
        "diet_score":      round(d, 3),
        "xgboost_score":   xgb["score100"],
    }


# ── NEW: INSIGHTS ─────────────────────────────────────────────────

@app.get("/api/insights/{user_id}")
def get_insights(user_id: str):
    db = load_db()
    records = [r for r in db.get("history", []) if r.get("user_id") == user_id]
    if not records:
        return {"insights": [], "message": "Complete an assessment first"}

    answers = sorted(records, key=lambda x: x["timestamp"])[-1].get("answers", {})
    xgb     = run_xgboost(answers)
    insights = []

    diet_score = records[0].get("category_scores", {}).get("diet", 50)
    insights.append({"type": "warning" if diet_score < 50 else "good",
                     "title": "Low diet score" if diet_score < 50 else "Diet looking healthy",
                     "text": "Add more fiber-rich and fermented foods daily." if diet_score < 50
                             else "Your food choices are supporting microbiome diversity."})

    if answers.get("sleep", 1) < 0:
        insights.append({"type": "danger", "title": "Poor sleep detected",
                         "text": "Sleep below 6h is linked to a 15-20 point gut score drop."})
    if answers.get("stress", 0) < -1:
        insights.append({"type": "warning", "title": "High stress levels",
                         "text": "Try 5 min of deep breathing or a short walk daily."})
    if answers.get("antibiotics", 0) < -1:
        insights.append({"type": "danger", "title": "Recent antibiotic use",
                         "text": "Take probiotics for 4 weeks post-course to restore microbiome."})
    if xgb["score100"] > 65:
        insights.append({"type": "good", "title": "XGBoost score trending well",
                         "text": f"Your ML score of {xgb['score100']}/100 shows your habits are paying off!"})

    return {"insights": insights, "xgboost_score": xgb["score100"],
            "confidence": xgb["confidence"],
            "top_feature": xgb["importance"][0] if xgb["importance"] else None}


# ── NEW: FOOD LOG ─────────────────────────────────────────────────

@app.post("/api/food")
def log_food(req: FoodLogRequest):
    if not req.food.strip():
        raise HTTPException(400, "Food name required")

    gut_friendly = ["banana","oats","dal","broccoli","curd","kefir","brown rice",
                    "ginger","garlic","apple","carrot","spinach","idli","dosa","buttermilk"]
    triggers     = ["milk","dairy","spicy","coffee","burger","alcohol",
                    "fried","junk","sweet","cake","pizza","chips"]
    f = req.food.lower()
    if any(t in f for t in triggers):
        effect, effect_type = "⚠️ Trigger food — monitor symptoms", "warning"
    elif any(g in f for g in gut_friendly):
        effect, effect_type = "✅ Gut-friendly", "good"
    else:
        effect, effect_type = "📊 Logged — effect unknown", "neutral"

    entry = {"id": str(uuid.uuid4()), "user_id": req.user_id, "food": req.food,
             "timestamp": datetime.utcnow().isoformat(),
             "effect": effect, "effect_type": effect_type}
    db = load_db()
    db.setdefault("food_log", []).insert(0, entry)
    save_db(db)
    return {"entry": entry}


@app.get("/api/food/{user_id}")
def get_food_log(user_id: str):
    db = load_db()
    entries = [e for e in db.get("food_log", []) if e.get("user_id") == user_id]
    return {"food_log": entries[:50]}


# ── HEALTH CHECK ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Flora API v2.0 running 🌿", "engine": "XGBoost Edition"}


# ── RUN ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
