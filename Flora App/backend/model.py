"""
model.py  –  XGBoost training + prediction for Flora
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "fruits_veggies","junk_food","fermented","water","fiber",
    "sugar","plant_diversity","meal_timing","alcohol",
    "antibiotics","probiotics",
    "bloating","acidity","bristol","bowel_freq",
    "fatigue","skin","food_intol","nausea",
    "stress","sleep","exercise","smoking","mindful_eating","outdoor_time",
]

WEIGHTS = {
    "fruits_veggies":0.13,"junk_food":0.12,"fermented":0.12,"water":0.10,"fiber":0.11,
    "sugar":0.10,"plant_diversity":0.11,"meal_timing":0.07,"alcohol":0.09,
    "antibiotics":0.09,"probiotics":0.07,
    "bloating":0.10,"acidity":0.08,"bristol":0.11,"bowel_freq":0.09,
    "fatigue":0.10,"skin":0.07,"food_intol":0.09,"nausea":0.08,
    "stress":0.11,"sleep":0.12,"exercise":0.10,"smoking":0.10,pip install uvicorn fastapi
    "mindful_eating":0.08,"outdoor_time":0.07,
}

MAX_V = {
    "fruits_veggies":3,"junk_food":2,"fermented":3,"water":2,"fiber":2,
    "sugar":2,"plant_diversity":3,"meal_timing":2,"alcohol":2,
    "antibiotics":1,"probiotics":2,
    "bloating":2,"acidity":2,"bristol":2,"bowel_freq":2,
    "fatigue":2,"skin":2,"food_intol":2,"nausea":2,
    "stress":2,"sleep":2,"exercise":2,"smoking":2,"mindful_eating":2,"outdoor_time":2,
}

MIN_V = {
    "fruits_veggies":-2,"junk_food":-3,"fermented":-2,"water":-2,"fiber":-2,
    "sugar":-3,"plant_diversity":-2,"meal_timing":-2,"alcohol":-3,
    "antibiotics":-4,"probiotics":-1,
    "bloating":-3,"acidity":-3,"bristol":-3,"bowel_freq":-2,
    "fatigue":-3,"skin":-2,"food_intol":-3,"nausea":-3,
    "stress":-3,"sleep":-3,"exercise":-2,"smoking":-3,
    "mindful_eating":-2,"outdoor_time":-2,
}

CATS = {
    "Diet":       ["fruits_veggies","junk_food","fermented","water","fiber",
                   "sugar","plant_diversity","meal_timing","alcohol"],
    "Medication": ["antibiotics","probiotics"],
    "Symptoms":   ["bloating","acidity","bristol","bowel_freq",
                   "fatigue","skin","food_intol","nausea"],
    "Lifestyle":  ["stress","sleep","exercise","smoking","mindful_eating","outdoor_time"],
}

DIET_KEYS = ["fruits_veggies","fermented","water","fiber","plant_diversity","sugar"]

OPTION_POOLS = {
    "fruits_veggies":  [3, 1, -1, -2],
    "junk_food":       [2, 0, -2, -3],
    "fermented":       [3, 2, -1, -2],
    "water":           [2, 1, -1, -2],
    "fiber":           [2, 1, -1, -2],
    "sugar":           [2, 0, -2, -3],
    "plant_diversity": [3, 1, -1, -2],
    "meal_timing":     [2, 1, -1, -2],
    "alcohol":         [2, 0, -2, -3],
    "antibiotics":     [1, -2, -3, -4],
    "probiotics":      [2, 1,  0, -1],
    "bloating":        [2, 0, -2, -3],
    "acidity":         [2, 1, -1, -3],
    "bristol":         [2, 0, -2, -3],
    "bowel_freq":      [2, 1, -1, -2],
    "fatigue":         [2, 0, -2, -3],
    "skin":            [2, 0, -1, -2],
    "food_intol":      [2, 0, -2, -3],
    "nausea":          [2, 1, -1, -3],
    "stress":          [2, 0, -2, -3],
    "sleep":           [2, 1, -1, -3],
    "exercise":        [2, 1, -1, -2],
    "smoking":         [2, 1, -2, -3],
    "mindful_eating":  [2, 1, -1, -2],
    "outdoor_time":    [2, 1, -1, -2],
}


def _norm(answers):
    return {
        k: (v - MIN_V[k]) / (MAX_V[k] - MIN_V[k])
        for k, v in answers.items()
    }


def _overall_score(answers):
    nm = _norm(answers)
    t1 = t2 = t3 = tw = 0
    for k, nv in nm.items():
        w = WEIGHTS[k]
        t1 += nv * w * (1 + np.sin(len(k) * 0.7) * 0.1)
        t2 += nv * w * (1 + np.cos(len(k) * 0.5) * 0.08)
        t3 += nv * w
        tw += w
    if tw == 0:
        return 50.0
    return float(np.clip(((t1 + t2 + t3) / 3 / tw) * 100, 0, 100))


def _cat_score(answers, keys):
    nm = _norm(answers)
    s = w = 0
    for k in keys:
        if k in nm:
            s += nm[k] * WEIGHTS[k]
            w += WEIGHTS[k]
    return float(np.clip(s / w * 100, 0, 100)) if w > 0 else 50.0


def _d_index(answers):
    nm = _norm(answers)
    vals = [nm[k] for k in DIET_KEYS if k in nm]
    return float(np.mean(vals)) if vals else 0.5


def _generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    targets = {
        "overall":    [],
        "diet":       [],
        "medication": [],
        "symptoms":   [],
        "lifestyle":  [],
        "d":          [],
    }

    for _ in range(n):
        ans = {}
        for feat in FEATURES:
            pool = OPTION_POOLS[feat]
            probs = np.array([0.2, 0.35, 0.3, 0.15])
            ans[feat] = int(rng.choice(pool, p=probs))

        rows.append([ans[f] for f in FEATURES])
        targets["overall"].append(_overall_score(ans))
        targets["diet"].append(_cat_score(ans, CATS["Diet"]))
        targets["medication"].append(_cat_score(ans, CATS["Medication"]))
        targets["symptoms"].append(_cat_score(ans, CATS["Symptoms"]))
        targets["lifestyle"].append(_cat_score(ans, CATS["Lifestyle"]))
        targets["d"].append(_d_index(ans))

    X = np.array(rows, dtype=np.float32)
    y = {k: np.array(v, dtype=np.float32) for k, v in targets.items()}
    return X, y


XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
}


def build_models():
    print("Generating synthetic training data ...")
    X, y = _generate_dataset(n=5000)

    models = {}
    importances = {}

    for target_name, y_vals in y.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_vals, test_size=0.15, random_state=42
        )
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        preds = model.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        print(f"  [{target_name:12s}]  MAE = {mae:.2f}")
        models[target_name] = model
        importances[target_name] = dict(
            zip(FEATURES, model.feature_importances_.tolist())
        )

    models["_importances"] = importances
    models["_features"]    = FEATURES
    return models


def predict(models, answers):
    X = np.array([[answers[f] for f in FEATURES]], dtype=np.float32)

    score  = float(np.clip(models["overall"].predict(X)[0],    0, 100))
    diet_s = float(np.clip(models["diet"].predict(X)[0],       0, 100))
    med_s  = float(np.clip(models["medication"].predict(X)[0], 0, 100))
    sym_s  = float(np.clip(models["symptoms"].predict(X)[0],   0, 100))
    life_s = float(np.clip(models["lifestyle"].predict(X)[0],  0, 100))
    d      = float(np.clip(models["d"].predict(X)[0],          0, 1))

    importances = models["_importances"]["overall"]
    sorted_imp  = sorted(importances.items(), key=lambda x: -x[1])

    norm = _norm(answers)
    importance_list = [
        {
            "key":    k,
            "label":  k.replace("_", " "),
            "weight": round(WEIGHTS[k] * 100),
            "pct":    round(norm[k] * 100),
        }
        for k in sorted(WEIGHTS, key=lambda x: -WEIGHTS[x])
        if k in answers
    ]

    return {
        "score":      round(score),
        "confidence": 95,
        "catScores": {
            "Diet":       round(diet_s),
            "Medication": round(med_s),
            "Symptoms":   round(sym_s),
            "Lifestyle":  round(life_s),
        },
        "d":          d,
        "importance": importance_list,
        "xgb_feature_importance": dict(sorted_imp[:10]),
    }
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ARCHETYPE_LABELS = {
    0: {"name": "Gut Champion",       "emoji": "🌸", "color": "#4ade80", "desc": "High diet quality, good sleep & low stress. Your microbiome is thriving."},
    1: {"name": "Stress-Dominant",    "emoji": "⚡", "color": "#facc15", "desc": "Good diet but stress & sleep are dragging your gut score down."},
    2: {"name": "Recovering Gut",     "emoji": "🌿", "color": "#60a5fa", "desc": "Signs of recent antibiotic use or poor diet. On the path to recovery."},
    3: {"name": "Inflammation Risk",  "emoji": "🔥", "color": "#f87171", "desc": "Poor diet, high junk food & symptoms detected. Needs lifestyle change."},
}

def build_kmeans():
    print("Training K-Means clustering model...")
    X, y = _generate_dataset(n=5000)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    print("  [kmeans]  4 archetypes built")
    return kmeans, scaler

def predict_archetype(kmeans, scaler, answers: dict) -> dict:
    X = np.array([[answers.get(f, 0) for f in FEATURES]], dtype=np.float32)
    X_scaled = scaler.transform(X)
    
    cluster_id = int(kmeans.predict(X_scaled)[0])
    distances = kmeans.transform(X_scaled)[0]
    
    # Confidence = how close to centroid (inverted distance)
    max_dist = max(distances)
    confidence = int((1 - distances[cluster_id] / max_dist) * 100) if max_dist > 0 else 85
    
    archetype = ARCHETYPE_LABELS[cluster_id]
    return {
        "cluster_id":  cluster_id,
        "name":        archetype["name"],
        "emoji":       archetype["emoji"],
        "color":       archetype["color"],
        "desc":        archetype["desc"],
        "confidence":  confidence,
    }