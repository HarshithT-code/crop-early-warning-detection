"""
dataset.py — Generates synthetic training data for KrishiRakshak EWS.
Run once to produce crop_data.csv used for model validation and analysis.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10_000

crops = [
    "Rice (Paddy)", "Wheat", "Maize", "Cotton", "Tomato",
    "Ragi (Finger Millet)", "Sugarcane", "Groundnut",
    "Soybean", "Mustard", "Chilli", "Onion", "Potato", "Mango"
]
soils = ["Black", "Red", "Alluvial", "Sandy", "Loamy", "Clay"]

data = {
    "crop":        np.random.choice(crops, n),
    "soil":        np.random.choice(soils, n),
    "rainfall":    np.random.randint(0, 500, n).astype(float),
    "temperature": np.random.randint(10, 45, n).astype(float),
    "humidity":    np.random.randint(20, 100, n).astype(float),
    "ndvi":        np.round(np.random.uniform(0.1, 0.9, n), 2),
    "pest_score":  np.random.randint(0, 100, n).astype(float),
}

df = pd.DataFrame(data)

def label(row):
    score = 0
    if row.rainfall < 30 or row.rainfall > 350: score += 1
    if row.temperature > 38 or row.temperature < 12: score += 1
    if row.humidity > 85:  score += 1
    if row.ndvi < 0.3:     score += 1
    if row.pest_score > 70: score += 1
    return 2 if score >= 3 else 1 if score >= 1 else 0

df["risk"] = df.apply(label, axis=1)

risk_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
df["risk_label"] = df["risk"].map(risk_map)

df.to_csv("crop_data.csv", index=False)

print(f"Dataset saved: crop_data.csv ({len(df)} rows)")
print(df["risk_label"].value_counts())
