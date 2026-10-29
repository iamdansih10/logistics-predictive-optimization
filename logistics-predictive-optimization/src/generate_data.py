"""Simulates shipments for a prediction problem: forecast transit time (hours).

Only information available at dispatch time is used as input features:
distance, weight, mode, carrier, hub, weekday, month, rain forecast and the
expected load at the origin hub. Relationships are deliberately non-linear
(hub congestion kicks in above a threshold; rain hurts road more than rail)
so that model choice actually matters.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
N = 5000
date = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
month, dow = date.month.values, date.dayofweek.values

mode = rng.choice(["Road", "Rail", "Air"], N, p=[.62, .26, .12])
carrier = rng.choice(["A", "B", "C", "D"], N, p=[.32, .28, .22, .18])
hub = rng.choice(["Hub1", "Hub2", "Hub3", "Hub4", "Hub5"], N, p=[.22, .2, .24, .18, .16])
base = {"Road": 450, "Rail": 900, "Air": 1400}
distance = np.array([max(30, rng.gamma(2.2, base[m] / 2.2)) for m in mode]).round(0)
weight = (rng.gamma(2.0, 180, N) + 15).round(1)

monsoon = np.isin(month, [6, 7, 8, 9])
rain = (rng.random(N) < np.where(monsoon, .35, .08)).astype(int)
hub_load = (45 + np.where(month >= 10, 22, 0) + np.where(dow == 0, 10, 0)
            + np.where(hub == "Hub3", 14, 0) + rng.normal(0, 7, N)).round(0)

speed = np.array([{"Road": 45, "Rail": 38, "Air": 400}[m] for m in mode])
carrier_eff = np.array([{"A": 0, "B": 0.3, "C": 4.0, "D": 1.0}[c] for c in carrier])
congestion = 0.16 * np.maximum(0, hub_load - 58)
rain_eff = rain * np.where(mode == "Road", 2.0 + 0.004 * distance, np.where(mode == "Rail", 0.8, 0.3))
transit = (distance / speed + 4 + congestion + 0.004 * weight + carrier_eff
           + rain_eff + np.where(dow == 0, 1.0, 0) + rng.normal(0, 2.0, N))
transit = np.clip(transit, 2, None).round(1)

df = pd.DataFrame({"date": date, "month": month, "weekday": dow, "mode": mode, "carrier": carrier,
    "hub": hub, "distance_km": distance, "weight_kg": weight, "rain": rain,
    "hub_load": hub_load, "transit_hours": transit})
df.to_csv("data/shipments.csv", index=False)
print(df.shape, df.transit_hours.describe().round(1).to_dict())
