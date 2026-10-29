"""Optimisation: (1) delivery route with 2-opt, (2) carrier allocation with linear programming."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

rng = np.random.default_rng(5)
O = {}

# ================= 1. route optimisation (15 stops, one depot) =================
n = 15
stops = rng.uniform(0, 30, (n, 2)); depot = np.array([15.0, 15.0])
pts = np.vstack([depot, stops])
D = np.linalg.norm(pts[:, None] - pts[None], axis=2)

def length(route):
    r = [0] + list(route) + [0]
    return sum(D[r[i], r[i + 1]] for i in range(len(r) - 1))

given = list(range(1, n + 1))                                   # order the stops arrived in
left, cur, nn = set(given), 0, []
while left:
    nxt = min(left, key=lambda j: D[cur, j]); nn.append(nxt); left.remove(nxt); cur = nxt

def two_opt(route):
    best, improved = route[:], True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if length(cand) < length(best) - 1e-9:
                    best, improved = cand, True
    return best

opt = two_opt(nn)
rand_lens = [length(rng.permutation(given)) for _ in range(500)]
O["route"] = {"stops": n, "as_received_km": round(length(given), 1), "random_avg_km": round(float(np.mean(rand_lens)), 1),
              "nearest_neighbour_km": round(length(nn), 1), "two_opt_km": round(length(opt), 1)}
O["route"]["saving_vs_received_pct"] = round((1 - length(opt) / length(given)) * 100, 1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, route, title in ((ax[0], given, f"As received: {length(given):.0f} km"), (ax[1], opt, f"After 2-opt: {length(opt):.0f} km")):
    r = [0] + list(route) + [0]
    a.plot(pts[r, 0], pts[r, 1], "-o", color="#2b6cb0", ms=4, lw=1.2)
    a.plot(*depot, "s", color="red", ms=9); a.set_title(title); a.set_aspect("equal")
plt.tight_layout(); plt.savefig("outputs/fig4_route.png", dpi=140); plt.close()

# ================= 2. carrier allocation (linear programme) =================
model = joblib.load("outputs/best_model.joblib")
carriers = ["A", "B", "C", "D"]
zones = {"Zone North (400 km)": (400, 250), "Zone Central (700 km)": (700, 200), "Zone South (1100 km)": (1100, 150)}
cap = np.array([220, 200, 120, 160])                       # shipments each carrier can take next week
rate = {"A": 1.00, "B": 0.95, "C": 0.85, "D": 1.10}       # rate card (C is cheapest, but slowest)
zone_cost = {400: 3200, 700: 4800, 1100: 7400}            # Rs base cost per shipment
VALUE_OF_HOUR = 120                                        # ASSUMPTION: Rs per hour of delay

rows, cost, hours = [], [], []
for z, (dist, demand) in zones.items():
    for c in carriers:
        x = pd.DataFrame([{"mode": "Road", "carrier": c, "hub": "Hub3", "distance_km": dist, "weight_kg": 350,
                           "rain": 0, "hub_load": 75, "month": 11, "weekday": 1}])
        h = float(model.predict(x)[0])
        hours.append(h); cost.append(zone_cost[dist] * rate[c])
cost, hours = np.array(cost), np.array(hours)
total_cost = cost + VALUE_OF_HOUR * hours                  # generalised cost
demand = np.array([d for _, d in zones.values()])
nz, nc = len(zones), len(carriers)

A_eq = np.zeros((nz, nz * nc)); A_ub = np.zeros((nc, nz * nc))
for i in range(nz): A_eq[i, i * nc:(i + 1) * nc] = 1
for j in range(nc): A_ub[j, j::nc] = 1
res = linprog(total_cost, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=demand, bounds=(0, None), method="highs")
lp = res.x.reshape(nz, nc).round(0)

# baseline: fill cheapest carrier (by rate card only) first, ignoring delay
base = np.zeros((nz, nc)); capleft = cap.copy()
order = np.argsort([rate[c] for c in carriers])
for i, d in enumerate(demand):
    rem = d
    for j in order:
        take = min(rem, capleft[j]); base[i, j] += take; capleft[j] -= take; rem -= take
        if rem <= 0: break

def summarise(alloc):
    flat = alloc.flatten()
    return {"rate_card_cost": int((flat * cost).sum()), "predicted_hours_total": round(float((flat * hours).sum()), 0),
            "avg_hours_per_shipment": round(float((flat * hours).sum() / flat.sum()), 2),
            "generalised_cost": int((flat * total_cost).sum())}

O["allocation"] = {"demand": int(demand.sum()), "value_of_hour_rs": VALUE_OF_HOUR,
    "cheapest_first": summarise(base), "optimised": summarise(lp),
    "carrier_totals_baseline": dict(zip(carriers, base.sum(axis=0).astype(int).tolist())),
    "carrier_totals_optimised": dict(zip(carriers, lp.sum(axis=0).astype(int).tolist()))}
a = O["allocation"]
a["saving_generalised_pct"] = round((1 - a["optimised"]["generalised_cost"] / a["cheapest_first"]["generalised_cost"]) * 100, 1)
a["hours_saved_per_shipment"] = round(a["cheapest_first"]["avg_hours_per_shipment"] - a["optimised"]["avg_hours_per_shipment"], 2)
O["pred_hours_example"] = {c: round(float(hours[1 * nc + k]), 1) for k, c in enumerate(carriers)}


# ---- sensitivity: how much does the answer depend on my assumed value of an hour? ----
sens = {}
for v in (0, 120, 250, 400, 600):
    tc = cost + v * hours
    r = linprog(tc, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=demand, bounds=(0, None), method="highs")
    x = r.x.reshape(nz, nc)
    sens[v] = {"carrier_C_shipments": int(round(x[:, 2].sum())), "avg_hours": round(float((x.flatten() * hours).sum() / x.sum()), 2),
               "rate_card_cost": int((x.flatten() * cost).sum())}
O["sensitivity"] = sens

json.dump(O, open("outputs/optimization_results.json", "w"), indent=2, default=float)
print(json.dumps(O, indent=2, default=float))
