"""Predictive modelling: compare models, cross-validate, tune, evaluate, explain."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/shipments.csv", parse_dates=["date"])
cat = ["mode", "carrier", "hub"]
num = ["distance_km", "weight_kg", "rain", "hub_load", "month", "weekday"]
X, y = df[cat + num], df.transit_hours
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

def pipe(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    return Pipeline([("pre", pre), ("m", model)])

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

models = {
    "Baseline (mean)": DummyRegressor(),
    "Linear regression": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(max_depth=8, random_state=0),
    "Random forest": RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
    "Gradient boosting": GradientBoostingRegressor(random_state=0),
}
kf = KFold(5, shuffle=True, random_state=1)
R = {"n_train": len(Xtr), "n_test": len(Xte), "models": {}}
for name, m in models.items():
    p = pipe(m)
    cv = -cross_val_score(p, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error")
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    R["models"][name] = {"cv_rmse": round(cv.mean(), 2), "cv_std": round(cv.std(), 2),
        "test_rmse": round(rmse(yte, pr), 2), "test_mae": round(mean_absolute_error(yte, pr), 2),
        "test_r2": round(r2_score(yte, pr), 3)}
    print(name, R["models"][name])

# ---- hyperparameter tuning (grid search, 3-fold) on gradient boosting ----
grid = {"m__n_estimators": [200, 400], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3, 4]}
gs = GridSearchCV(pipe(GradientBoostingRegressor(random_state=0)), grid, cv=3,
                  scoring="neg_root_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
best = gs.best_estimator_; pr = best.predict(Xte)
R["best_params"] = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
R["tuned"] = {"cv_rmse": round(-gs.best_score_, 2), "test_rmse": round(rmse(yte, pr), 2),
    "test_mae": round(mean_absolute_error(yte, pr), 2), "test_r2": round(r2_score(yte, pr), 3)}
print("tuned", R["tuned"], R["best_params"])
joblib.dump(best, "outputs/best_model.joblib")

# ---- temporal check: train Jan-Sep, test Oct-Dec (the peak season) ----
tr = df.month <= 9
tm = GridSearchCV(pipe(GradientBoostingRegressor(random_state=0)), {k: [v] for k, v in gs.best_params_.items()},
                  cv=3).fit(X[tr], y[tr])
pt = tm.predict(X[~tr])
R["temporal"] = {"rmse": round(rmse(y[~tr], pt), 2), "mae": round(mean_absolute_error(y[~tr], pt), 2),
                 "r2": round(r2_score(y[~tr], pt), 3), "train_rows": int(tr.sum()), "test_rows": int((~tr).sum())}

# ---- permutation importance ----
pi = permutation_importance(best, Xte, yte, n_repeats=10, random_state=0, scoring="neg_root_mean_squared_error")
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
R["importance_rmse_increase"] = imp.round(2).to_dict()

# ---- charts ----
names = list(R["models"]); vals = [R["models"][n]["test_rmse"] for n in names]
fig, ax = plt.subplots(figsize=(8, 4))
cols = ["#a0aec0"] * len(names) + ["#2f855a"]
ax.barh(names + ["Tuned gradient boosting"], vals + [R["tuned"]["test_rmse"]], color=cols)
ax.set(xlabel="Test RMSE (hours) - lower is better", title="Model comparison")
for i, v in enumerate(vals + [R["tuned"]["test_rmse"]]): ax.text(v + .05, i, f"{v:.2f}", va="center")
plt.tight_layout(); plt.savefig("outputs/fig1_model_comparison.png", dpi=140); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].scatter(yte, pr, s=8, alpha=.4); lim = [0, max(yte.max(), pr.max())]
ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual hours", ylabel="Predicted hours", title="Actual vs predicted (test set)")
res = yte - pr
sns.histplot(res, bins=40, kde=True, ax=ax[1], color="#2b6cb0"); ax[1].set(title="Residuals (actual - predicted)", xlabel="hours")
plt.tight_layout(); plt.savefig("outputs/fig2_actual_vs_predicted.png", dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(7, 4))
imp[::-1].plot.barh(ax=ax, color="#dd6b20"); ax.set(xlabel="RMSE increase when feature is shuffled (hours)", title="What drives predicted transit time?")
plt.tight_layout(); plt.savefig("outputs/fig3_importance.png", dpi=140); plt.close()

# ---- what-if: ease Hub 3 congestion by 25% ----
h3 = Xte[Xte.hub == "Hub3"].copy(); base_p = best.predict(h3)
h3["hub_load"] = h3.hub_load * 0.75
R["whatif_hub3"] = {"rows": len(h3), "avg_pred_before": round(base_p.mean(), 2),
                    "avg_pred_after": round(best.predict(h3).mean(), 2)}
R["whatif_hub3"]["saving_hours"] = round(R["whatif_hub3"]["avg_pred_before"] - R["whatif_hub3"]["avg_pred_after"], 2)
json.dump(R, open("outputs/model_results.json", "w"), indent=2, default=float)
print(json.dumps({k: R[k] for k in ["temporal", "importance_rmse_increase", "whatif_hub3"]}, indent=2))
