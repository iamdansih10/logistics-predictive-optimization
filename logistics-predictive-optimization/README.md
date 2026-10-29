# Predictive Modeling & Optimization in Logistics (Week 4)

Final task of my Logistics Data Analyst internship. The project predicts shipment
**transit time** with machine learning, then uses the model to make decisions:
a better delivery route and a better split of shipments across carriers.
Earlier weeks: [Week 1](https://github.com/iamdansih10/logistics-delivery-analysis),
[Week 2](https://github.com/iamdansih10/logistics-data-preprocessing),
[Week 3](https://github.com/iamdansih10/logistics-data-visualization).

> **Data note:** simulated (5,000 shipments). I built non-linear effects in on purpose
> (hub congestion above a threshold, rain hurting road more than rail), so scores
> here are better than real data would give. Treat results as a demonstration of method.

## Run it

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/model.py
python src/optimize.py      # needs outputs/best_model.joblib from model.py
```

## What's inside

| File | Purpose |
|---|---|
| `src/generate_data.py` | Simulates shipments (features known at dispatch + transit time) |
| `src/model.py` | Compares 5 models, 5-fold CV, grid-search tuning, test metrics, permutation importance, Hub 3 what-if |
| `src/optimize.py` | 2-opt route improvement; carrier allocation with linear programming; sensitivity check |
| `outputs/` | Charts, `model_results.json`, `optimization_results.json`, trained model |

## Headline results

| Model | Test RMSE (h) | Test R² |
|---|---|---|
| Baseline (mean) | 11.51 | -0.00 |
| Linear regression | 6.07 | 0.72 |
| Random forest | 2.32 | 0.96 |
| Gradient boosting (tuned) | 2.12 | 0.97 |

- Distance and transport mode explain most of the prediction; carrier, hub load and rain come next.
- Easing Hub 3 congestion by 25% cuts predicted transit time by about 1.1 hours per shipment.
- 2-opt shortens a 15-stop route to 115 km (nearest-neighbour 121 km).
- LP carrier allocation lowers combined cost by about 2.9% versus a cheapest-first rule.

## Limitations

Simulated data; random-split scores are optimistic (temporal test is a little worse);
the Rs 120/hour value of delay is an assumption; routes use straight-line distance;
the LP is a simplified single-week model.
