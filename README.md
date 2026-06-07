# Pricing Optimisation Dashboard v2 — McKinsey Aligned

Rebuilt to reflect the actual McKinsey methodology from the handover.

## What changed from v1

| Feature | v1 | v2 (McKinsey aligned) |
|---|---|---|
| Price variable | Flat rate | **IRR** (McKinsey uses Log IRR as β input) |
| Demand model | Simple power law | **Log-log with β, γ_festival, γ_eid, γ_national** |
| Prediction correction | None | **Power-law correction (α_adj, i) per cluster** |
| Calibration | None | **Monthly factor: actual/predicted 14-week window per sector × salary** |
| Objective | Revenue or volume | **Max sales given IRR floor OR max IRR given sales floor** |
| Output | Flat rate directly | **IRR → flat rate derived per cell by tenure and fee** |
| Tabs | 4 | **6 (adds Model Components and Calibration tabs)** |

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Files

```
pricing_dashboard_v2/
├── app.py              ← Streamlit UI (6 tabs)
├── engine.py           ← Full McKinsey pipeline
├── requirements.txt
└── README.md
```

## McKinsey pipeline in engine.py

```
1. Log-log demand:  ln(Sales) = α + β·ln(IRR) + γ_f·Festival + γ_e·Eid + γ_n·National
2. Power-law corr:  Adj ln(S) = α_adj × raw_ln_pred + i
3. Calibration:     Final Sales = e^(Adj ln(S)) × cal_factor
4. IRR optimisation: SLSQP with IRR floor or sales floor constraint
5. Flat rate output: derived from optimal IRR per cell (tenure + fee)
```

## Python version: 3.11
