# Pricing Optimisation Dashboard

Interactive Streamlit app implementing the Sales & Net Revenue Based Pricing Strategy methodology.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## What it does

- **Price Grid tab** — current vs optimal price per cluster, revenue uplift table
- **Demand Curves tab** — revenue & sales curves per cluster with optimal price marked
- **Scenarios tab** — Best / Base / Downside / Worst case comparison + robustness check
- **Export tab** — download CSV and Excel with price grid + scenario sheet

## Sidebar controls

| Control | What it does |
|---|---|
| Objective toggle | Switch between maximising net revenue or sales volume |
| Max price movement | Total SAR movement allowed across all clusters |
| Price search range | How far from current price the optimizer searches |
| Unit cost adjustment | Simulate funding cost changes |
| Min volume per cluster | Constrain minimum weekly sales per cluster |

## Files

```
pricing_dashboard/
├── app.py          ← Streamlit application
├── engine.py       ← Pricing logic (demand model, optimizer, scenarios)
├── requirements.txt
└── README.md
```

## Methodology

- Demand model: Power-law `Sales = Baseline × (P_new / P_current) ^ β`
- Optimizer: scipy SLSQP with bounds + nonlinear constraints
- Clusters: 8 customer segments from Phase 3 K-Means
- Elasticity: log-log OLS estimates from Phase 2
