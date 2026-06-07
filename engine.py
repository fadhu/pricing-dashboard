"""
engine.py — McKinsey-aligned pricing engine
=============================================
Implements the full pipeline observed in the McKinsey handover:

  1. Clusters with pooled log-log elasticity (β)
  2. Calendar features: festival, Eid, National Day (separate γ coefficients)
  3. IRR as the price variable (not flat rate directly)
  4. Power-law correction layer (α_adj, i) per cluster
  5. Monthly calibration factor (actual/predicted over 14-week window)
     per economic sector × salary bucket
  6. Constrained optimisation: maximise sales given target IRR
     OR maximise IRR given minimum sales volume
  7. Cell-level flat rate derivation from cluster-level IRR target
"""

import numpy as np
import pandas as pd
import numpy_financial as nf
from scipy.optimize import minimize, Bounds, brentq
import warnings

warnings.filterwarnings("ignore")

COF = 0.04   # cost of funds


# ── Cluster definitions (from McKinsey slide) ─────────────────────────────────
# beta:           pooled log-log elasticity (Log FR column on slide)
# gamma_festival: coefficient for Festival/Hajj/Eid al-Adha
# gamma_eid:      coefficient for Eid holiday days (negative — offices shut)
# gamma_national: coefficient for National Day
# alpha_adj:      power-law correction coefficient (right-hand table)
# corr_i:         power-law correction constant
# baseline_sales: average weekly sales in stable non-holiday weeks
# baseline_irr:   IRR at which baseline_sales was observed
# tenure:         representative tenure for this cluster (months)
# fee_pct:        admin fee for this cluster

CLUSTERS = pd.DataFrame([
    # id  label                      beta   g_fest  g_eid   g_nat  a_adj   i      base_s  base_irr  ten  fee
    (0, "Saudi Gov 40k+",           -0.94,  1.22,  -1.35,   0.28,  1.06, -0.44,   45,    0.1252,   60, 0.010),
    (1, "Saudi Gov 10-20k",         -1.44,  2.16,  -2.26,   0.54,  1.01, -0.05,   38,    0.1326,   60, 0.010),
    (2, "Saudi SemiGov",            -1.37,  1.93,  -2.08,   0.44,  1.02, -0.18,   29,    0.1422,   48, 0.012),
    (3, "Expat Gov",                -1.17,  1.50,  -1.85,   0.32,  0.93,  1.13,   22,    0.1479,   48, 0.012),
    (4, "Saudi PrivateA",           -0.46,  1.54,  -1.74,   0.24,  1.06, -0.55,   18,    0.1580,   36, 0.015),
    (5, "Expat SemiGov",            -1.26,  0.00,  -0.97,   0.00,  1.02,  0.13,   14,    0.1680,   36, 0.015),
    (6, "Expat PrivateB 5-10k",     -1.13,  0.00,  -1.15,   0.44,  0.95,  0.99,   10,    0.1801,   36, 0.020),
    (7, "Expat PrivateB <5k",       -2.88,  0.00,   0.00,   0.00,  1.16, -1.57,    6,    0.1950,   24, 0.020),
], columns=["id","label","beta","gamma_festival","gamma_eid","gamma_national",
            "alpha_adj","corr_i","baseline_sales","baseline_irr",
            "tenure","fee_pct"])

# ── Calibration factors: economic sector × salary bucket ──────────────────────
# In production: computed fresh each month from actual vs predicted 14-week window
# Here: representative values to demonstrate the mechanism

SECTORS  = ["Government","Semi_Gov","Private_A","Private_B"]
SALARIES = ["<5k","5-10k","10-20k","20-40k","40k+"]

np.random.seed(42)
_cal_index = pd.MultiIndex.from_product([SECTORS, SALARIES],
                                         names=["sector","salary"])
CALIBRATION_FACTORS = pd.DataFrame({
    "sector":     [s for s, _ in _cal_index],
    "salary":     [sl for _, sl in _cal_index],
    "cal_factor": np.random.uniform(0.88, 1.18, len(_cal_index)),
    "n_weeks":    np.random.randint(8, 15, len(_cal_index)),
    "last_updated": "2025-10-01",
})
CALIBRATION_FACTORS["reliable"] = CALIBRATION_FACTORS["n_weeks"] >= 8


# ═════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_irr(flat_rate, tenure_months, fee_pct):
    """
    Annual IRR (= APR) from flat rate.
    Loan amount cancels — only tenure and fee matter.
    """
    if flat_rate <= 0:
        return 0.0
    total_interest   = flat_rate * (tenure_months / 12)
    monthly_payment  = (1 + total_interest) / tenure_months
    net_disbursement = (1 - fee_pct)
    cf = [-net_disbursement] + [monthly_payment] * int(tenure_months)
    try:
        m = nf.irr(cf)
        return float((1 + m) ** 12 - 1) if not np.isnan(m) else 0.0
    except:
        return 0.0


def flat_rate_for_irr(target_irr, tenure_months, fee_pct):
    """Find flat rate that delivers exactly target_irr (Brent's method)."""
    def gap(fr):
        return compute_irr(fr, tenure_months, fee_pct) - target_irr
    try:
        return brentq(gap, 0.001, 0.50)
    except:
        return None


def forecast_demand(cluster_row, irr_new, is_festival=0, is_eid=0,
                    is_national=0, apply_correction=True,
                    cal_factor=1.0):
    """
    Full McKinsey demand forecast pipeline for one cluster.

    Step 1: Log-log model
      ln(Sales) = α + β·ln(IRR) + γ_f·Festival + γ_e·Eid + γ_n·National

    Step 2: Power-law correction
      Adj ln(Sales) = α_adj × ln(Sales) + i

    Step 3: Calibration factor
      Final Sales = e^(Adj ln(Sales)) × cal_factor
    """
    beta          = float(cluster_row["beta"])
    base_irr      = float(cluster_row["baseline_irr"])
    base_sales    = float(cluster_row["baseline_sales"])
    gamma_festival= float(cluster_row["gamma_festival"])
    gamma_eid     = float(cluster_row["gamma_eid"])
    gamma_national= float(cluster_row["gamma_national"])
    alpha_adj     = float(cluster_row["alpha_adj"])
    corr_i        = float(cluster_row["corr_i"])

    if irr_new <= 0 or base_irr <= 0:
        return 0.0

    # ── Step 1: Raw log-log prediction ────────────────────────────────────────
    # Using ratio formula: baseline anchors at observed level
    # ln(Sales) = ln(Baseline) + β·ln(IRR_new/IRR_base) + γ·calendars
    ln_base     = np.log(base_sales)
    ln_irr_ratio= np.log(irr_new / base_irr)
    calendar    = (gamma_festival * is_festival
                   + gamma_eid     * is_eid
                   + gamma_national* is_national)
    raw_ln_pred = ln_base + beta * ln_irr_ratio + calendar

    # ── Step 2: Power-law correction ──────────────────────────────────────────
    if apply_correction:
        adj_ln_pred = alpha_adj * raw_ln_pred + corr_i
    else:
        adj_ln_pred = raw_ln_pred

    # ── Step 3: Calibration factor ────────────────────────────────────────────
    adj_sales = np.exp(adj_ln_pred)
    return max(adj_sales * cal_factor, 0.0)


def get_calibration_factor(sector, salary, cal_df=None):
    """Look up calibration factor for a sector × salary cell."""
    if cal_df is None:
        cal_df = CALIBRATION_FACTORS
    match = cal_df[(cal_df["sector"] == sector) & (cal_df["salary"] == salary)]
    if len(match) == 0:
        return 1.0
    r = match.iloc[0]
    if r["reliable"]:
        return float(r["cal_factor"])
    # Fallback: sector average
    sector_avg = cal_df[cal_df["sector"] == sector]["cal_factor"].mean()
    return float(sector_avg) if not np.isnan(sector_avg) else 1.0


def get_cluster_cal_factor(cluster_id, cal_df=None):
    """
    Aggregate cell-level calibration factors up to cluster level.
    Uses sales-weighted average across cells in the cluster.
    In production: map cluster → constituent (sector, salary) cells
    and weight by their baseline_sales contribution.
    Here: simple average across all cells as a proxy.
    """
    if cal_df is None:
        cal_df = CALIBRATION_FACTORS
    return float(cal_df["cal_factor"].mean())


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════

def run_optimisation(clusters_df, objective="max_sales_given_irr",
                     irr_target=0.12, min_sales_retention=0.90,
                     max_irr_increase=0.30, is_festival=0, is_eid=0,
                     is_national=0, apply_correction=True,
                     cal_df=None):
    """
    McKinsey-aligned constrained optimisation.

    Objective 1 — max_sales_given_irr:
      Maximise total sales volume
      Subject to: IRR_i >= irr_target for each cluster

    Objective 2 — max_irr_given_sales:
      Maximise portfolio weighted IRR
      Subject to: Sales_i >= retention% × baseline for each cluster

    Decision variable: IRR per cluster (not flat rate directly)
    Flat rate is derived from the optimal IRR per cluster.
    """
    N           = len(clusters_df)
    cal_factors = [get_cluster_cal_factor(i, cal_df)
                   for i in clusters_df["id"]]

    # Bounds on IRR (not flat rate)
    # Floor: irr_target (no point going below if it's the constraint)
    # Cap:   current IRR × (1 + max_irr_increase)
    irr_floors = np.full(N, irr_target * 0.85)   # allow slight dip for volume
    irr_caps   = clusters_df["baseline_irr"].values * (1 + max_irr_increase)
    bounds     = Bounds(lb=irr_floors, ub=irr_caps)

    x0 = clusters_df["baseline_irr"].values.astype(float)

    def get_sales(irr_vec, i):
        row = clusters_df.iloc[i]
        return forecast_demand(row, irr_vec[i], is_festival, is_eid,
                                is_national, apply_correction, cal_factors[i])

    def get_portfolio_irr(irr_vec):
        total_w, total_wv = 0.0, 0.0
        for i, row in clusters_df.iterrows():
            iloc_i = clusters_df.index.get_loc(i)
            sales  = get_sales(irr_vec, iloc_i)
            w      = sales * row["tenure"]   # weight by loan-months
            total_wv += irr_vec[iloc_i] * w
            total_w  += w
        return total_wv / total_w if total_w > 0 else 0.0

    # ── Objective 1: Maximise sales given IRR floor ────────────────────────────
    if objective == "max_sales_given_irr":
        def obj(irr_vec):
            return -sum(get_sales(irr_vec, i) for i in range(N))

        constraints = []
        # IRR floor per cluster
        for i, row in clusters_df.iterrows():
            iloc_i = clusters_df.index.get_loc(i)
            constraints.append({
                "type": "ineq",
                "fun":  (lambda v, ii=iloc_i: v[ii] - irr_target)
            })

    # ── Objective 2: Maximise IRR given sales floor ────────────────────────────
    else:
        def obj(irr_vec):
            return -get_portfolio_irr(irr_vec)

        constraints = []
        for i, row in clusters_df.iterrows():
            iloc_i   = clusters_df.index.get_loc(i)
            min_sales = float(row["baseline_sales"]) * min_sales_retention
            constraints.append({
                "type": "ineq",
                "fun":  (lambda v, ii=iloc_i, r=row, ms=min_sales:
                         get_sales(v, ii) - ms)
            })

    result = minimize(obj, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"ftol": 1e-6, "maxiter": 500})

    # ── Build output ───────────────────────────────────────────────────────────
    out = clusters_df.copy()
    out["optimal_irr"]        = result.x
    out["optimal_flat_rate"]  = [
        flat_rate_for_irr(result.x[i], row["tenure"], row["fee_pct"]) or 0
        for i, (_, row) in enumerate(clusters_df.iterrows())
    ]
    out["optimal_sales"] = [
        get_sales(result.x, i) for i in range(N)
    ]
    out["baseline_flat_rate"] = [
        flat_rate_for_irr(row["baseline_irr"], row["tenure"], row["fee_pct"]) or 0
        for _, row in clusters_df.iterrows()
    ]
    out["irr_change_bps"]   = (out["optimal_irr"] - out["baseline_irr"]) * 10000
    out["sales_change_pct"] = ((out["optimal_sales"] / out["baseline_sales"]) - 1) * 100
    out["cal_factor"]       = cal_factors
    out["solver_status"]    = result.message
    out["objective"]        = objective
    return out


def get_scenario_results(irr_target, min_sales_retention, is_festival,
                         is_eid, is_national, apply_correction, cal_df=None):
    """Four-scenario comparison: best/base/downside/worst."""
    scenarios = {
        "🟢 Best case":  {"beta_adj": +0.20, "sales_adj": +0.20},
        "⚪ Base case":  {"beta_adj":  0.00, "sales_adj":  0.00},
        "🟠 Downside":   {"beta_adj": -0.15, "sales_adj": -0.10},
        "🔴 Worst case": {"beta_adj": -0.30, "sales_adj": -0.20},
    }
    rows = []
    for name, adj in scenarios.items():
        df_s = CLUSTERS.copy()
        df_s["beta"]           = df_s["beta"]           + adj["beta_adj"]
        df_s["baseline_sales"] = df_s["baseline_sales"] * (1 + adj["sales_adj"])
        out = run_optimisation(
            df_s, objective="max_sales_given_irr",
            irr_target=irr_target,
            min_sales_retention=min_sales_retention,
            is_festival=is_festival, is_eid=is_eid, is_national=is_national,
            apply_correction=apply_correction, cal_df=cal_df,
        )
        rows.append({
            "Scenario":      name,
            "Total Sales":   out["optimal_sales"].sum(),
            "Avg IRR":       out["optimal_irr"].mean(),
            "Avg Flat Rate": out["optimal_flat_rate"].mean(),
        })
    df_s = pd.DataFrame(rows)
    base = df_s[df_s["Scenario"]=="⚪ Base case"]["Total Sales"].values[0]
    df_s["vs Base (%)"] = ((df_s["Total Sales"] - base) / base * 100).round(1)
    return df_s
