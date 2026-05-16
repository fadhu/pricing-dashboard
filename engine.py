import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds


def get_default_clusters():
    return pd.DataFrame([
        (0, "Saudi Gov 40k+",        -0.62, 45, 620, 280),
        (1, "Saudi Gov 10-20k",      -0.98, 38, 540, 240),
        (2, "Saudi SemiGov",         -1.18, 29, 490, 210),
        (3, "Expat Gov",             -1.31, 22, 460, 200),
        (4, "Saudi PrivateA",        -1.55, 18, 410, 180),
        (5, "Expat SemiGov",         -1.72, 14, 380, 165),
        (6, "Expat PrivateB 5-10k",  -1.94, 10, 340, 150),
        (7, "Expat PrivateB <5k",    -2.21,  6, 300, 130),
    ], columns=["id", "label", "beta", "baseline_sales", "current_price", "unit_cost"])


def projected_sales(price, row):
    return float(row["baseline_sales"]) * (float(price) / float(row["current_price"])) ** float(row["beta"])


def net_revenue(price, row):
    return projected_sales(price, row) * (float(price) - float(row["unit_cost"]))


def simulate_curve(row, price_range_pct=0.30, steps=200):
    cur   = float(row["current_price"])
    beta  = float(row["beta"])
    base  = float(row["baseline_sales"])
    uc    = float(row["unit_cost"])

    prices  = np.linspace(cur * (1 - price_range_pct), cur * (1 + price_range_pct), steps)
    sales   = base * (prices / cur) ** beta
    revenue = sales * (prices - uc)
    idx     = np.argmax(revenue)

    return {
        "prices":          prices,
        "sales":           sales,
        "revenue":         revenue,
        "rev_max_price":   prices[idx],
        "rev_max_revenue": revenue[idx],
    }


def run_optimisation(clusters, objective="revenue", min_volumes=None,
                     max_movement=600, cost_adjustment=0.0, price_range_pct=0.30):
    df = clusters.copy()
    df["unit_cost"] = df["unit_cost"] * (1 + cost_adjustment)

    N   = len(df)
    x0  = df["current_price"].values.astype(float)
    flr = (df["unit_cost"] + 50).values.astype(float)
    cap = (df["current_price"] * (1 + price_range_pct)).values.astype(float)
    bnd = Bounds(lb=flr, ub=cap)
    con = []

    if min_volumes:
        for cid, mv in min_volumes.items():
            matches = df[df["id"] == cid]
            if matches.empty:
                continue
            iloc_idx = df.index.get_loc(matches.index[0])
            row_snap = df.iloc[iloc_idx].copy()
            con.append({
                "type": "ineq",
                "fun": (lambda p, ii=iloc_idx, rr=row_snap, m=mv:
                        projected_sales(p[ii], rr) - m)
            })

    cur_prices = df["current_price"].values.astype(float)
    con.append({
        "type": "ineq",
        "fun": lambda p, c=cur_prices, m=max_movement:
               m - np.sum(np.abs(p - c))
    })

    if objective == "revenue":
        def obj_fn(p):
            return -sum(net_revenue(p[i], df.iloc[i]) for i in range(N))
    else:
        def obj_fn(p):
            return -sum(projected_sales(p[i], df.iloc[i]) for i in range(N))

    res = minimize(obj_fn, x0, method="SLSQP", bounds=bnd, constraints=con,
                   options={"ftol": 1e-10, "maxiter": 1000})

    df["optimal_price"]   = np.round(res.x, 0)
    df["price_change"]    = df["optimal_price"] - df["current_price"]
    df["price_chg_pct"]   = (df["price_change"] / df["current_price"] * 100).round(1)
    df["optimal_sales"]   = [projected_sales(res.x[i], df.iloc[i]) for i in range(N)]
    df["optimal_revenue"] = [net_revenue(res.x[i], df.iloc[i]) for i in range(N)]
    df["current_revenue"] = [net_revenue(df.iloc[i]["current_price"], df.iloc[i]) for i in range(N)]
    df["current_sales"]   = [projected_sales(df.iloc[i]["current_price"], df.iloc[i]) for i in range(N)]
    df["revenue_uplift"]  = df["optimal_revenue"] - df["current_revenue"]
    df["status"]          = res.message
    return df


def get_scenario_results(objective, max_movement, price_range_pct):
    scenarios = {
        "🟢 Best case":  {"beta_adj": +0.20, "cost_adj": -0.10, "sales_adj": +0.20},
        "⚪ Base case":  {"beta_adj":  0.00, "cost_adj":  0.00, "sales_adj":  0.00},
        "🟠 Downside":   {"beta_adj": -0.15, "cost_adj": +0.08, "sales_adj": -0.10},
        "🔴 Worst case": {"beta_adj": -0.30, "cost_adj": +0.15, "sales_adj": -0.20},
    }
    rows = []
    for name, adj in scenarios.items():
        df_s = get_default_clusters().copy()
        df_s["beta"]           = df_s["beta"] + adj["beta_adj"]
        df_s["unit_cost"]      = df_s["unit_cost"] * (1 + adj["cost_adj"])
        df_s["baseline_sales"] = df_s["baseline_sales"] * (1 + adj["sales_adj"])
        out = run_optimisation(df_s, objective=objective,
                               max_movement=max_movement,
                               price_range_pct=price_range_pct)
        rows.append({
            "Scenario":      name,
            "Total Revenue": out["optimal_revenue"].sum(),
            "Total Sales":   out["optimal_sales"].sum(),
            "Avg Opt Price": out["optimal_price"].mean(),
        })
    df_scen = pd.DataFrame(rows)
    base_rev = df_scen[df_scen["Scenario"] == "⚪ Base case"]["Total Revenue"].values[0]
    df_scen["vs Base (%)"] = ((df_scen["Total Revenue"] - base_rev) / base_rev * 100).round(1)
    return df_scen
