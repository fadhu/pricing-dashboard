"""
app.py — McKinsey-aligned Pricing Optimisation Dashboard
==========================================================
Reflects the actual model structure from the McKinsey handover:
  - IRR as the optimisation variable (not flat rate)
  - Flat rate derived per cell from IRR + tenure + fee
  - Log-log demand model with β, γ_festival, γ_eid, γ_national
  - Power-law correction (α_adj, i) per cluster
  - Monthly calibration factor (actual/predicted, 14-week window)
  - Two objectives: max sales | IRR floor  OR  max IRR | sales floor
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
from engine import (
    CLUSTERS, CALIBRATION_FACTORS, SECTORS, SALARIES,
    compute_irr, flat_rate_for_irr, forecast_demand,
    get_cluster_cal_factor, run_optimisation, get_scenario_results,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pricing Optimisation",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
}
.main-header {
    background: linear-gradient(135deg, #0a2540 0%, #1a4a6b 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    border-left: 4px solid #00b4d8;
}
.main-header h1 { color:#fff; font-size:1.6rem; font-weight:700; margin:0 0 0.2rem; }
.main-header p  { color:#90caf9; font-size:0.82rem; margin:0; font-family:monospace; }
.kpi-card { background:#0a2540; border:1px solid #1a4a6b; border-radius:10px;
            padding:1rem 1.2rem; text-align:center; margin-bottom:0.4rem; }
.kpi-label { color:#90caf9; font-size:0.72rem; font-family:monospace;
             text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem; }
.kpi-value { color:#fff; font-size:1.5rem; font-weight:700; }
.kpi-delta-pos { color:#4caf50; font-size:0.85rem; font-weight:600; }
.kpi-delta-neg { color:#ef5350; font-size:0.85rem; font-weight:600; }
.kpi-delta-neu { color:#90caf9; font-size:0.85rem; font-weight:600; }
section[data-testid="stSidebar"] { background: #0a2540 !important; }
section[data-testid="stSidebar"] * { color: #e0f0ff !important; }
.stTabs [data-baseweb="tab-list"] { background:#0f2f4a; border-radius:8px; padding:3px; }
.stTabs [data-baseweb="tab"] { color:#90caf9; font-family:monospace; font-size:0.78rem; }
.stTabs [aria-selected="true"] { background:#00b4d8 !important; color:#0a2540 !important;
                                   border-radius:5px; font-weight:700; }
.stDownloadButton button { background:#00b4d8 !important; color:#0a2540 !important;
                            font-weight:700 !important; border:none !important; width:100%; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>💰 Loan Pricing Optimisation Dashboard</h1>
  <p>McKinsey methodology · Log-log demand · IRR optimisation · Power-law correction · Monthly calibration</p>
</div>
""", unsafe_allow_html=True)

# ── Matplotlib theme ───────────────────────────────────────────────────────────
@st.cache_resource
def _init_mpl():
    plt.rcParams.update({
        "figure.facecolor": "#0a2540", "axes.facecolor": "#0f2f4a",
        "axes.edgecolor": "#1a4a6b",   "axes.labelcolor": "#90caf9",
        "axes.titlecolor": "#ffffff",  "xtick.color": "#90caf9",
        "ytick.color": "#90caf9",      "grid.color": "#1a4a6b",
        "text.color": "#ffffff",       "legend.facecolor": "#0a2540",
        "legend.edgecolor": "#1a4a6b", "legend.labelcolor": "#e0f0ff",
        "axes.titlesize": 10,          "axes.labelsize": 9,
        "font.family": "monospace",
        "path.simplify": True,         "path.simplify_threshold": 0.5,
    })
_init_mpl()

BLUE = "#00b4d8"; RED = "#ef5350"; GREEN = "#4caf50"
AMBER = "#ffb74d"; GREY = "#546e7a"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Optimisation Settings")

    objective = st.radio(
        "Business objective",
        ["max_sales_given_irr", "max_irr_given_sales"],
        format_func=lambda x: (
            "📈 Max Sales | IRR floor"
            if x == "max_sales_given_irr"
            else "📊 Max IRR | Sales floor"
        ),
    )

    irr_target = st.slider(
        "IRR floor / target (%)",
        min_value=8.0, max_value=20.0, value=12.0, step=0.5,
        help="Minimum IRR required per cluster (Objective 1) or target IRR (Objective 2)"
    ) / 100

    min_sales_retention = st.slider(
        "Min sales retention (%)",
        min_value=60, max_value=100, value=90, step=5,
        help="Minimum % of current volume to maintain (Objective 2)"
    ) / 100

    max_irr_increase = st.slider(
        "Max IRR increase (%)",
        min_value=5, max_value=50, value=30, step=5,
        help="How far above current IRR the optimizer can go"
    ) / 100

    st.markdown("---")
    st.markdown("**📅 Calendar week type**")
    st.caption("Affects demand forecast via γ coefficients")
    is_festival = st.checkbox("Festival week (Ramadan / Hajj / National Day)", value=False)
    is_eid      = st.checkbox("Eid holiday week", value=False)
    is_national = st.checkbox("National Day week", value=False)

    st.markdown("---")
    st.markdown("**🔧 Model settings**")
    apply_correction = st.checkbox(
        "Apply power-law correction (α_adj, i)",
        value=True,
        help="Applies the per-cluster structural bias correction from the McKinsey slide"
    )
    apply_calibration = st.checkbox(
        "Apply monthly calibration factor",
        value=True,
        help="Applies actual/predicted ratio from most recent 14 weeks per sector × salary"
    )

    st.markdown("---")
    st.caption("Built on McKinsey methodology · SLSQP solver · scipy")


# ══════════════════════════════════════════════════════════════════════════════
# RUN OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cached_optimisation(objective, irr_target, min_sales_retention,
                         max_irr_increase, is_festival, is_eid, is_national,
                         apply_correction, apply_calibration):
    cal_df = CALIBRATION_FACTORS if apply_calibration else None
    return run_optimisation(
        CLUSTERS, objective=objective,
        irr_target=irr_target,
        min_sales_retention=min_sales_retention,
        max_irr_increase=max_irr_increase,
        is_festival=int(is_festival), is_eid=int(is_eid),
        is_national=int(is_national),
        apply_correction=apply_correction,
        cal_df=cal_df,
    )

results = cached_optimisation(
    objective, irr_target, min_sales_retention, max_irr_increase,
    is_festival, is_eid, is_national, apply_correction, apply_calibration,
)

# ── KPI Row ────────────────────────────────────────────────────────────────────
cur_sales    = results["baseline_sales"].sum()
opt_sales    = results["optimal_sales"].sum()
sales_delta  = (opt_sales / cur_sales - 1) * 100

cur_irr_wt   = (results["baseline_irr"]  * results["baseline_sales"]).sum() / cur_sales
opt_irr_wt   = (results["optimal_irr"]   * results["optimal_sales"]).sum()  / opt_sales
irr_delta_bps= (opt_irr_wt - cur_irr_wt) * 10000

cur_fr       = results["baseline_flat_rate"].mean()
opt_fr       = results["optimal_flat_rate"].mean()

def delta_html(val, suffix="%", prefix=""):
    cls  = "kpi-delta-pos" if val >= 0 else "kpi-delta-neg"
    sign = "▲" if val >= 0 else "▼"
    return f'<div class="{cls}">{sign} {prefix}{abs(val):.1f}{suffix}</div>'

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, delta, suffix in [
    (c1, "Current weekly sales",   f"{cur_sales:.0f} units",  None,         None),
    (c2, "Optimal weekly sales",   f"{opt_sales:.1f} units",  sales_delta,  "%"),
    (c3, "Current wtd IRR",        f"{cur_irr_wt:.2%}",       None,         None),
    (c4, "Optimal wtd IRR",        f"{opt_irr_wt:.2%}",       irr_delta_bps,"bps"),
    (c5, "Avg flat rate change",   f"{opt_fr:.2%}",
     (opt_fr - cur_fr)*10000, "bps"),
]:
    d_html = delta_html(delta, suffix) if delta is not None else \
             '<div class="kpi-delta-neu">baseline</div>'
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{d_html}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
if results["solver_status"].iloc[0] != "Optimization terminated successfully":
    st.warning(f"⚠️ Solver: {results['solver_status'].iloc[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Price Grid",
    "📈 Demand Curves",
    "🔧 Model Components",
    "📅 Calibration",
    "🔀 Scenarios",
    "⬇️ Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE GRID
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Optimised price grid")
    week_type = ("🕌 Festival week" if is_festival else
                 "🌙 Eid week"      if is_eid      else
                 "🇸🇦 National Day"  if is_national else
                 "📅 Normal week")
    st.caption(f"Objective: **{objective.replace('_',' ')}** · "
               f"IRR target: **{irr_target:.1%}** · "
               f"Week type: **{week_type}** · "
               f"Correction: **{'on' if apply_correction else 'off'}** · "
               f"Calibration: **{'on' if apply_calibration else 'off'}**")

    display = results[[
        "label","beta","baseline_irr","optimal_irr","irr_change_bps",
        "baseline_flat_rate","optimal_flat_rate",
        "baseline_sales","optimal_sales","sales_change_pct","cal_factor",
    ]].copy()
    display.columns = [
        "Cluster","β","Cur IRR","Opt IRR","IRR Δ (bps)",
        "Cur Flat Rate","Opt Flat Rate",
        "Cur Sales","Opt Sales","Sales Δ %","Cal Factor",
    ]

    def style_delta(val):
        if not isinstance(val, (int, float)): return ""
        return ("color:#4caf50;font-weight:600" if val > 0
                else "color:#ef5350;font-weight:600" if val < 0 else "")

    st.dataframe(
        display.style
        .format({
            "β":            "{:.2f}",
            "Cur IRR":      "{:.2%}",  "Opt IRR":       "{:.2%}",
            "IRR Δ (bps)":  "{:+.0f}", "Cur Flat Rate": "{:.2%}",
            "Opt Flat Rate":"{:.2%}",  "Cur Sales":     "{:.1f}",
            "Opt Sales":    "{:.1f}",  "Sales Δ %":     "{:+.1f}%",
            "Cal Factor":   "{:.3f}",
        })
        .applymap(style_delta, subset=["IRR Δ (bps)","Sales Δ %"])
        .set_properties(**{"font-family":"monospace","font-size":"12px"}),
        use_container_width=True, height=320,
    )

    # Bar chart: IRR and flat rate comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0a2540")
    x = np.arange(len(results)); w = 0.35
    labels = [r[:16] for r in results["label"]]

    axes[0].bar(x - w/2, results["baseline_irr"]*100, w,
                label="Current IRR",  color=GREY,  alpha=0.85)
    axes[0].bar(x + w/2, results["optimal_irr"]*100, w,
                label="Optimal IRR",  color=BLUE,  alpha=0.85)
    axes[0].axhline(irr_target*100, color=RED, linestyle="--",
                    linewidth=1.2, label=f"IRR floor {irr_target:.1%}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("IRR (%)"); axes[0].set_title("Current vs Optimal IRR")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - w/2, results["baseline_flat_rate"]*100, w,
                label="Current flat rate", color=GREY, alpha=0.85)
    axes[1].bar(x + w/2, results["optimal_flat_rate"]*100, w,
                label="Optimal flat rate", color=AMBER, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("Flat Rate (%)"); axes[1].set_title("Flat Rate (derived from IRR)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEMAND CURVES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Demand curves — log-log model + correction + calibration")
    st.caption("Shows how sales volume responds to IRR changes per cluster. "
               "Blue = normal | Dashed = festival | Dotted = Eid")

    selected = st.multiselect(
        "Select clusters",
        options=list(results["label"]),
        default=list(results["label"][:4]),
    )

    if selected:
        n_sel = len(selected)
        fig   = plt.figure(figsize=(15, 4.5 * n_sel))
        fig.patch.set_facecolor("#0a2540")
        gs_   = gridspec.GridSpec(n_sel, 2, figure=fig, hspace=0.55, wspace=0.35)

        for pi, seg_label in enumerate(selected):
            row   = results[results["label"] == seg_label].iloc[0]
            cal_f = float(row["cal_factor"])
            base_irr = float(row["baseline_irr"])

            irr_grid = np.linspace(base_irr * 0.65, base_irr * 1.35, 150)
            sales_normal   = [forecast_demand(row, irr, 0, 0, 0,
                               apply_correction, cal_f) for irr in irr_grid]
            sales_festival = [forecast_demand(row, irr, 1, 0, 0,
                               apply_correction, cal_f) for irr in irr_grid]
            sales_eid      = [forecast_demand(row, irr, 0, 1, 0,
                               apply_correction, cal_f) for irr in irr_grid]

            # Revenue proxy: sales × (IRR - CoF)
            rev_normal = [s * max(irr - COF, 0)
                          for s, irr in zip(sales_normal, irr_grid)]
            from engine import COF as _COF

            beta  = float(row["beta"])
            color = (GREEN if beta > -1 else AMBER if beta > -1.5 else RED)

            ax_s = fig.add_subplot(gs_[pi, 0])
            ax_r = fig.add_subplot(gs_[pi, 1])

            ax_s.fill_between(irr_grid*100, sales_festival,
                               alpha=0.12, color=color)
            ax_s.plot(irr_grid*100, sales_normal,
                      color=color, linewidth=2.5, label="Normal")
            ax_s.plot(irr_grid*100, sales_festival,
                      color=color, linewidth=1.2, linestyle="--",
                      alpha=0.7, label="Festival")
            ax_s.plot(irr_grid*100, sales_eid,
                      color=color, linewidth=1.0, linestyle=":",
                      alpha=0.6, label="Eid")
            ax_s.axvline(base_irr*100, color=GREY, linestyle=":",
                          linewidth=1.2, label="Current IRR")
            ax_s.axvline(float(row["optimal_irr"])*100, color=color,
                          linestyle="-.", linewidth=1.8, label="Optimal IRR")
            ax_s.set_xlabel("IRR (%)"); ax_s.set_ylabel("Weekly Sales")
            ax_s.set_title(f"{seg_label[:28]}\nβ={beta:.2f}  "
                           f"γ_fest={row['gamma_festival']:.2f}  "
                           f"α_adj={row['alpha_adj']:.2f}  cal={cal_f:.3f}",
                           fontsize=8)
            ax_s.legend(fontsize=7); ax_s.grid(alpha=0.25)

            ax_r.fill_between(irr_grid*100, rev_normal,
                               alpha=0.12, color=color)
            ax_r.plot(irr_grid*100, rev_normal, color=color, linewidth=2.5)
            ax_r.axvline(base_irr*100, color=GREY, linestyle=":", linewidth=1.2)
            ax_r.axvline(float(row["optimal_irr"])*100, color=BLUE,
                          linestyle="-.", linewidth=1.8, label="Optimal")
            ax_r.set_xlabel("IRR (%)"); ax_r.set_ylabel("Revenue proxy")
            ax_r.set_title("Revenue curve\n(sales × spread)")
            ax_r.legend(fontsize=7); ax_r.grid(alpha=0.25)

        plt.suptitle("Demand & Revenue Curves — full McKinsey pipeline",
                     fontsize=12, fontweight="bold", color="white", y=1.01)
        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Model components per cluster")
    st.caption("Shows every parameter from the McKinsey slide: "
               "β, γ coefficients, power-law correction (α_adj, i), baseline")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Log-log regression coefficients**")
        coef_df = CLUSTERS[[
            "label","beta",
            "gamma_festival","gamma_eid","gamma_national"
        ]].copy()
        coef_df.columns = ["Cluster","β (IRR)","γ Festival","γ Eid","γ National"]
        st.dataframe(
            coef_df.style.format({
                "β (IRR)":    "{:.2f}",
                "γ Festival": "{:+.2f}",
                "γ Eid":      "{:+.2f}",
                "γ National": "{:+.2f}",
            }).applymap(
                lambda v: "color:#ef5350" if isinstance(v,float) and v < 0
                          else "color:#4caf50" if isinstance(v,float) and v > 0
                          else "",
                subset=["β (IRR)","γ Festival","γ Eid","γ National"]
            ).set_properties(**{"font-size":"12px","font-family":"monospace"}),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**γ interpretation** (multiplicative lift)")
        for _, r in CLUSTERS.iterrows():
            gf = np.exp(r["gamma_festival"])
            ge = np.exp(r["gamma_eid"])
            st.markdown(
                f"  `{r['label'][:20]}`  "
                f"Festival: **×{gf:.2f}**  |  "
                f"Eid: **×{ge:.2f}**",
                unsafe_allow_html=False,
            )

    with col_b:
        st.markdown("**Power-law correction parameters**")
        corr_df = CLUSTERS[["label","alpha_adj","corr_i"]].copy()
        corr_df["e^i (level multiplier)"] = np.exp(corr_df["corr_i"])
        corr_df["effect"] = corr_df["alpha_adj"].apply(
            lambda a: "Amplify range ↑" if a > 1.02 else
                      "Compress range ↓" if a < 0.98 else "No range change"
        )
        corr_df.columns = ["Cluster","α_adj","i","e^i","Effect"]
        st.dataframe(
            corr_df.style.format({
                "α_adj": "{:.2f}", "i": "{:+.2f}", "e^i": "{:.3f}"
            }).applymap(
                lambda v: "color:#ffb74d" if isinstance(v,float) and abs(v-1)>0.03
                          else "",
                subset=["α_adj"]
            ).set_properties(**{"font-size":"12px","font-family":"monospace"}),
            use_container_width=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**IRR → Flat rate conversion**")
        st.caption("Same IRR, different flat rate by tenure")
        rows_irr = []
        for _, r in CLUSTERS.iterrows():
            for tenure in [24, 36, 48, 60]:
                fr = flat_rate_for_irr(float(r["baseline_irr"]), tenure,
                                       float(r["fee_pct"]))
                rows_irr.append({
                    "Cluster":  r["label"][:18],
                    "Tenure":   f"{tenure}m",
                    "IRR":      f"{r['baseline_irr']:.2%}",
                    "Flat Rate":f"{fr:.2%}" if fr else "N/A",
                })
        irr_df = pd.DataFrame(rows_irr)
        st.dataframe(
            irr_df.pivot(index="Cluster", columns="Tenure", values="Flat Rate"),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — CALIBRATION FACTORS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Monthly calibration factors")
    st.caption("Actual / Predicted ratio over most recent 14 weeks · "
               "Per economic sector × salary bucket · "
               "Refreshed monthly")

    col_c, col_d = st.columns([3, 2])

    with col_c:
        # Heatmap pivot
        pivot = CALIBRATION_FACTORS.pivot(
            index="sector", columns="salary", values="cal_factor"
        ).reindex(columns=["<5k","5-10k","10-20k","20-40k","40k+"])

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0a2540")
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.80, vmax=1.20,
                       aspect="auto")
        plt.colorbar(im, ax=ax, label="Calibration factor")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("Calibration factor heatmap\n"
                     "Green > 1 = model under-predicting · "
                     "Red < 1 = over-predicting", fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(val - 1) > 0.10 else "#cccccc")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("**Factor table**")
        st.dataframe(
            CALIBRATION_FACTORS[["sector","salary","cal_factor",
                                  "n_weeks","reliable","last_updated"]]
            .style.format({"cal_factor": "{:.4f}"})
            .applymap(
                lambda v: ("color:#4caf50" if isinstance(v,float) and v > 1.05
                           else "color:#ef5350" if isinstance(v,float) and v < 0.95
                           else ""),
                subset=["cal_factor"]
            ).set_properties(**{"font-size":"11px","font-family":"monospace"}),
            use_container_width=True, height=400,
        )

    st.markdown("---")
    st.markdown("**How calibration factor affects the forecast**")
    st.caption("Shows raw vs calibrated forecast for a selected cluster and IRR")

    demo_cluster = st.selectbox("Demo cluster",
                                 options=list(CLUSTERS["label"]), index=0)
    demo_irr = st.slider("Demo IRR", 0.08, 0.25,
                          float(CLUSTERS[CLUSTERS.label==demo_cluster]["baseline_irr"].values[0]),
                          0.005, format="%.3f")

    row_demo = CLUSTERS[CLUSTERS.label == demo_cluster].iloc[0]
    cal_f_demo = float(results[results.label==demo_cluster]["cal_factor"].values[0])
    raw_sales  = forecast_demand(row_demo, demo_irr, int(is_festival),
                                  int(is_eid), int(is_national),
                                  apply_correction, 1.0)
    cal_sales  = forecast_demand(row_demo, demo_irr, int(is_festival),
                                  int(is_eid), int(is_national),
                                  apply_correction, cal_f_demo)

    c1d, c2d, c3d = st.columns(3)
    c1d.metric("Raw forecast",        f"{raw_sales:.1f} units/wk")
    c2d.metric("Calibration factor",  f"{cal_f_demo:.4f}",
               f"{(cal_f_demo-1)*100:+.1f}%")
    c3d.metric("Calibrated forecast", f"{cal_sales:.1f} units/wk",
               f"{cal_sales-raw_sales:+.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("#### Scenario comparison")
    st.caption("Best / Base / Downside / Worst — β and baseline sales shift simultaneously")

    @st.cache_data
    def cached_scenarios(irr_target, min_sales_retention, is_festival,
                          is_eid, is_national, apply_correction, apply_calibration):
        cal_df = CALIBRATION_FACTORS if apply_calibration else None
        return get_scenario_results(
            irr_target, min_sales_retention,
            int(is_festival), int(is_eid), int(is_national),
            apply_correction, cal_df,
        )

    scen_df = cached_scenarios(
        irr_target, min_sales_retention, is_festival, is_eid,
        is_national, apply_correction, apply_calibration,
    )
    base_sales = scen_df[scen_df["Scenario"]=="⚪ Base case"]["Total Sales"].values[0]

    st.dataframe(
        scen_df.style.format({
            "Total Sales": "{:.1f} units",
            "Avg IRR":     "{:.2%}",
            "Avg Flat Rate":"{:.2%}",
            "vs Base (%)": "{:+.1f}%",
        }).applymap(
            lambda v: ("color:#4caf50;font-weight:600" if isinstance(v,float) and v > 0
                       else "color:#ef5350;font-weight:600" if isinstance(v,float) and v < 0
                       else ""),
            subset=["vs Base (%)"]
        ).set_properties(**{"font-family":"monospace","font-size":"12px"}),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0a2540")
    colors_s = [GREEN, BLUE, AMBER, RED]
    bars = ax.bar(scen_df["Scenario"], scen_df["Total Sales"],
                  color=colors_s, alpha=0.85, edgecolor="#0a2540")
    ax.axhline(base_sales, color=GREY, linestyle="--",
               linewidth=1.5, label="Base")
    for bar, (_, row) in zip(bars, scen_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{row['vs Base (%)']:+.1f}%",
                ha="center", fontsize=10, fontweight="bold",
                color=GREEN if row["vs Base (%)"] >= 0 else RED)
    ax.set_ylabel("Total weekly sales (units)")
    ax.set_title("Sales volume by scenario")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Robustness check
    worst = scen_df[scen_df["Scenario"]=="🔴 Worst case"]["Total Sales"].values[0]
    cur   = results["baseline_sales"].sum()
    if worst > cur * 0.80:
        st.success(f"✅ Robust — worst case still delivers "
                   f"{worst:.0f} units/week ({worst/cur:.0%} of current)")
    else:
        st.warning(f"⚠️ Fragile — worst case drops to {worst:.0f} units/week "
                   f"({worst/cur:.0%} of current). Consider A/B test before full rollout.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("#### Export results")

    export = results[[
        "label","beta","gamma_festival","gamma_eid",
        "alpha_adj","corr_i",
        "baseline_irr","optimal_irr","irr_change_bps",
        "baseline_flat_rate","optimal_flat_rate",
        "baseline_sales","optimal_sales","sales_change_pct",
        "cal_factor","tenure","fee_pct",
    ]].copy()
    export.columns = [
        "Cluster","β","γ Festival","γ Eid",
        "α_adj (correction)","i (correction)",
        "Baseline IRR","Optimal IRR","IRR Δ (bps)",
        "Baseline Flat Rate","Optimal Flat Rate",
        "Baseline Sales/wk","Optimal Sales/wk","Sales Δ %",
        "Calibration Factor","Tenure (months)","Fee %",
    ]
    export["Objective"]   = objective
    export["IRR Target"]  = irr_target
    export["Week Type"]   = ("Festival" if is_festival else
                              "Eid"      if is_eid      else
                              "Normal")
    export["Correction Applied"]   = apply_correction
    export["Calibration Applied"]  = apply_calibration

    csv_bytes = export.to_csv(index=False).encode("utf-8")

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="Price Grid", index=False)
        CLUSTERS.to_excel(writer, sheet_name="Model Params", index=False)
        CALIBRATION_FACTORS.to_excel(writer, sheet_name="Calibration", index=False)
        scen_df.to_excel(writer, sheet_name="Scenarios", index=False)
    excel_buf.seek(0)

    c1e, c2e = st.columns(2)
    with c1e:
        st.download_button("⬇️ Download CSV",
                            data=csv_bytes,
                            file_name=f"optimal_prices_{objective}.csv",
                            mime="text/csv",
                            use_container_width=True)
    with c2e:
        st.download_button("⬇️ Download Excel (all sheets)",
                            data=excel_buf,
                            file_name=f"pricing_report_{objective}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)

    st.markdown("---")
    st.markdown("**Preview**")

    def safe_fmt(val):
        try:    return f"{float(val):,.4f}"
        except: return str(val) if val is not None else ""

    sar_cols = {c: safe_fmt for c in export.columns
                if any(x in c for x in ["IRR","Rate","β","γ","α","Factor"])}
    st.dataframe(
        export.style.format(sar_cols, na_rep="")
        .set_properties(**{"font-family":"monospace","font-size":"11px"}),
        use_container_width=True, height=350,
    )
