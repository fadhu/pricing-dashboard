import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import io
from engine import (
    get_default_clusters, simulate_curve,
    run_optimisation, net_revenue, projected_sales,
    get_scenario_results,
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0a2540 0%, #1a4a6b 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #00b4d8;
}
.main-header h1 {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    font-family: 'IBM Plex Sans', sans-serif;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #90caf9;
    font-size: 0.9rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}

/* KPI cards */
.kpi-card {
    background: #0a2540;
    border: 1px solid #1a4a6b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-label {
    color: #90caf9;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.kpi-value {
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'IBM Plex Sans', sans-serif;
}
.kpi-delta-pos { color: #4caf50; font-size: 0.9rem; font-weight: 600; }
.kpi-delta-neg { color: #ef5350; font-size: 0.9rem; font-weight: 600; }
.kpi-delta-neu { color: #90caf9; font-size: 0.9rem; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a2540 !important;
}
section[data-testid="stSidebar"] * {
    color: #e0f0ff !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e0f0ff !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0f2f4a;
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #90caf9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #00b4d8 !important;
    color: #0a2540 !important;
    border-radius: 6px;
    font-weight: 700;
}

/* Dividers */
hr { border-color: #1a4a6b; }

/* Download button */
.stDownloadButton button {
    background: #00b4d8 !important;
    color: #0a2540 !important;
    font-weight: 700 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    width: 100%;
}

/* Segment badge */
.seg-badge {
    display: inline-block;
    background: #1a4a6b;
    color: #90caf9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💰 Pricing Optimisation Dashboard</h1>
    <p>Sales &amp; Net Revenue Based Pricing Strategy · Cluster-level price optimisation</p>
</div>
""", unsafe_allow_html=True)

# ── Matplotlib theme ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a2540",
    "axes.facecolor":    "#0f2f4a",
    "axes.edgecolor":    "#1a4a6b",
    "axes.labelcolor":   "#90caf9",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#90caf9",
    "ytick.color":       "#90caf9",
    "grid.color":        "#1a4a6b",
    "text.color":        "#ffffff",
    "legend.facecolor":  "#0a2540",
    "legend.edgecolor":  "#1a4a6b",
    "legend.labelcolor": "#e0f0ff",
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "font.family":       "monospace",
})

BLUE   = "#00b4d8"
RED    = "#ef5350"
GREEN  = "#4caf50"
AMBER  = "#ffb74d"
GREY   = "#546e7a"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    objective = st.radio(
        "Business objective",
        ["revenue", "volume"],
        format_func=lambda x: "📈 Maximise Net Revenue"
                               if x == "revenue" else "📦 Maximise Sales Volume",
    )

    st.markdown("---")
    st.markdown("**Optimisation bounds**")

    max_movement = st.slider(
        "Max total price movement (SAR)",
        100, 2000, 600, 50,
        help="Total absolute SAR change across all clusters",
    )
    price_range_pct = st.slider(
        "Price search range (±%)",
        5, 50, 30, 5,
        help="How far from current price the optimizer searches",
    ) / 100

    cost_adj = st.slider(
        "Unit cost adjustment (%)",
        -20, 20, 0, 1,
        help="Simulate funding cost changes",
    ) / 100

    st.markdown("---")
    st.markdown("**Min volume per cluster (units/week)**")

    clusters_default = get_default_clusters()
    min_volumes = {}
    for _, row in clusters_default.iterrows():
        mv = st.number_input(
            f"C{row['id']}: {row['label'][:22]}",
            min_value=0,
            max_value=int(row["baseline_sales"]),
            value=0, step=1,
            key=f"mv_{row['id']}",
        )
        if mv > 0:
            min_volumes[int(row["id"])] = mv

    st.markdown("---")
    st.caption("Built with scipy SLSQP · Power-law demand model")


# ══════════════════════════════════════════════════════════════════════════════
# RUN OPTIMISATION (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_results(objective, max_movement, price_range_pct, cost_adj, mv_tuple):
    mv = dict(mv_tuple)
    return run_optimisation(
        get_default_clusters(),
        objective=objective,
        min_volumes=mv if mv else None,
        max_movement=max_movement,
        cost_adjustment=cost_adj,
        price_range_pct=price_range_pct,
    )

results = get_results(
    objective, max_movement, price_range_pct, cost_adj,
    tuple(sorted(min_volumes.items())),
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
cur_rev  = results["current_revenue"].sum()
opt_rev  = results["optimal_revenue"].sum()
cur_vol  = results["current_sales"].sum()
opt_vol  = results["optimal_sales"].sum()
rev_d    = (opt_rev - cur_rev) / cur_rev * 100
vol_d    = (opt_vol - cur_vol) / cur_vol * 100
uplift   = opt_rev - cur_rev

def delta_html(val, prefix="", suffix="%"):
    cls = "kpi-delta-pos" if val >= 0 else "kpi-delta-neg"
    sign = "▲" if val >= 0 else "▼"
    return f'<div class="{cls}">{sign} {prefix}{abs(val):.1f}{suffix}</div>'

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Current Weekly Revenue</div>
        <div class="kpi-value">SAR {cur_rev:,.0f}</div>
        <div class="kpi-delta-neu">baseline</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Optimal Weekly Revenue</div>
        <div class="kpi-value">SAR {opt_rev:,.0f}</div>
        {delta_html(rev_d)}
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Weekly Revenue Uplift</div>
        <div class="kpi-value">SAR {uplift:,.0f}</div>
        {delta_html(rev_d)}
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Volume Change</div>
        <div class="kpi-value">{opt_vol:.0f} <span style="font-size:1rem;color:#90caf9">units</span></div>
        {delta_html(vol_d)}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📋  Price Grid",
    "📈  Demand Curves",
    "🔀  Scenarios",
    "⬇️  Export",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — PRICE GRID
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Optimised Price Grid")
    st.caption(f"Objective: **{objective.upper()}** · Max movement: SAR {max_movement} · "
               f"Cost adj: {cost_adj*100:+.0f}%")

    # ── Styled table ──────────────────────────────────────────────────────────
    display = results[[
        "label", "beta", "current_price", "optimal_price",
        "price_chg_pct", "current_sales", "optimal_sales",
        "current_revenue", "optimal_revenue", "revenue_uplift",
    ]].copy()
    display.columns = [
        "Segment", "β", "Current (SAR)", "Optimal (SAR)",
        "Δ Price %", "Cur Sales", "Opt Sales",
        "Cur Revenue", "Opt Revenue", "Uplift (SAR)",
    ]

    def style_delta(val):
        if isinstance(val, (int, float)):
            if val > 0:   return "color: #4caf50; font-weight: 600"
            elif val < 0: return "color: #ef5350; font-weight: 600"
        return ""

    def style_beta(val):
        if isinstance(val, float):
            intensity = min(abs(val) / 2.5, 1.0)
            r = int(10 + intensity * 60)
            g = int(47 + intensity * 30)
            b = int(74 + intensity * 20)
            return f"background-color: rgb({r},{g},{b})"
        return ""

    styled = (
        display.style
        .format({
            "β":            "{:.2f}",
            "Current (SAR)":"SAR {:.0f}",
            "Optimal (SAR)":"SAR {:.0f}",
            "Δ Price %":    "{:+.1f}%",
            "Cur Sales":    "{:.1f}",
            "Opt Sales":    "{:.1f}",
            "Cur Revenue":  "SAR {:,.0f}",
            "Opt Revenue":  "SAR {:,.0f}",
            "Uplift (SAR)": "SAR {:+,.0f}",
        })
        .applymap(style_delta, subset=["Δ Price %", "Uplift (SAR)"])
        .applymap(style_beta, subset=["β"])
        .set_properties(**{
            "font-family": "IBM Plex Mono, monospace",
            "font-size": "12px",
        })
    )

    st.dataframe(styled, use_container_width=True, height=320)

    # ── Price comparison chart ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0a2540")

    x      = np.arange(len(results))
    w      = 0.35
    labels = [r[:16] for r in results["label"]]

    # Prices
    axes[0].bar(x - w/2, results["current_price"], w,
                label="Current",  color=GREY,  alpha=0.9)
    axes[0].bar(x + w/2, results["optimal_price"], w,
                label="Optimal",  color=BLUE,  alpha=0.9)

    # Add price labels on bars
    for xi, (cp, op) in enumerate(zip(results["current_price"], results["optimal_price"])):
        axes[0].text(xi - w/2, cp + 5, f"{cp:.0f}",
                     ha="center", fontsize=6.5, color="#90caf9")
        col = GREEN if op > cp else RED
        axes[0].text(xi + w/2, op + 5, f"{op:.0f}",
                     ha="center", fontsize=6.5, color=col)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Price (SAR)")
    axes[0].set_title("Current vs Optimal Price per Cluster")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.4)

    # Revenue uplift waterfall
    uplifts = results["revenue_uplift"].values
    colors_u = [GREEN if u >= 0 else RED for u in uplifts]
    bars = axes[1].bar(x, uplifts, color=colors_u, alpha=0.9, edgecolor="#0a2540")
    axes[1].axhline(0, color="#90caf9", linewidth=0.8)
    for xi, u in enumerate(uplifts):
        axes[1].text(xi, u + (8 if u >= 0 else -18),
                     f"SAR {u:+.0f}",
                     ha="center", fontsize=6.5,
                     color=GREEN if u >= 0 else RED)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    axes[1].set_ylabel("Weekly Revenue Uplift (SAR)")
    axes[1].set_title("Revenue Uplift per Cluster")
    axes[1].grid(axis="y", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DEMAND CURVES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Revenue & Sales Curves")
    st.caption("Each curve sweeps price ±30% from current. "
               "Dashed = revenue-max | Dotted = current | Dash-dot = optimal")

    selected = st.multiselect(
        "Clusters to display",
        options=list(results["label"]),
        default=list(results["label"][:4]),
    )

    if not selected:
        st.info("Select at least one cluster above.")
    else:
        n_sel = len(selected)
        fig   = plt.figure(figsize=(15, 4.5 * n_sel))
        fig.patch.set_facecolor("#0a2540")
        gs    = gridspec.GridSpec(n_sel, 2, figure=fig,
                                  hspace=0.55, wspace=0.35)

        for pi, seg_label in enumerate(selected):
            row = results[results["label"] == seg_label].iloc[0]
            sim = simulate_curve(row, price_range_pct=price_range_pct)

            ax_s = fig.add_subplot(gs[pi, 0])
            ax_r = fig.add_subplot(gs[pi, 1])

            # Elasticity regime colour
            beta = float(row["beta"])
            if beta > -1:    regime_col = GREEN
            elif beta > -1.5: regime_col = AMBER
            else:             regime_col = RED

            # Sales curve
            ax_s.fill_between(sim["prices"], sim["sales"],
                               alpha=0.15, color=BLUE)
            ax_s.plot(sim["prices"], sim["sales"],
                      color=BLUE, linewidth=2.5)
            ax_s.axvline(float(row["current_price"]), color=GREY,
                         linestyle=":", linewidth=1.5, label="Current")
            ax_s.axvline(float(row["optimal_price"]), color=regime_col,
                         linestyle="-.", linewidth=1.8, label="Optimal")
            ax_s.set_xlabel("Price (SAR)"); ax_s.set_ylabel("Weekly Units")
            ax_s.set_title(f"{seg_label}  ·  β = {beta:.2f}  ·  Sales Curve")
            ax_s.legend(fontsize=8); ax_s.grid(alpha=0.3)

            # Revenue curve
            ax_r.fill_between(sim["prices"], sim["revenue"],
                               alpha=0.15, color=regime_col)
            ax_r.plot(sim["prices"], sim["revenue"],
                      color=regime_col, linewidth=2.5)
            ax_r.axvline(sim["rev_max_price"], color=AMBER,
                         linestyle="--", linewidth=1.8,
                         label=f"Rev-max SAR {sim['rev_max_price']:.0f}")
            ax_r.axvline(float(row["current_price"]), color=GREY,
                         linestyle=":", linewidth=1.5, label="Current")
            ax_r.axvline(float(row["optimal_price"]), color=BLUE,
                         linestyle="-.", linewidth=1.8, label="Optimal")

            # Shade opportunity area
            opt_p   = float(row["optimal_price"])
            cur_p   = float(row["current_price"])
            mask    = ((sim["prices"] >= min(opt_p, cur_p)) &
                       (sim["prices"] <= max(opt_p, cur_p)))
            ax_r.fill_between(sim["prices"][mask], sim["revenue"][mask],
                               alpha=0.25, color=GREEN,
                               label="Revenue opportunity")

            ax_r.set_xlabel("Price (SAR)"); ax_r.set_ylabel("Net Revenue (SAR)")
            ax_r.set_title(f"{seg_label}  ·  Revenue Curve")
            ax_r.legend(fontsize=7.5); ax_r.grid(alpha=0.3)

        plt.suptitle("Demand & Revenue Curves by Cluster",
                     fontsize=13, fontweight="bold", color="white", y=1.01)
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Scenario Comparison")
    st.caption("Best / Base / Downside / Worst — all inputs shift simultaneously")

    @st.cache_data
    def cached_scenarios(objective, max_movement, price_range_pct):
        return get_scenario_results(objective, max_movement, price_range_pct)

    scen_df = cached_scenarios(objective, max_movement, price_range_pct)

    # Table
    st.dataframe(
        scen_df.style
        .format({
            "Total Revenue": "SAR {:,.0f}",
            "Total Sales":   "{:.1f} units",
            "Avg Opt Price": "SAR {:.0f}",
            "vs Base (%)":   "{:+.1f}%",
        })
        .applymap(
            lambda v: ("color:#4caf50;font-weight:600" if isinstance(v, float) and v > 0
                       else "color:#ef5350;font-weight:600" if isinstance(v, float) and v < 0
                       else ""),
            subset=["vs Base (%)"]
        )
        .set_properties(**{"font-family": "IBM Plex Mono, monospace", "font-size": "12px"}),
        use_container_width=True,
    )

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0a2540")

    scen_colors = [GREEN, BLUE, AMBER, RED]
    base_rev    = scen_df[scen_df["Scenario"] == "⚪ Base case"]["Total Revenue"].values[0]

    # Revenue bar chart
    bars = axes[0].bar(
        range(len(scen_df)),
        scen_df["Total Revenue"],
        color=scen_colors, alpha=0.9, edgecolor="#0a2540", width=0.6,
    )
    axes[0].axhline(base_rev, color=GREY, linestyle="--",
                    linewidth=1.5, label="Base revenue")
    for bar, (_, row) in zip(bars, scen_df.iterrows()):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 30,
            f"{row['vs Base (%)']:+.1f}%",
            ha="center", fontsize=10, fontweight="bold",
            color=GREEN if row["vs Base (%)"] >= 0 else RED,
        )
    axes[0].set_xticks(range(len(scen_df)))
    axes[0].set_xticklabels(
        [s.split()[-1] for s in scen_df["Scenario"]],
        fontsize=9,
    )
    axes[0].set_ylabel("Total Weekly Net Revenue (SAR)")
    axes[0].set_title("Revenue by Scenario")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.4)

    # Revenue vs volume scatter
    axes[1].scatter(
        scen_df["Total Sales"],
        scen_df["Total Revenue"],
        c=scen_colors, s=180, zorder=5, edgecolors="#0a2540", linewidths=1.5,
    )
    for _, row in scen_df.iterrows():
        axes[1].annotate(
            row["Scenario"].split()[-1],
            (row["Total Sales"], row["Total Revenue"]),
            textcoords="offset points", xytext=(8, 4),
            fontsize=8, color="#e0f0ff",
        )
    # Connect the dots
    axes[1].plot(scen_df["Total Sales"], scen_df["Total Revenue"],
                 color=GREY, linewidth=1, linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Total Weekly Sales (units)")
    axes[1].set_ylabel("Total Weekly Net Revenue (SAR)")
    axes[1].set_title("Revenue–Volume Frontier by Scenario")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Robustness check
    st.markdown("#### Robustness Check")
    worst_rev = scen_df[scen_df["Scenario"] == "🔴 Worst case"]["Total Revenue"].values[0]
    cur_rev_check = results["current_revenue"].sum()

    if worst_rev > cur_rev_check:
        st.success(f"✅ **Robust** — even in the worst case, optimal revenue "
                   f"(SAR {worst_rev:,.0f}) exceeds current revenue "
                   f"(SAR {cur_rev_check:,.0f}). Proceed with confidence.")
    else:
        st.warning(f"⚠️ **Fragile** — in the worst case, optimal revenue "
                   f"(SAR {worst_rev:,.0f}) falls below current "
                   f"(SAR {cur_rev_check:,.0f}). Run A/B test before full rollout.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### Export Results")

    # ── Build export dataframe ────────────────────────────────────────────────
    export = results[[
        "label", "beta",
        "current_price", "optimal_price", "price_chg_pct",
        "current_sales", "optimal_sales",
        "current_revenue", "optimal_revenue", "revenue_uplift",
    ]].copy()
    export.columns = [
        "Segment", "Elasticity (β)",
        "Current Price (SAR)", "Optimal Price (SAR)", "Price Change (%)",
        "Current Weekly Sales", "Optimal Weekly Sales",
        "Current Weekly Revenue (SAR)", "Optimal Weekly Revenue (SAR)",
        "Revenue Uplift (SAR)",
    ]
    export["Objective"]          = objective.capitalize()
    export["Max Movement (SAR)"] = max_movement
    export["Cost Adj (%)"]       = cost_adj * 100
    export["Price Range (±%)"]   = price_range_pct * 100

    # ── Summary row ───────────────────────────────────────────────────────────
    summary_row = pd.DataFrame([{
        "Segment": "── TOTAL ──",
        "Elasticity (β)": "",
        "Current Price (SAR)": "",
        "Optimal Price (SAR)": "",
        "Price Change (%)": "",
        "Current Weekly Sales":          export["Current Weekly Sales"].sum(),
        "Optimal Weekly Sales":          export["Optimal Weekly Sales"].sum(),
        "Current Weekly Revenue (SAR)":  export["Current Weekly Revenue (SAR)"].sum(),
        "Optimal Weekly Revenue (SAR)":  export["Optimal Weekly Revenue (SAR)"].sum(),
        "Revenue Uplift (SAR)":          export["Revenue Uplift (SAR)"].sum(),
        "Objective": "",
        "Max Movement (SAR)": "",
        "Cost Adj (%)": "",
        "Price Range (±%)": "",
    }])
    export_full = pd.concat([export, summary_row], ignore_index=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_bytes = export_full.to_csv(index=False).encode("utf-8")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇️  Download Price Grid (CSV)",
            data=csv_bytes,
            file_name=f"optimal_price_grid_{objective}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Excel ─────────────────────────────────────────────────────────────────
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        export_full.to_excel(writer, sheet_name="Price Grid", index=False)
        scen_df_exp = cached_scenarios(objective, max_movement, price_range_pct)
        scen_df_exp.to_excel(writer, sheet_name="Scenarios", index=False)
    excel_buf.seek(0)

    with col_dl2:
        st.download_button(
            label="⬇️  Download Full Report (Excel)",
            data=excel_buf,
            file_name=f"pricing_report_{objective}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**Preview**")
    def safe_sar_fmt(val):
        try:
            return f"SAR {float(val):,.0f}"
        except (ValueError, TypeError):
            return val if val is not None else ""

    sar_cols = {c: safe_sar_fmt for c in export_full.columns if "SAR" in c}

    st.dataframe(
        export_full.style.format(
            sar_cols,
            na_rep="",
        ).set_properties(**{
            "font-family": "IBM Plex Mono, monospace",
            "font-size": "11px",
        }),
        use_container_width=True,
        height=360,
    )

    st.markdown("---")
    st.markdown("#### Settings used in this run")
    settings_md = f"""
| Setting | Value |
|---|---|
| Objective | {objective.capitalize()} |
| Max price movement | SAR {max_movement} |
| Price search range | ±{price_range_pct*100:.0f}% |
| Unit cost adjustment | {cost_adj*100:+.0f}% |
| Min volume constraints | {min_volumes if min_volumes else 'None'} |
| Solver | SLSQP (scipy) |
| Demand model | Power-law: Sales = Baseline × (P_new/P_cur)^β |
"""
    st.markdown(settings_md)
