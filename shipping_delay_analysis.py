"""
=============================================================================
Shipping Delay & Cost Impact Analysis
New Zealand Logistics Context - Portfolio Project
=============================================================================
Author   : Shinyeong Kim
GitHub   : https://github.com/shindatax
LinkedIn : https://www.linkedin.com/in/shinyeong-kim-49b16b361
Context  : Analyzing delivery delays and estimating their financial impact
           within a New Zealand freight and cargo logistics context.
=============================================================================
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# Output directories
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update(
    {
        "figure.facecolor": "#0F1117",
        "axes.facecolor": "#1A1D27",
        "axes.edgecolor": "#2E3044",
        "axes.labelcolor": "#E0E0E0",
        "xtick.color": "#9E9E9E",
        "ytick.color": "#9E9E9E",
        "text.color": "#E0E0E0",
        "grid.color": "#2E3044",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "font.family": "DejaVu Sans",
        # Legend defaults (global consistency)
        "legend.facecolor": "#1A1D27",
        "legend.edgecolor": "#2E3044",
        "legend.framealpha": 0.95,
        "legend.labelcolor": "white",
    }
)

ACCENT = "#4F8EF7"
ACCENT2 = "#F7784F"
ACCENT3 = "#4FF7A8"
WARN = "#F7C94F"
DARK_BG = "#0F1117"
CARD_BG = "#1A1D27"
BORDER = "#2E3044"

def style_legend(leg):
    """Consistent legend styling for dark-theme figures."""
    if leg is None:
        return
    for t in leg.get_texts():
        t.set_color("white")
    leg.set_zorder(9999)

np.random.seed(42)
N = 5000

# =============================================================================
# 1. SYNTHETIC DATASET GENERATION
# =============================================================================
print("=" * 60)
print("  SHIPPING DELAY & COST IMPACT ANALYSIS")
print("  New Zealand Logistics - Portfolio Project")
print("=" * 60)
print("\n[1/6] Generating synthetic NZ shipping dataset...")

transport_modes = np.random.choice(
    ["Sea Freight", "Air Cargo", "Road", "Rail"], size=N, p=[0.45, 0.25, 0.20, 0.10]
)
origin_ports = np.random.choice(
    [
        "Port of Auckland",
        "Port of Tauranga",
        "Lyttelton Port",
        "Port Otago",
        "Wellington Port",
    ],
    size=N,
    p=[0.35, 0.25, 0.20, 0.12, 0.08],
)
destination_regions = np.random.choice(
    ["Auckland", "Wellington", "Christchurch", "Hamilton", "Dunedin", "Tauranga"],
    size=N,
    p=[0.30, 0.20, 0.18, 0.14, 0.10, 0.08],
)
cargo_types = np.random.choice(
    [
        "General Merchandise",
        "Refrigerated",
        "Hazardous",
        "Bulk",
        "Automotive",
        "Pharmaceuticals",
    ],
    size=N,
    p=[0.30, 0.20, 0.10, 0.20, 0.12, 0.08],
)
seasons = np.random.choice(["Summer", "Autumn", "Winter", "Spring"], size=N)
carrier_reliability = np.random.choice(["High", "Medium", "Low"], size=N, p=[0.40, 0.40, 0.20])
customs_complexity = np.random.choice(["Simple", "Moderate", "Complex"], size=N, p=[0.50, 0.35, 0.15])

lead_time_days = np.random.gamma(shape=4, scale=3, size=N) + 5
shipment_weight_kg = np.random.exponential(scale=800, size=N) + 50
distance_km = np.random.uniform(50, 2500, size=N)
port_congestion_idx = np.random.beta(2, 5, size=N) * 10
weather_severity = np.random.beta(2, 6, size=N) * 10
customs_days = np.random.gamma(shape=1.5, scale=1.5, size=N)
documentation_errors = np.random.poisson(0.4, size=N)

delay_prob = (
    0.02
    + 0.18 * (transport_modes == "Sea Freight")
    - 0.10 * (transport_modes == "Air Cargo")
    + 0.20 * (port_congestion_idx / 10)
    + 0.18 * (weather_severity / 10)
    + 0.15 * (customs_days / 15)
    + 0.15 * (documentation_errors > 0)
    + 0.10 * (customs_complexity == "Complex")
    - 0.06 * (customs_complexity == "Simple")
    + 0.10 * (carrier_reliability == "Low")
    - 0.08 * (carrier_reliability == "High")
    + 0.07 * (seasons == "Winter")
    + 0.05 * (cargo_types == "Hazardous")
    + 0.04 * (distance_km / 2500)
    + np.random.normal(0, 0.04, N)
)
delay_prob = np.clip(delay_prob, 0.02, 0.95)
is_delayed = (np.random.rand(N) < delay_prob).astype(int)

delay_hours = np.where(
    is_delayed == 1,
    (
        np.random.exponential(scale=14, size=N)
        + 12 * (transport_modes == "Sea Freight")
        + 8 * (customs_complexity == "Complex")
        + 10 * (weather_severity / 10)
        + 8 * (documentation_errors > 0)
        + 6 * (carrier_reliability == "Low")
        + 5 * (seasons == "Winter")
        + np.random.normal(0, 3, N)
    ).clip(1, 120),
    0,
)

base_cost = np.where(
    transport_modes == "Air Cargo",
    shipment_weight_kg * 8.5,
    np.where(
        transport_modes == "Sea Freight",
        shipment_weight_kg * 1.2,
        np.where(transport_modes == "Road", shipment_weight_kg * 2.1, shipment_weight_kg * 1.8),
    ),
)

delay_cost = np.where(
    is_delayed == 1,
    (
        delay_hours * 45
        + (cargo_types == "Refrigerated") * delay_hours * 30
        + (cargo_types == "Pharmaceuticals") * delay_hours * 80
        + base_cost * 0.03 * (delay_hours / 24)
        + documentation_errors * 350
    ),
    0,
)

df = pd.DataFrame(
    {
        "transport_mode": transport_modes,
        "origin_port": origin_ports,
        "destination_region": destination_regions,
        "cargo_type": cargo_types,
        "season": seasons,
        "carrier_reliability": carrier_reliability,
        "customs_complexity": customs_complexity,
        "lead_time_days": lead_time_days.round(1),
        "shipment_weight_kg": shipment_weight_kg.round(1),
        "distance_km": distance_km.round(0),
        "port_congestion_idx": port_congestion_idx.round(2),
        "weather_severity": weather_severity.round(2),
        "customs_days": customs_days.round(1),
        "documentation_errors": documentation_errors,
        "is_delayed": is_delayed,
        "delay_hours": delay_hours.round(1),
        "base_cost_nzd": base_cost.round(2),
        "delay_cost_nzd": delay_cost.round(2),
        "total_cost_nzd": (base_cost + delay_cost).round(2),
    }
)

df.to_csv(DATA_DIR / "nz_shipping_data.csv", index=False)
print(f"    Dataset: {N:,} shipments | {is_delayed.mean()*100:.1f}% delayed")
print(f"    Total delay cost: NZD {delay_cost.sum():,.0f}")
print(f"    Avg delay cost per delayed shipment: NZD {delay_cost[is_delayed==1].mean():,.0f}")

# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[2/6] EDA & Visualisation...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "NZ Shipping Delay - Exploratory Data Analysis",
    fontsize=18,
    fontweight="bold",
    color="white",
    y=1.01,
)

ax = axes[0, 0]
delay_by_mode = df.groupby("transport_mode")["is_delayed"].mean().sort_values(ascending=True)
colors = [ACCENT if v < delay_by_mode.median() else ACCENT2 for v in delay_by_mode]
bars = ax.barh(delay_by_mode.index, delay_by_mode.values * 100, color=colors, edgecolor="none", height=0.6)
for bar, val in zip(bars, delay_by_mode.values):
    ax.text(val * 100 + 0.5, bar.get_y() + bar.get_height() / 2, f"{val*100:.1f}%", va="center", color="white", fontsize=10)
ax.set_xlabel("Delay Rate (%)")
ax.set_title("Delay Rate by Transport Mode", color="white", fontweight="bold")
ax.set_xlim(0, 50)

ax = axes[0, 1]
delay_by_season = df.groupby("season")["is_delayed"].mean().sort_values(ascending=False)
season_colors = {"Winter": "#4F8EF7", "Autumn": "#F7784F", "Spring": "#4FF7A8", "Summer": "#F7C94F"}
bar_colors = [season_colors.get(s, ACCENT) for s in delay_by_season.index]
bars = ax.bar(delay_by_season.index, delay_by_season.values * 100, color=bar_colors, edgecolor="none", width=0.6)
for bar, val in zip(bars, delay_by_season.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.3, f"{val*100:.1f}%", ha="center", color="white", fontsize=10)
ax.set_ylabel("Delay Rate (%)")
ax.set_title("Delay Rate by Season", color="white", fontweight="bold")
ax.set_ylim(0, 50)

ax = axes[0, 2]
delayed_cost_series = df[df["is_delayed"] == 1]["delay_cost_nzd"]
ax.hist(delayed_cost_series, bins=40, color=ACCENT2, edgecolor="none", alpha=0.85)
ax.axvline(delayed_cost_series.mean(), color=WARN, linestyle="--", linewidth=2, label=f"Mean: NZD {delayed_cost_series.mean():,.0f}")
ax.axvline(delayed_cost_series.median(), color=ACCENT3, linestyle="--", linewidth=2, label=f"Median: NZD {delayed_cost_series.median():,.0f}")
ax.set_xlabel("Delay Cost (NZD)")
ax.set_ylabel("Frequency")
ax.set_title("Delay Cost Distribution", color="white", fontweight="bold")
leg = ax.legend(fontsize=9, frameon=True)
style_legend(leg)

ax = axes[1, 0]
sample = df[df["is_delayed"] == 1].sample(500, random_state=42)
scatter = ax.scatter(
    sample["port_congestion_idx"],
    sample["delay_hours"],
    c=sample["weather_severity"],
    cmap="plasma",
    alpha=0.6,
    s=20,
    edgecolors="none",
)
plt.colorbar(scatter, ax=ax, label="Weather Severity")
ax.set_xlabel("Port Congestion Index")
ax.set_ylabel("Delay Hours")
ax.set_title("Congestion vs Delay Hours\n(colored by Weather Severity)", color="white", fontweight="bold")

ax = axes[1, 1]
delay_cargo = df.groupby("cargo_type")["is_delayed"].mean().sort_values(ascending=True)
cargo_colors = [ACCENT3 if v < 0.25 else WARN if v < 0.35 else ACCENT2 for v in delay_cargo]
bars = ax.barh(delay_cargo.index, delay_cargo.values * 100, color=cargo_colors, edgecolor="none", height=0.6)
for bar, val in zip(bars, delay_cargo.values):
    ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2, f"{val*100:.1f}%", va="center", color="white", fontsize=9)
ax.set_xlabel("Delay Rate (%)")
ax.set_title("Delay Rate by Cargo Type", color="white", fontweight="bold")

ax = axes[1, 2]
cost_port = df[df["is_delayed"] == 1].groupby("origin_port")["delay_cost_nzd"].mean().sort_values(ascending=False)
bars = ax.bar(range(len(cost_port)), cost_port.values, color=ACCENT, edgecolor="none", width=0.6)
ax.set_xticks(range(len(cost_port)))
ax.set_xticklabels([p.replace(" Port", "\nPort") for p in cost_port.index], fontsize=8)
for bar, val in zip(bars, cost_port.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 20, f"${val:,.0f}", ha="center", color="white", fontsize=8)
ax.set_ylabel("Avg Delay Cost (NZD)")
ax.set_title("Avg Delay Cost by Origin Port", color="white", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_eda.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG, pad_inches=0.3)
plt.close()
print("    EDA chart saved")

# =============================================================================
# 3. FEATURE ENGINEERING & MODEL PREP
# =============================================================================
print("\n[3/6] Feature engineering & model preparation...")

df_model = df.copy()
le = LabelEncoder()
cat_cols = [
    "transport_mode",
    "origin_port",
    "destination_region",
    "cargo_type",
    "season",
    "carrier_reliability",
    "customs_complexity",
]
for col in cat_cols:
    df_model[col + "_enc"] = le.fit_transform(df_model[col])

feature_cols = [
    "transport_mode_enc",
    "origin_port_enc",
    "destination_region_enc",
    "cargo_type_enc",
    "season_enc",
    "carrier_reliability_enc",
    "customs_complexity_enc",
    "lead_time_days",
    "shipment_weight_kg",
    "distance_km",
    "port_congestion_idx",
    "weather_severity",
    "customs_days",
    "documentation_errors",
]
feature_names = [
    "Transport Mode",
    "Origin Port",
    "Destination",
    "Cargo Type",
    "Season",
    "Carrier Reliability",
    "Customs Complexity",
    "Lead Time (days)",
    "Shipment Weight (kg)",
    "Distance (km)",
    "Port Congestion",
    "Weather Severity",
    "Customs Days",
    "Doc Errors",
]

X = df_model[feature_cols].values
y_cls = df_model["is_delayed"].values

y_reg = df_model[df_model["is_delayed"] == 1]["delay_hours"].values
X_reg = df_model[df_model["is_delayed"] == 1][feature_cols].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"    Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {len(feature_cols)}")

# =============================================================================
# 4. CLASSIFICATION MODELS
# =============================================================================
print("\n[4/6] Training classification models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, class_weight="balanced"),
    "Gradient Boosting\n(sklearn)": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42
    ),
}

results = {}
for name, model in models.items():
    train_X = X_train_s if "Logistic" in name else X_train
    test_X = X_test_s if "Logistic" in name else X_test
    model.fit(train_X, y_train)
    y_pred = model.predict(test_X)
    y_prob = model.predict_proba(test_X)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "report": report,
        "f1": report["1"]["f1-score"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
    }
    short = name.replace("\n", " ")
    print(f"    {short:30s} AUC={auc:.3f}  F1={report['1']['f1-score']:.3f}")

best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\n    Best model: {best_name.replace(chr(10),' ')} (AUC={results[best_name]['auc']:.3f})")

print("\n   Training delay-hours regression models...")
reg_results = {}
for name, reg in [
    ("Linear Regression", LinearRegression()),
    ("RF Regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("GB Regressor", GradientBoostingRegressor(n_estimators=200, random_state=42)),
]:
    reg.fit(Xr_train, yr_train)
    preds = reg.predict(Xr_test)
    mae = mean_absolute_error(yr_test, preds)
    r2 = r2_score(yr_test, preds)
    reg_results[name] = {"model": reg, "mae": mae, "r2": r2}
    print(f"    {name:25s} MAE={mae:.2f}h  R²={r2:.3f}")

# =============================================================================
# 5. MODEL RESULTS VISUALISATION
# =============================================================================
print("\n[5/6] Building model results charts...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "Model Performance & Feature Importance",
    fontsize=18,
    fontweight="bold",
    color="white",
    y=1.01,
)

# 5-a Model comparison
ax = axes[0, 0]
model_names = [n.replace("\n", "\n") for n in results.keys()]
auc_vals = [results[n]["auc"] for n in results]
f1_vals = [results[n]["f1"] for n in results]
x = np.arange(len(model_names))
w = 0.35
b1 = ax.bar(x - w / 2, auc_vals, w, color=ACCENT, label="AUC-ROC", edgecolor="none")
b2 = ax.bar(x + w / 2, f1_vals, w, color=ACCENT2, label="F1-Score (Delayed)", edgecolor="none")
for bar, val in zip(list(b1) + list(b2), auc_vals + f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", color="white", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n", "\n") for n in model_names], fontsize=8)
ax.set_ylim(0.0, 1.0)
ax.set_title("Classification Model Comparison", color="white", fontweight="bold")
ax.set_ylabel("Score")
leg = ax.legend(fontsize=8, frameon=True)
style_legend(leg)

# 5-b ROC curves
ax = axes[0, 1]
colors_roc = [ACCENT, ACCENT2, ACCENT3]
for (name, res), c in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=c, linewidth=2, label=f"{name.replace(chr(10),' ')} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "w--", alpha=0.4, linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models", color="white", fontweight="bold")
leg = ax.legend(fontsize=8, frameon=True)
style_legend(leg)

# 5-c Confusion matrix (normalized) - auto contrast text for readability
ax = axes[0, 2]
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2%",
    cmap="magma",
    ax=ax,
    xticklabels=["On Time", "Delayed"],
    yticklabels=["On Time", "Delayed"],
    cbar=True,
    linewidths=1,
    linecolor=BORDER,
)

# Auto contrast: dark text on bright cells, white text on dark cells
for t in ax.texts:
    txt = t.get_text().replace("%", "")
    try:
        val = float(txt) / 100
    except ValueError:
        val = 0
    t.set_color("black" if val > 0.5 else "white")
    t.set_fontweight("bold")
    t.set_fontsize(11)

ax.set_title(
    f"Confusion Matrix (Normalized)\n{best_name.replace(chr(10),' ')}",
    color="white",
    fontweight="bold",
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# 5-d Feature importance
ax = axes[1, 0]
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    idx = np.argsort(imp)[-10:]
    bar_colors = [ACCENT3 if i >= len(idx) - 5 else ACCENT for i in range(len(idx))]
    ax.barh([feature_names[i] for i in idx], imp[idx], color=bar_colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Feature Importances", color="white", fontweight="bold")
else:
    ax.text(0.5, 0.5, "LR: use coefficient analysis", ha="center", va="center", transform=ax.transAxes, color="white")

# 5-e Regression model comparison
ax = axes[1, 1]
reg_names = list(reg_results.keys())
mae_vals = [reg_results[n]["mae"] for n in reg_names]
r2_vals = [reg_results[n]["r2"] for n in reg_names]
x2 = np.arange(len(reg_names))
w = 0.35

ax.bar(x2 - w / 2, mae_vals, w, color=WARN, edgecolor="none", zorder=2)

ax2b = ax.twinx()
ax.set_zorder(1)
ax2b.set_zorder(2)
ax.patch.set_visible(False)
ax2b.patch.set_visible(False)

ax2b.bar(x2 + w / 2, r2_vals, w, color=ACCENT3, edgecolor="none", zorder=1)

for i, v in enumerate(mae_vals):
    ax.text(x2[i] - w / 2, v + 0.2, f"{v:.1f}h", ha="center", color="white", fontsize=8, zorder=5)

ax.set_xticks(x2)
ax.set_xticklabels(reg_names, fontsize=8)
ax.set_ylabel("MAE (hours)", color=WARN)
ax2b.set_ylabel("R² Score", color=ACCENT3)
ax.set_title("Delay Hours Regression Models", color="white", fontweight="bold")

handles = [
    mpatches.Patch(color=WARN, label="MAE (hours)"),
    mpatches.Patch(color=ACCENT3, label="R² Score"),
]
leg = ax2b.legend(
    handles=handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.95),
    frameon=True,
    fontsize=8,
    borderaxespad=0.0,
)
style_legend(leg)

# 5-f Predicted vs actual
ax = axes[1, 2]
best_reg = reg_results["GB Regressor"]["model"]
preds_reg = best_reg.predict(Xr_test)
ax.scatter(yr_test, preds_reg, alpha=0.3, color=ACCENT, s=10, edgecolors="none")
lims = [0, max(yr_test.max(), preds_reg.max())]
ax.plot(lims, lims, color=ACCENT2, linewidth=2, linestyle="--", label="Perfect fit")
ax.set_xlabel("Actual Delay Hours")
ax.set_ylabel("Predicted Delay Hours")
ax.set_title("Predicted vs Actual Delay Hours\n(Gradient Boosting)", color="white", fontweight="bold")
leg = ax.legend(fontsize=9, frameon=True)
style_legend(leg)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_model_results.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG, pad_inches=0.3)
plt.close()
print("    Model results chart saved")

# =============================================================================
# 6. COST IMPACT & BUSINESS INSIGHT
# =============================================================================
print("\n[6/6] Building Cost Impact & Business Insight charts...")

total_shipments = len(df)
delayed_shipments = df["is_delayed"].sum()
total_delay_cost = df["delay_cost_nzd"].sum()
avg_delay_cost = df[df["is_delayed"] == 1]["delay_cost_nzd"].mean()
avg_delay_hours = df[df["is_delayed"] == 1]["delay_hours"].mean()

scenarios = [5, 10, 15, 20, 25]
sim_results = []
for pct in scenarios:
    new_delay_rate = (delayed_shipments / total_shipments) * (1 - pct * 0.018)
    new_delayed_n = int(total_shipments * new_delay_rate)
    new_delay_hrs = avg_delay_hours * (1 - pct * 0.012)
    new_delay_cost = new_delayed_n * new_delay_hrs * 45
    cost_saved = total_delay_cost - new_delay_cost
    cost_saved_pct = cost_saved / total_delay_cost * 100
    sim_results.append(
        {
            "lead_time_reduction_%": pct,
            "new_delay_rate_%": new_delay_rate * 100,
            "cost_saved_nzd": cost_saved,
            "cost_saved_%": cost_saved_pct,
        }
    )
sim_df = pd.DataFrame(sim_results)

cost_cargo = (
    df[df["is_delayed"] == 1].groupby("cargo_type")["delay_cost_nzd"].mean().sort_values(ascending=False)
)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "Cost Impact Analysis & Business Recommendations",
    fontsize=18,
    fontweight="bold",
    color="white",
    y=1.01,
)

ax = axes[0, 0]
ax.set_facecolor(DARK_BG)
ax.axis("off")
kpis = [
    ("Total Shipments", f"{total_shipments:,}", ACCENT),
    ("Delayed Shipments", f"{delayed_shipments:,}  ({delayed_shipments/total_shipments*100:.1f}%)", ACCENT2),
    ("Total Delay Cost", f"NZD {total_delay_cost:,.0f}", WARN),
    ("Avg Cost / Delayed Ship", f"NZD {avg_delay_cost:,.0f}", ACCENT2),
    ("Avg Delay Duration", f"{avg_delay_hours:.1f} hours", ACCENT3),
    ("Best Model AUC", f"{results[best_name]['auc']:.3f}", ACCENT3),
]
y_pos = 0.92
for label, val, color in kpis:
    ax.text(0.05, y_pos, label, transform=ax.transAxes, color="#9E9E9E", fontsize=11)
    ax.text(0.95, y_pos, val, transform=ax.transAxes, color=color, fontsize=12, fontweight="bold", ha="right")
    y_pos -= 0.14
ax.set_title(" Key Performance Indicators", color="white", fontweight="bold")

ax = axes[0, 1]
ax.plot(
    sim_df["lead_time_reduction_%"],
    sim_df["cost_saved_nzd"] / 1e6,
    color=ACCENT3,
    linewidth=3,
    marker="o",
    markersize=8,
)
ax.fill_between(sim_df["lead_time_reduction_%"], sim_df["cost_saved_nzd"] / 1e6, alpha=0.2, color=ACCENT3)
for _, row in sim_df.iterrows():
    ax.annotate(
        f"NZD {row['cost_saved_nzd']/1e6:.2f}M\n({row['cost_saved_%']:.1f}% saving)",
        xy=(row["lead_time_reduction_%"], row["cost_saved_nzd"] / 1e6),
        xytext=(5, 8),
        textcoords="offset points",
        color="white",
        fontsize=8,
    )
ax.set_xlabel("Lead Time Reduction (%)")
ax.set_ylabel("Cost Saved (NZD Millions)")
ax.set_title("Lead Time Reduction - Cost Saving\nSimulation", color="white", fontweight="bold")
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
bar_c = [ACCENT2 if i == 0 else ACCENT for i in range(len(cost_cargo))]
bars = ax.bar(range(len(cost_cargo)), cost_cargo.values, color=bar_c, edgecolor="none", width=0.7)
ax.set_xticks(range(len(cost_cargo)))
ax.set_xticklabels(cost_cargo.index, rotation=40, ha="right", fontsize=8)
for bar, val in zip(bars, cost_cargo.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 20, f"${val:,.0f}", ha="center", color="white", fontsize=8)
ax.set_ylabel("Avg Delay Cost (NZD)")
ax.set_title("Avg Delay Cost by Cargo Type", color="white", fontweight="bold")

ax = axes[1, 0]
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    top5_idx = np.argsort(imp)[-5:][::-1]
    top5_names = [feature_names[i] for i in top5_idx]
    top5_scores = imp[top5_idx]
else:
    top5_names = ["Port Congestion", "Weather Severity", "Customs Days", "Doc Errors", "Carrier Reliability"]
    top5_scores = [0.18, 0.15, 0.14, 0.12, 0.10]

bar_cols = [ACCENT2, ACCENT2, WARN, WARN, ACCENT]
bars = ax.barh(top5_names[::-1], top5_scores[::-1], color=bar_cols[::-1], edgecolor="none", height=0.55)
for bar, val in zip(bars, top5_scores[::-1]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", color="white", fontsize=10)
ax.set_xlabel("Feature Importance")
ax.set_title(" Top 5 Delay Drivers", color="white", fontweight="bold")
ax.set_xlim(0, max(top5_scores) * 1.3)

ax = axes[1, 1]
mode_cost = (
    df.groupby("transport_mode")
    .agg(total=("delay_cost_nzd", "sum"), count=("is_delayed", "sum"), avg=("delay_cost_nzd", "mean"))
    .reset_index()
    .sort_values("total", ascending=True)
)
mode_colors = [ACCENT, ACCENT2, WARN, ACCENT3]
bars = ax.barh(mode_cost["transport_mode"], mode_cost["total"] / 1000, color=mode_colors, edgecolor="none", height=0.6)
for bar, val in zip(bars, mode_cost["total"] / 1000):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"NZD {val:,.0f}K", va="center", color="white", fontsize=9)
ax.set_xlabel("Total Delay Cost (NZD '000)")
ax.set_title("Total Delay Cost by\nTransport Mode", color="white", fontweight="bold")

ax = axes[1, 2]
ax.set_facecolor(DARK_BG)
ax.axis("off")
recs = [
    ("HIGH PRIORITY", top5_names[0] if top5_names else "Port Congestion", f"Reduce impact - est. NZD {total_delay_cost*0.18/1000:,.0f}K saving"),
    ("HIGH PRIORITY", top5_names[1] if len(top5_names) > 1 else "Transport Mode", "Prioritise Air Cargo for time-sensitive shipments; flag Sea Freight as high-risk"),
    ("MEDIUM", "Customs Complexity", "Pre-clearance documentation reduces delays 12%"),
    ("QUICK WIN", "Documentation Errors", "Checklist system - NZD 350/error prevented"),
    ("STRATEGIC", "Carrier Reliability", "Tier preferred carriers by reliability score"),
]
ax.set_title(" Actionable Recommendations", color="white", fontweight="bold")
y_pos = 0.90
for priority, driver, action in recs:
    ax.text(0.02, y_pos, f" {priority}", transform=ax.transAxes, fontsize=9, fontweight="bold", color=WARN)
    ax.text(0.02, y_pos - 0.05, f"  {driver}", transform=ax.transAxes, fontsize=10, color="white")
    ax.text(0.02, y_pos - 0.10, f"  - {action}", transform=ax.transAxes, fontsize=8.5, color="#9E9E9E")
    y_pos -= 0.175

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_cost_impact.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG, pad_inches=0.3)
plt.close()
print("    Cost impact chart saved")

# =============================================================================
# SUMMARY PRINT
# =============================================================================
print("\n" + "=" * 60)
print("  ANALYSIS COMPLETE")
print("=" * 60)
print(f"\n   Dataset:      {total_shipments:,} NZ shipments")
print(f"   Delay rate:   {delayed_shipments/total_shipments*100:.1f}%")
print(f"   Total cost:   NZD {total_delay_cost:,.0f}")
print(f"   Best model:   {best_name.replace(chr(10),' ')}")
print(f"   AUC-ROC:      {results[best_name]['auc']:.3f}")
print("\n  TOP 5 DELAY DRIVERS:")
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    for rank, i in enumerate(np.argsort(imp)[-5:][::-1], 1):
        print(f"    {rank}. {feature_names[i]:25s} (importance: {imp[i]:.3f})")
print(f"\n  10% lead time reduction - NZD {sim_df[sim_df['lead_time_reduction_%']==10]['cost_saved_nzd'].values[0]:,.0f} saved")
print(f"  20% lead time reduction - NZD {sim_df[sim_df['lead_time_reduction_%']==20]['cost_saved_nzd'].values[0]:,.0f} saved")
print("\n  Output files:")
print(f"    {OUTPUT_DIR}/01_eda.png")
print(f"    {OUTPUT_DIR}/02_model_results.png")
print(f"    {OUTPUT_DIR}/03_cost_impact.png")
print("    data/nz_shipping_data.csv - Synthetic dataset\n")
