import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

base_path = Path(r"E:\IMD Project\Final Maps")

anomaly_file = base_path / "river_basin_monthly_rainfall_anomaly_1971_2020.csv"
cv_file = base_path / "river_basin_annual_coefficient_of_variation_1971_2020.csv"

output_file = (
    base_path /
    "ganga_subbasins_seasonal_rainfall_anomaly_1971_2020.png"
)

# ============================================================
# READ DATA
# ============================================================

anomaly_data = pd.read_csv(anomaly_file)
cv_data = pd.read_csv(cv_file)

# ============================================================
# STANDARDIZE IDs
# ============================================================

anomaly_data["Sub Basin ID"] = (
    anomaly_data["Sub Basin ID"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)

cv_data["subbasin_id"] = (
    cv_data["subbasin_id"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
)

# ============================================================
# EXACT 17 GANGA SUB-BASINS
# ============================================================

ganga_subbasins = {
    "10092": "Upper Ganga",
    "18153": "Ganga Narora to Phapham",
    "18154": "Ganga Phaphamau to Ball",
    "18156": "Gomti",
    "18162": "Sai",
    "22180": "Bagmati Adhwara",
    "22183": "Chandan",
    "22184": "Gandak",
    "22186": "Kiul",
    "22187": "Kosi",
    "22189": "Mahananda",
    "22191": "North Koel",
    "22192": "Punpun/Dhab Nadi",
    "22193": "Sone",
    "23195": "Kanhar",
    "6048": "Chhatnag to Mirzapur",
    "6056": "Upper Sone"
}

ganga_ids = list(ganga_subbasins.keys())

# ============================================================
# USE CV CSV AS MASTER SUB-BASIN LIST
# ============================================================

cv_ids = set(
    cv_data["subbasin_id"]
    .dropna()
    .astype(str)
)

ganga_data = anomaly_data[
    anomaly_data["Sub Basin ID"].isin(cv_ids)
].copy()

# ============================================================
# FILTER GANGA BASIN
# ============================================================

ganga_data = ganga_data[
    ganga_data["River Basin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains("ganga", na=False)
].copy()

# ============================================================
# SELECT EXACTLY THE 17 SUB-BASINS
# ============================================================

ganga_data = ganga_data[
    ganga_data["Sub Basin ID"].isin(ganga_ids)
].copy()

# ============================================================
# CONVERT YEAR TO NUMERIC
# ============================================================

ganga_data["Year"] = pd.to_numeric(
    ganga_data["Year"],
    errors="coerce"
)

# ============================================================
# CONVERT ANOMALY COLUMNS TO NUMERIC
# ============================================================

anomaly_columns = [
    "January_Anomaly",
    "February_Anomaly",
    "March_Anomaly",
    "April_Anomaly",
    "May_Anomaly",
    "June_Anomaly",
    "July_Anomaly",
    "August_Anomaly",
    "September_Anomaly",
    "October_Anomaly",
    "November_Anomaly",
    "December_Anomaly"
]

for column in anomaly_columns:
    ganga_data[column] = pd.to_numeric(
        ganga_data[column],
        errors="coerce"
    )

# ============================================================
# CALCULATE SEASONAL MEAN ANOMALIES
# ============================================================

ganga_data["JF"] = ganga_data[
    ["January_Anomaly", "February_Anomaly"]
].mean(axis=1)

ganga_data["MAM"] = ganga_data[
    ["March_Anomaly", "April_Anomaly", "May_Anomaly"]
].mean(axis=1)

ganga_data["JJAS"] = ganga_data[
    [
        "June_Anomaly",
        "July_Anomaly",
        "August_Anomaly",
        "September_Anomaly"
    ]
].mean(axis=1)

ganga_data["OND"] = ganga_data[
    [
        "October_Anomaly",
        "November_Anomaly",
        "December_Anomaly"
    ]
].mean(axis=1)

# ============================================================
# SORT DATA
# ============================================================

ganga_data = ganga_data.sort_values(
    ["Sub Basin ID", "Year"]
)

# ============================================================
# SEASONS
# ============================================================

seasons = {
    "JF": "JF (January–February)",
    "MAM": "MAM (March–May)",
    "JJAS": "JJAS (June–September)",
    "OND": "OND (October–December)"
}

# ============================================================
# FIXED COLORS FOR 17 SUB-BASINS
# SAME COLOR WILL BE USED IN ALL PANELS
# ============================================================

colors = plt.cm.tab20(
    np.linspace(0, 1, len(ganga_ids))
)

color_map = dict(
    zip(ganga_ids, colors)
)

# ============================================================
# CREATE 2 × 2 SEASONAL PANELS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(26, 18),
    sharex=True,
    sharey=True
)

axes = axes.flatten()

# ============================================================
# PLOT EACH SEASON
# ============================================================

for ax, (season, season_title) in zip(axes, seasons.items()):

    for subbasin_id in ganga_ids:

        sub_data = ganga_data[
            ganga_data["Sub Basin ID"] == subbasin_id
        ].sort_values("Year")

        if sub_data.empty:
            continue

        ax.plot(
            sub_data["Year"],
            sub_data[season],
            linewidth=1.6,
            color=color_map[subbasin_id],
            label=ganga_subbasins[subbasin_id]
        )

    # Zero anomaly reference line
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    # Title
    ax.set_title(
        season_title,
        fontsize=22,
        fontweight="bold"
    )

    # Grid
    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    # X-axis
    ax.set_xlim(1971, 2020)
    ax.set_xticks(
        np.arange(1971, 2021, 5)
    )

    ax.tick_params(
        axis="both",
        labelsize=17
    )

    # Border
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

# ============================================================
# AXIS LABELS
# ============================================================

axes[0].set_ylabel(
    "Seasonal Rainfall Anomaly (mm)",
    fontsize=20,
    fontweight="bold"
)

axes[2].set_ylabel(
    "Seasonal Rainfall Anomaly (mm)",
    fontsize=20,
    fontweight="bold"
)

axes[2].set_xlabel(
    "Year",
    fontsize=20,
    fontweight="bold"
)

axes[3].set_xlabel(
    "Year",
    fontsize=20,
    fontweight="bold"
)

# ============================================================
# ONE COMMON LEGEND FOR ALL 17 SUB-BASINS
# ============================================================

handles = []
labels = []

for subbasin_id in ganga_ids:

    handle, = axes[0].plot(
        [],
        [],
        color=color_map[subbasin_id],
        linewidth=2
    )

    handles.append(handle)
    labels.append(
        f"{ganga_subbasins[subbasin_id]} ({subbasin_id})"
    )

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    fontsize=15,
    frameon=True
)

# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "Seasonal Rainfall Anomaly of Ganga Sub-Basins (1971–2020)",
    fontsize=27,
    fontweight="bold",
    y=0.995
)

# ============================================================
# ADJUST LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.98,
    top=0.93,
    bottom=0.20,
    wspace=0.12,
    hspace=0.20
)

# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Plot saved successfully:")
print(output_file)

print("\nNumber of Ganga sub-basins plotted:", len(ganga_ids))

print("\nSub-basins:")
for subbasin_id in ganga_ids:
    print(
        subbasin_id,
        "-",
        ganga_subbasins[subbasin_id]
    )