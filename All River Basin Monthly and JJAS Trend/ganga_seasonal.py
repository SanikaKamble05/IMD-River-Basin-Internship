# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:24:12 2026

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

base_path = Path(
    r"E:\IMD Project\Final Maps"
)

anomaly_file = (
    base_path /
    "river_basin_monthly_rainfall_anomaly_1971_2020.csv"
)

cv_file = (
    base_path /
    "river_basin_annual_coefficient_of_variation_1971_2020.csv"
)

output_plot = (
    base_path /
    "ganga_subbasins_seasonal_rainfall_anomaly_1971_2020.png"
)


# ============================================================
# 2. READ FILES
# ============================================================

anomaly_data = pd.read_csv(
    anomaly_file
)

cv_data = pd.read_csv(
    cv_file
)


# ============================================================
# 3. STANDARDIZE SUB-BASIN IDs
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
# 4. TAKE ONLY SUB-BASINS PRESENT IN CV CSV
# ============================================================

cv_ids = set(
    cv_data["subbasin_id"]
    .dropna()
)

ganga_data = anomaly_data[
    anomaly_data["Sub Basin ID"].isin(cv_ids)
].copy()


# ============================================================
# 5. KEEP ONLY GANGA BASIN
# ============================================================

ganga_data = ganga_data[
    ganga_data["River Basin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains("ganga", na=False)
].copy()


# ============================================================
# 6. DEFINE THE 17 GANGA SUB-BASINS
# ============================================================

panel_groups = {

    "Panel A - Main Ganga": [
        "10092",
        "18153",
        "18154",
        "6048"
    ],

    "Panel B - Western/Central Tributaries": [
        "18156",
        "18162",
        "22183",
        "22186"
    ],

    "Panel C - Eastern Tributaries": [
        "22180",
        "22184",
        "22187",
        "22189",
        "22193"
    ],

    "Panel D - Southern/Southwestern Tributaries": [
        "22191",
        "22192",
        "23195",
        "6056"
    ]
}


# ============================================================
# 7. COMBINE ALL 17 IDs
# ============================================================

selected_ids = []

for ids in panel_groups.values():
    selected_ids.extend(ids)

selected_ids = set(selected_ids)


# ============================================================
# 8. KEEP ONLY THE 17 SELECTED SUB-BASINS
# ============================================================

ganga_data = ganga_data[
    ganga_data["Sub Basin ID"].isin(selected_ids)
].copy()


# ============================================================
# 9. GET SUB-BASIN NAMES
# ============================================================

ganga_subbasins = (
    ganga_data[
        [
            "Sub Basin ID",
            "Sub Basin Name"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


# ============================================================
# 10. PRINT SUB-BASIN CHECK
# ============================================================

print()
print("==========================================")
print("FINAL GANGA SUB-BASINS")
print("==========================================")

print(
    ganga_subbasins.to_string(index=False)
)

print()
print(
    "Number of Ganga sub-basins:",
    len(ganga_subbasins)
)


# ============================================================
# 11. CONVERT MONTHLY ANOMALIES TO NUMERIC
# ============================================================

monthly_anomaly_columns = [
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

for column in monthly_anomaly_columns:

    ganga_data[column] = pd.to_numeric(
        ganga_data[column],
        errors="coerce"
    )


# ============================================================
# 12. CALCULATE SEASONAL MEAN ANOMALIES
# ============================================================

ganga_data["JF"] = ganga_data[
    [
        "January_Anomaly",
        "February_Anomaly"
    ]
].mean(axis=1)


ganga_data["MAM"] = ganga_data[
    [
        "March_Anomaly",
        "April_Anomaly",
        "May_Anomaly"
    ]
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
# 13. CREATE SEASONAL DATAFRAME
# ============================================================

seasonal_data = ganga_data[
    [
        "Sub Basin ID",
        "Sub Basin Name",
        "Year",
        "JF",
        "MAM",
        "JJAS",
        "OND"
    ]
].copy()


seasonal_data = seasonal_data.sort_values(
    [
        "Sub Basin ID",
        "Year"
    ]
).reset_index(drop=True)


# ============================================================
# 14. CHECK SEASONAL DATA
# ============================================================

print()
print("==========================================")
print("SEASONAL DATA CHECK")
print("==========================================")

print(
    "Number of sub-basins:",
    seasonal_data["Sub Basin ID"].nunique()
)

print(
    "Years:",
    seasonal_data["Year"].min(),
    "to",
    seasonal_data["Year"].max()
)

print(
    "Total seasonal records:",
    len(seasonal_data)
)

print(
    "Expected records:",
    17 * 50
)


# ============================================================
# 15. CREATE 2 × 2 PANEL PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(28, 18),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


# ============================================================
# 16. PLOT EACH PANEL
# ============================================================

for ax, (panel_title, panel_ids) in zip(
    axes,
    panel_groups.items()
):

    for subbasin_id in panel_ids:

        subbasin_data = seasonal_data[
            seasonal_data["Sub Basin ID"]
            == subbasin_id
        ].copy()

        subbasin_data = (
            subbasin_data
            .sort_values("Year")
        )


        # Get sub-basin name

        name_data = ganga_subbasins[
            ganga_subbasins["Sub Basin ID"]
            == subbasin_id
        ]


        if len(name_data) == 0:
            continue


        subbasin_name = str(
            name_data.iloc[0]["Sub Basin Name"]
        )


        # ----------------------------------------------------
        # Plot JF
        # ----------------------------------------------------

        ax.plot(
            subbasin_data["Year"],
            subbasin_data["JF"],
            linewidth=1.6,
            label=f"{subbasin_name} ({subbasin_id})"
        )


        # ----------------------------------------------------
        # Plot MAM
        # ----------------------------------------------------

        ax.plot(
            subbasin_data["Year"],
            subbasin_data["MAM"],
            linewidth=1.6,
            linestyle="--"
        )


        # ----------------------------------------------------
        # Plot JJAS
        # ----------------------------------------------------

        ax.plot(
            subbasin_data["Year"],
            subbasin_data["JJAS"],
            linewidth=1.6,
            linestyle="-."
        )


        # ----------------------------------------------------
        # Plot OND
        # ----------------------------------------------------

        ax.plot(
            subbasin_data["Year"],
            subbasin_data["OND"],
            linewidth=1.6,
            linestyle=":"
        )


    # ========================================================
    # ZERO ANOMALY LINE
    # ========================================================

    ax.axhline(
        y=0,
        linewidth=1.5,
        linestyle="--"
    )


    # ========================================================
    # PANEL TITLE
    # ========================================================

    ax.set_title(
        panel_title,
        fontsize=20,
        fontweight="bold",
        pad=12
    )


    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )


# ============================================================
# 17. AXIS LIMITS
# ============================================================

for ax in axes:

    ax.set_xlim(
        1971,
        2020
    )

    ax.tick_params(
        axis="both",
        labelsize=13
    )


# ============================================================
# 18. AXIS LABELS
# ============================================================

axes[2].set_xlabel(
    "Year",
    fontsize=18
)

axes[3].set_xlabel(
    "Year",
    fontsize=18
)

axes[0].set_ylabel(
    "Seasonal Rainfall Anomaly (mm)",
    fontsize=18
)

axes[2].set_ylabel(
    "Seasonal Rainfall Anomaly (mm)",
    fontsize=18
)


# ============================================================
# 19. SEASON LEGEND
# ============================================================

season_handles = [

    plt.Line2D(
        [0],
        [0],
        linewidth=2,
        linestyle="-",
        label="JF (Jan–Feb)"
    ),

    plt.Line2D(
        [0],
        [0],
        linewidth=2,
        linestyle="--",
        label="MAM (Mar–May)"
    ),

    plt.Line2D(
        [0],
        [0],
        linewidth=2,
        linestyle="-.",
        label="JJAS (Jun–Sep)"
    ),

    plt.Line2D(
        [0],
        [0],
        linewidth=2,
        linestyle=":",
        label="OND (Oct–Dec)"
    )

]


fig.legend(
    handles=season_handles,
    loc="upper center",
    ncol=4,
    fontsize=14,
    frameon=True,
    bbox_to_anchor=(0.5, 0.965)
)


# ============================================================
# 20. OVERALL TITLE
# ============================================================

fig.suptitle(
    "Seasonal Rainfall Anomaly of Ganga Basin Sub-basins (1971–2020)",
    fontsize=26,
    fontweight="bold",
    y=0.995
)


# ============================================================
# 21. LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# 22. SAVE FIGURE
# ============================================================

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 23. SHOW FIGURE
# ============================================================

plt.show()


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("SEASONAL PANEL PLOT COMPLETED")
print("==========================================")

print(
    "Final number of sub-basins:",
    len(ganga_subbasins)
)

print(
    "Output plot:"
)

print(
    output_plot
)