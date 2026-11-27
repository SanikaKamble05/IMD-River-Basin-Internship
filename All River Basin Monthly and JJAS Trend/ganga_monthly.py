# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:09:16 2026

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    "ganga_subbasins_monthly_rainfall_anomaly_panel_1971_2020.png"
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
# 9. GET FINAL SUB-BASIN LIST
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
# 10. PRINT FINAL SUB-BASINS
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
# 11. MONTHLY ANOMALY COLUMNS
# ============================================================

month_columns = {

    "January": "January_Anomaly",
    "February": "February_Anomaly",
    "March": "March_Anomaly",
    "April": "April_Anomaly",
    "May": "May_Anomaly",
    "June": "June_Anomaly",
    "July": "July_Anomaly",
    "August": "August_Anomaly",
    "September": "September_Anomaly",
    "October": "October_Anomaly",
    "November": "November_Anomaly",
    "December": "December_Anomaly"

}


# ============================================================
# 12. CREATE MONTHLY TIME SERIES
# ============================================================

records = []

for _, row in ganga_data.iterrows():

    year = int(row["Year"])

    for month_number, (
        month_name,
        anomaly_column
    ) in enumerate(
        month_columns.items(),
        start=1
    ):

        records.append({

            "Sub Basin ID":
                row["Sub Basin ID"],

            "Sub Basin Name":
                row["Sub Basin Name"],

            "Date":
                pd.Timestamp(
                    year=year,
                    month=month_number,
                    day=1
                ),

            "Anomaly":
                pd.to_numeric(
                    row[anomaly_column],
                    errors="coerce"
                )

        })


ganga_monthly = pd.DataFrame(
    records
)


# ============================================================
# 13. SORT TIME SERIES
# ============================================================

ganga_monthly = (
    ganga_monthly
    .sort_values(
        [
            "Sub Basin ID",
            "Date"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. CHECK DATA
# ============================================================

print()
print("==========================================")
print("TIME SERIES CHECK")
print("==========================================")

print(
    "Number of Ganga sub-basins:",
    ganga_monthly["Sub Basin ID"].nunique()
)

print(
    "Total monthly records:",
    len(ganga_monthly)
)

print(
    "Date range:",
    ganga_monthly["Date"].min(),
    "to",
    ganga_monthly["Date"].max()
)

print(
    "Expected records per sub-basin:",
    50 * 12
)


# ============================================================
# 15. CREATE 2 × 2 PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(24, 16),
    sharex=True,
    sharey=True
)


# Convert axes to 1D array
axes = axes.flatten()


# ============================================================
# 16. PLOT EACH PANEL
# ============================================================

for ax, (panel_title, panel_ids) in zip(
    axes,
    panel_groups.items()
):

    for subbasin_id in panel_ids:

        subbasin_data = ganga_monthly[
            ganga_monthly["Sub Basin ID"]
            == subbasin_id
        ].copy()

        subbasin_data = (
            subbasin_data
            .sort_values("Date")
        )


        # Get sub-basin name
        name_data = ganga_subbasins[
            ganga_subbasins["Sub Basin ID"]
            == subbasin_id
        ]


        if len(name_data) == 0:
            continue


        subbasin_name = (
            name_data.iloc[0]["Sub Basin Name"]
        )


        # Plot line
        ax.plot(
            subbasin_data["Date"],
            subbasin_data["Anomaly"],
            linewidth=1.2,
            label=(
                str(subbasin_name)
                + " ("
                + str(subbasin_id)
                + ")"
            )
        )


    # --------------------------------------------------------
    # Zero anomaly line
    # --------------------------------------------------------

    ax.axhline(
        y=0,
        linewidth=1.3,
        linestyle="--"
    )


    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        panel_title,
        fontsize=18,
        fontweight="bold"
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        alpha=0.35
    )


    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=9,
        loc="upper right",
        frameon=True
    )


# ============================================================
# 17. X-AXIS
# ============================================================

for ax in axes:

    ax.set_xlim(
        pd.Timestamp("1971-01-01"),
        pd.Timestamp("2020-12-31")
    )

    ax.xaxis.set_major_locator(
        mdates.YearLocator(5)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    ax.tick_params(
        axis="x",
        labelsize=11
    )

    ax.tick_params(
        axis="y",
        labelsize=12
    )


# ============================================================
# 18. X-AXIS LABELS
# ============================================================

axes[2].set_xlabel(
    "Year",
    fontsize=17
)

axes[3].set_xlabel(
    "Year",
    fontsize=17
)


# ============================================================
# 19. Y-AXIS LABELS
# ============================================================

axes[0].set_ylabel(
    "Rainfall Anomaly (mm)",
    fontsize=17
)

axes[2].set_ylabel(
    "Rainfall Anomaly (mm)",
    fontsize=17
)


# ============================================================
# 20. OVERALL TITLE
# ============================================================

fig.suptitle(
    "Monthly Rainfall Anomaly of Ganga Basin Sub-basins (1971–2020)",
    fontsize=24,
    fontweight="bold"
)


# ============================================================
# 21. ADJUST LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
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
print("PANEL PLOT COMPLETED")
print("==========================================")

print(
    "Final number of Ganga sub-basins:",
    len(ganga_subbasins)
)

print(
    "Output plot:"
)

print(
    output_plot
)