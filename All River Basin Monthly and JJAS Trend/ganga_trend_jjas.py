import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D


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
    "ganga_JJAS_sum_rainfall_anomaly_1971_2020.png"
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
    .str.contains(
        "ganga",
        na=False
    )
].copy()


# ============================================================
# 6. REMOVE UNWANTED SUB-BASINS
# ============================================================

ganga_data = ganga_data[
    ~ganga_data["Sub Basin Name"]
    .astype(str)
    .str.lower()
    .str.contains(
        r"nepal|ramganga|damanganga",
        na=False,
        regex=True
    )
].copy()


# ============================================================
# 7. GET FINAL GANGA SUB-BASIN LIST
# ============================================================

ganga_subbasins = (
    ganga_data[
        [
            "Sub Basin ID",
            "Sub Basin Name"
        ]
    ]
    .drop_duplicates()
    .sort_values("Sub Basin ID")
    .reset_index(drop=True)
)


# ============================================================
# 8. PRINT FINAL SUB-BASINS
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
# 9. JJAS MONTHLY ANOMALY COLUMNS
# ============================================================

jjas_columns = [
    "June_Anomaly",
    "July_Anomaly",
    "August_Anomaly",
    "September_Anomaly"
]


# ============================================================
# 10. CONVERT JJAS ANOMALIES TO NUMERIC
# ============================================================

for column in jjas_columns:

    ganga_data[column] = pd.to_numeric(
        ganga_data[column],
        errors="coerce"
    )


# ============================================================
# 11. CALCULATE JJAS SUM FOR EACH SUB-BASIN AND YEAR
# ============================================================

ganga_data["JJAS_Anomaly"] = (
    ganga_data["June_Anomaly"]
    + ganga_data["July_Anomaly"]
    + ganga_data["August_Anomaly"]
    + ganga_data["September_Anomaly"]
)


# ============================================================
# 12. SUM JJAS ANOMALIES OF ALL 17 SUB-BASINS
# ============================================================

ganga_JJAS_sum = (
    ganga_data
    .groupby(
        "Year",
        as_index=False
    )["JJAS_Anomaly"]
    .sum()
)


# ============================================================
# 13. SORT BY YEAR
# ============================================================

ganga_JJAS_sum = (
    ganga_JJAS_sum
    .sort_values("Year")
    .reset_index(drop=True)
)


# ============================================================
# 14. CALCULATE LINEAR TREND
# ============================================================

trend_x = (
    ganga_JJAS_sum["Year"]
    .to_numpy()
)

trend_y = (
    ganga_JJAS_sum["JJAS_Anomaly"]
    .to_numpy()
)


trend_coefficients = np.polyfit(
    trend_x,
    trend_y,
    1
)


trend_function = np.poly1d(
    trend_coefficients
)


ganga_JJAS_sum["Trend"] = (
    trend_function(trend_x)
)


# ============================================================
# 15. PRINT TREND INFORMATION
# ============================================================

slope = trend_coefficients[0]
intercept = trend_coefficients[1]

print()
print("==========================================")
print("JJAS SUM AND TREND")
print("==========================================")

print(
    "Number of Ganga sub-basins:",
    ganga_data["Sub Basin ID"].nunique()
)

print(
    "Number of years:",
    len(ganga_JJAS_sum)
)

print(
    "Year range:",
    ganga_JJAS_sum["Year"].min(),
    "to",
    ganga_JJAS_sum["Year"].max()
)

print(
    "Trend slope:",
    slope,
    "mm/year"
)

print(
    "Trend intercept:",
    intercept
)


# ============================================================
# 16. CHECK JJAS VALUES
# ============================================================

print()
print("==========================================")
print("JJAS SUM CHECK")
print("==========================================")

print(
    ganga_JJAS_sum.to_string(index=False)
)


# ============================================================
# 17. CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 12)
)


# ============================================================
# 18. PLOT JJAS SUM
# ============================================================

ax.plot(
    ganga_JJAS_sum["Year"],
    ganga_JJAS_sum["JJAS_Anomaly"],
    linewidth=2,
    marker="o",
    markersize=5,
    label="JJAS Sum Anomaly of 17 Ganga Sub-basins"
)


# ============================================================
# 19. PLOT SINGLE LINEAR TREND
# ============================================================

ax.plot(
    ganga_JJAS_sum["Year"],
    ganga_JJAS_sum["Trend"],
    linewidth=2.5,
    linestyle="--",
    color="red",
    label="Linear Trend"
)


# ============================================================
# 20. ZERO ANOMALY REFERENCE LINE
# ============================================================

ax.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--"
)


# ============================================================
# 21. LEGEND FOR DATA AND TREND
# ============================================================

ax.legend(
    loc="upper right",
    fontsize=18,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white"
)


# ============================================================
# 22. X-AXIS: 1971–2020
# ============================================================

ax.set_xlim(
    1971,
    2020
)

ax.set_xticks(
    np.arange(
        1971,
        2021,
        1
    )
)


# ============================================================
# 23. AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=18
)

ax.set_ylabel(
    "JJAS Rainfall Anomaly Sum (mm)",
    fontsize=18
)

ax.set_title(
    "JJAS (June–September) Sum Rainfall Anomaly of Ganga Sub-basins (1971–2020)",
    fontsize=22,
    fontweight="bold"
)


# ============================================================
# 24. X-AXIS TICK LABELS
# ============================================================

plt.xticks(
    rotation=90,
    fontsize=11
)

ax.tick_params(
    axis="y",
    labelsize=14
)


# ============================================================
# 25. GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 26. 17 GANGA SUB-BASIN NAMES AND IDs
# ============================================================

subbasins_for_legend = [
    ("Upper Ganga", "10092"),
    ("Ganga Narora to Phapham", "18153"),
    ("Ganga Phaphamau to Ball", "18154"),
    ("Gomti", "18156"),
    ("Sai", "18162"),
    ("Bagmati Adhwara", "22180"),
    ("Chandan", "22183"),
    ("Gandak", "22184"),
    ("Kiul", "22186"),
    ("Kosi", "22187"),
    ("Mahananda", "22189"),
    ("North Koel", "22191"),
    ("Punpun/Dhab Nadi", "22192"),
    ("Sone", "22193"),
    ("Kanhar", "23195"),
    ("Chhatnag to Mirzapur", "6048"),
    ("Upper Sone", "6056")
]


# ============================================================
# 27. CREATE NUMBERED LEGEND HANDLES
# ============================================================

legend_handles = []

for number, (
    name,
    subbasin_id
) in enumerate(
    subbasins_for_legend,
    start=1
):

    handle = Line2D(
        [0],
        [0],
        linestyle="None",
        label=f"{number}  {name} ({subbasin_id})"
    )

    legend_handles.append(
        handle
    )


# ============================================================
# 28. CREATE LEGEND BOX BELOW THE MAP
# ============================================================

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=3,
    fontsize=20,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white",
    handlelength=0,
    handletextpad=0,
    columnspacing=2.5,
    borderpad=1.5
)


# ============================================================
# 29. ADJUST LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.30,
        1,
        1
    ]
)


# ============================================================
# 30. SAVE FIGURE
# ============================================================

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 31. SHOW FIGURE
# ============================================================

plt.show()


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("PLOT COMPLETED")
print("==========================================")

print(
    "Final number of Ganga sub-basins:",
    len(ganga_subbasins)
)

print(
    "Total JJAS yearly values:",
    len(ganga_JJAS_sum)
)

print(
    "Output plot:"
)

print(
    output_plot
)