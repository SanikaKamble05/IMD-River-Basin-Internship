import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

base_path = Path(r"E:\IMD Project\Final Maps")

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
    "ganga_JJAS_monthly_anomaly_sum_1971_2020.png"
)


# ============================================================
# READ DATA
# ============================================================

anomaly_data = pd.read_csv(anomaly_file)
cv_data = pd.read_csv(cv_file)


# ============================================================
# CLEAN SUB-BASIN IDs
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
# KEEP ONLY SUB-BASINS PRESENT IN CV DATA
# ============================================================

cv_ids = set(
    cv_data["subbasin_id"].dropna()
)

ganga_data = anomaly_data[
    anomaly_data["Sub Basin ID"].isin(cv_ids)
].copy()


# ============================================================
# SELECT GANGA BASIN
# ============================================================

ganga_data = ganga_data[
    ganga_data["River Basin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains("ganga", na=False)
].copy()


# ============================================================
# REMOVE UNWANTED SUB-BASINS
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
# CHECK GANGA SUB-BASINS
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


print()
print("FINAL GANGA SUB-BASINS")
print("======================")
print(ganga_subbasins.to_string(index=False))
print()

print(
    "Number of Ganga sub-basins:",
    len(ganga_subbasins)
)


# ============================================================
# JJAS ANOMALY COLUMNS
# ============================================================

jjas_columns = [
    "June_Anomaly",
    "July_Anomaly",
    "August_Anomaly",
    "September_Anomaly"
]


# ============================================================
# CONVERT ANOMALY VALUES TO NUMERIC
# ============================================================

for column in jjas_columns:

    ganga_data[column] = pd.to_numeric(
        ganga_data[column],
        errors="coerce"
    )


# ============================================================
# SUM EACH MONTH ACROSS ALL 17 GANGA SUB-BASINS
# ============================================================
#
# Example:
#
# 1971 June =
# June anomaly of all 17 sub-basins added together
#
# 1971 July =
# July anomaly of all 17 sub-basins added together
#
# and so on.
#
# Therefore:
#
# 1971 -> 4 values
# 1972 -> 4 values
# ...
# 2020 -> 4 values
#
# Total = 50 × 4 = 200 values
#
# ============================================================

ganga_JJAS = (
    ganga_data
    .groupby(
        "Year",
        as_index=False
    )[jjas_columns]
    .sum()
)


# ============================================================
# CONVERT FROM WIDE FORMAT TO LONG FORMAT
# ============================================================

ganga_JJAS_long = ganga_JJAS.melt(
    id_vars="Year",
    value_vars=jjas_columns,
    var_name="Month",
    value_name="Rainfall_Anomaly"
)


# ============================================================
# REMOVE "_Anomaly" FROM MONTH NAMES
# ============================================================

ganga_JJAS_long["Month"] = (
    ganga_JJAS_long["Month"]
    .str.replace(
        "_Anomaly",
        "",
        regex=False
    )
)


# ============================================================
# MONTH ORDER
# ============================================================

month_order = [
    "June",
    "July",
    "August",
    "September"
]

ganga_JJAS_long["Month"] = pd.Categorical(
    ganga_JJAS_long["Month"],
    categories=month_order,
    ordered=True
)


# ============================================================
# SORT YEAR AND MONTH
# ============================================================

ganga_JJAS_long = (
    ganga_JJAS_long
    .sort_values(
        ["Year", "Month"]
    )
    .reset_index(drop=True)
)


# ============================================================
# CREATE DATE FOR EACH MONTH
# ============================================================

month_numbers = {
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9
}

ganga_JJAS_long["Month_Number"] = (
    ganga_JJAS_long["Month"]
    .astype(str)
    .map(month_numbers)
)


ganga_JJAS_long["Date"] = pd.to_datetime(
    dict(
        year=ganga_JJAS_long["Year"],
        month=ganga_JJAS_long["Month_Number"],
        day=1
    )
)


# ============================================================
# CHECK DATA
# ============================================================

print()
print("JJAS DATA CHECK")
print("===============")

print(
    "Number of Ganga sub-basins:",
    ganga_data["Sub Basin ID"].nunique()
)

print(
    "Number of years:",
    ganga_JJAS_long["Year"].nunique()
)

print(
    "Number of months per year:",
    ganga_JJAS_long.groupby("Year").size().unique()
)

print(
    "Total observations:",
    len(ganga_JJAS_long)
)

print(
    "Expected observations:",
    200
)

print(
    "Year range:",
    ganga_JJAS_long["Year"].min(),
    "to",
    ganga_JJAS_long["Year"].max()
)

print()

print(
    ganga_JJAS_long[
        [
            "Year",
            "Month",
            "Rainfall_Anomaly"
        ]
    ].head(12)
)


# ============================================================
# LINEAR TREND
# ============================================================
#
# Trend is fitted to all 200 monthly observations.
#
# x = observation order
# y = monthly JJAS anomaly sum
#
# ============================================================

trend_x = np.arange(
    len(ganga_JJAS_long)
)

trend_y = (
    ganga_JJAS_long["Rainfall_Anomaly"]
    .values
)


trend_coefficients = np.polyfit(
    trend_x,
    trend_y,
    1
)


trend_function = np.poly1d(
    trend_coefficients
)


ganga_JJAS_long["Trend"] = (
    trend_function(trend_x)
)


# ============================================================
# PRINT TREND
# ============================================================

print()
print("LINEAR TREND")
print("=============")

print(
    "Trend slope:",
    trend_coefficients[0],
    "mm per observation"
)

print(
    "Trend intercept:",
    trend_coefficients[1]
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 12)
)


# ============================================================
# MAIN SINGLE LINE
# ============================================================

ax.plot(
    ganga_JJAS_long["Date"],
    ganga_JJAS_long["Rainfall_Anomaly"],
    linewidth=1.8,
    marker="o",
    markersize=3.5,
    label="JJAS Monthly Anomaly Sum"
)


# ============================================================
# TREND LINE
# ============================================================

ax.plot(
    ganga_JJAS_long["Date"],
    ganga_JJAS_long["Trend"],
    linewidth=2.5,
    linestyle="--",
    color="red",
    label="Linear Trend"
)


# ============================================================
# ZERO ANOMALY LINE
# ============================================================

ax.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--",
    color="black"
)


# ============================================================
# AXIS LIMITS
# ============================================================

ax.set_xlim(
    pd.Timestamp("1971-06-01"),
    pd.Timestamp("2020-09-01")
)


# ============================================================
# X-AXIS
# ============================================================

ax.xaxis.set_major_locator(
    mdates.YearLocator(1)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)


# ============================================================
# LABELS
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
    "JJAS (June–September) Monthly Rainfall Anomaly Sum of Ganga Sub-basins (1971–2020)",
    fontsize=22,
    fontweight="bold"
)


# ============================================================
# TICK SETTINGS
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
# GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# MAIN LEGEND INSIDE PLOT
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
# 17 GANGA SUB-BASINS
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
# TEXT-ONLY LEGEND HANDLES
# ============================================================

legend_handles = []


for number, (name, subbasin_id) in enumerate(
    subbasins_for_legend,
    start=1
):

    handle = Line2D(
        [0],
        [0],
        linestyle="None",
        label=f"{number}  {name} ({subbasin_id})"
    )

    legend_handles.append(handle)


# ============================================================
# SUB-BASIN BOX
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
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.30, 1, 1]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# SHOW
# ============================================================

plt.show()


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("PLOT COMPLETED")
print("==============")

print(
    "Final number of Ganga sub-basins:",
    len(ganga_subbasins)
)

print(
    "Total observations:",
    len(ganga_JJAS_long)
)

print(
    "Expected observations:",
    200
)

print(
    "Output plot:",
    output_plot
)