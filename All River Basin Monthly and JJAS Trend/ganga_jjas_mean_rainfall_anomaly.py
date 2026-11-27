import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


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
    "ganga_JJAS_mean_rainfall_anomaly_1971_2020.png"
)


anomaly_data = pd.read_csv(anomaly_file)
cv_data = pd.read_csv(cv_file)


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


cv_ids = set(
    cv_data["subbasin_id"].dropna()
)

ganga_data = anomaly_data[
    anomaly_data["Sub Basin ID"].isin(cv_ids)
].copy()


ganga_data = ganga_data[
    ganga_data["River Basin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains("ganga", na=False)
].copy()


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


jjas_columns = [
    "June_Anomaly",
    "July_Anomaly",
    "August_Anomaly",
    "September_Anomaly"
]


for column in jjas_columns:
    ganga_data[column] = pd.to_numeric(
        ganga_data[column],
        errors="coerce"
    )


ganga_data["JJAS_Anomaly"] = (
    ganga_data[jjas_columns].mean(axis=1)
)


ganga_JJAS = (
    ganga_data
    .groupby(
        "Year",
        as_index=False
    )["JJAS_Anomaly"]
    .mean()
)


trend_x = ganga_JJAS["Year"]
trend_y = ganga_JJAS["JJAS_Anomaly"]

trend_coefficients = np.polyfit(
    trend_x,
    trend_y,
    1
)

trend_function = np.poly1d(
    trend_coefficients
)

ganga_JJAS["Trend"] = (
    trend_function(trend_x)
)


print()
print("JJAS MEAN ANOMALY CHECK")
print("=======================")

print(
    "Number of Ganga sub-basins:",
    ganga_data["Sub Basin ID"].nunique()
)

print(
    "Number of yearly JJAS observations:",
    len(ganga_JJAS)
)

print(
    "Expected yearly observations:",
    50
)

print(
    "Year range:",
    ganga_JJAS["Year"].min(),
    "to",
    ganga_JJAS["Year"].max()
)


fig, ax = plt.subplots(
    figsize=(24, 12)
)


ax.plot(
    ganga_JJAS["Year"],
    ganga_JJAS["JJAS_Anomaly"],
    linewidth=1.5,
    marker="o",
    markersize=4,
    label="JJAS Mean of 17 Ganga Sub-basins"
)

ax.plot(
    ganga_JJAS["Year"],
    ganga_JJAS["Trend"],
    linewidth=2.5,
    linestyle="--",
    color="red",
    label="Linear Trend"
)


ax.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--"
)


ax.legend(
    loc="upper right",
    fontsize=18,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white"
)


ax.set_xlim(
    1971,
    2020
)

ax.set_xticks(
    range(1971, 2021)
)


ax.set_xlabel(
    "Year",
    fontsize=18
)

ax.set_ylabel(
    "JJAS Mean Rainfall Anomaly (mm)",
    fontsize=18
)

ax.set_title(
    "JJAS (June–September) Mean Rainfall Anomaly of Ganga Sub-basins (1971–2020)",
    fontsize=22,
    fontweight="bold"
)


plt.xticks(
    rotation=90,
    fontsize=11
)

ax.tick_params(
    axis="y",
    labelsize=14
)


ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)


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


plt.tight_layout(
    rect=[0, 0.30, 1, 1]
)


plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print()
print("PLOT COMPLETED")
print("==============")
print(
    "Final number of Ganga sub-basins:",
    len(ganga_subbasins)
)

print(
    "Total yearly JJAS mean values:",
    len(ganga_JJAS)
)

print(
    "Output plot:",
    output_plot
)