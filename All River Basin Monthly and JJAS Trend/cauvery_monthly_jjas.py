import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
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
    "cauvery_mean_monthly_and_JJAS_rainfall_anomaly_1971_2020.png"
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
    cv_data["subbasin_id"].dropna()
)

cauvery_data = anomaly_data[
    anomaly_data["Sub Basin ID"].isin(cv_ids)
].copy()


# ============================================================
# 5. KEEP ONLY CAUVERY BASIN
# ============================================================

cauvery_data = cauvery_data[
    cauvery_data["River Basin"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.contains(
        "cauvery",
        na=False
    )
].copy()


# ============================================================
# 6. FINAL CAUVERY SUB-BASIN LIST
# ============================================================

cauvery_subbasins = (
    cauvery_data[
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
# 7. PRINT FINAL SUB-BASINS
# ============================================================

print()
print("==========================================")
print("FINAL CAUVERY SUB-BASINS")
print("==========================================")

print(
    cauvery_subbasins.to_string(
        index=False
    )
)

print()

print(
    "Number of Cauvery sub-basins:",
    len(cauvery_subbasins)
)


# ============================================================
# 8. MONTHLY ANOMALY COLUMNS
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
# 9. CREATE MONTHLY TIME SERIES
# ============================================================

records = []

for _, row in cauvery_data.iterrows():

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

            "Month":
                month_name,

            "Anomaly":
                pd.to_numeric(
                    row[anomaly_column],
                    errors="coerce"
                )
        })


cauvery_monthly = pd.DataFrame(
    records
)


# ============================================================
# 10. SORT MONTHLY DATA
# ============================================================

cauvery_monthly = (
    cauvery_monthly
    .sort_values(
        [
            "Date",
            "Sub Basin ID"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 11. CALCULATE MEAN MONTHLY ANOMALY
# ============================================================

cauvery_mean_monthly = (
    cauvery_monthly
    .groupby(
        "Date",
        as_index=False
    )["Anomaly"]
    .mean()
)

cauvery_mean_monthly = (
    cauvery_mean_monthly
    .rename(
        columns={
            "Anomaly":
                "Mean_Rainfall_Anomaly"
        }
    )
)


# ============================================================
# 12. CALCULATE MONTHLY LINEAR TREND
# ============================================================

monthly_trend_x = mdates.date2num(
    cauvery_mean_monthly["Date"]
)

monthly_trend_y = (
    cauvery_mean_monthly[
        "Mean_Rainfall_Anomaly"
    ]
)

monthly_trend_coefficients = np.polyfit(
    monthly_trend_x,
    monthly_trend_y,
    1
)

monthly_trend_function = np.poly1d(
    monthly_trend_coefficients
)

cauvery_mean_monthly["Trend"] = (
    monthly_trend_function(
        monthly_trend_x
    )
)


# ============================================================
# 13. CONVERT JJAS MONTHLY VALUES TO NUMERIC
# ============================================================

jjas_columns = [
    "June_Anomaly",
    "July_Anomaly",
    "August_Anomaly",
    "September_Anomaly"
]

for column in jjas_columns:

    cauvery_data[column] = pd.to_numeric(
        cauvery_data[column],
        errors="coerce"
    )


# ============================================================
# 14. CALCULATE JJAS MEAN FOR EACH SUB-BASIN AND YEAR
# ============================================================

cauvery_data["JJAS_Anomaly"] = (
    cauvery_data[
        jjas_columns
    ].mean(
        axis=1
    )
)


# ============================================================
# 15. CALCULATE CAUVERY JJAS MEAN FOR EACH YEAR
# ============================================================

cauvery_JJAS = (
    cauvery_data
    .groupby(
        "Year",
        as_index=False
    )["JJAS_Anomaly"]
    .mean()
)


# ============================================================
# 16. CALCULATE JJAS LINEAR TREND
# ============================================================

jjas_trend_x = (
    cauvery_JJAS["Year"]
)

jjas_trend_y = (
    cauvery_JJAS[
        "JJAS_Anomaly"
    ]
)

jjas_trend_coefficients = np.polyfit(
    jjas_trend_x,
    jjas_trend_y,
    1
)

jjas_trend_function = np.poly1d(
    jjas_trend_coefficients
)

cauvery_JJAS["Trend"] = (
    jjas_trend_function(
        jjas_trend_x
    )
)


# ============================================================
# 17. CHECK DATA
# ============================================================

print()
print("==========================================")
print("DATA CHECK")
print("==========================================")

print(
    "Number of Cauvery sub-basins:",
    cauvery_data[
        "Sub Basin ID"
    ].nunique()
)

print(
    "Number of monthly observations:",
    len(cauvery_mean_monthly)
)

print(
    "Expected monthly observations:",
    50 * 12
)

print(
    "Number of yearly JJAS observations:",
    len(cauvery_JJAS)
)

print(
    "Expected yearly JJAS observations:",
    50
)

print(
    "Monthly date range:",
    cauvery_mean_monthly[
        "Date"
    ].min(),
    "to",
    cauvery_mean_monthly[
        "Date"
    ].max()
)

print(
    "JJAS year range:",
    cauvery_JJAS[
        "Year"
    ].min(),
    "to",
    cauvery_JJAS[
        "Year"
    ].max()
)

print(
    "=========================================="
)


# ============================================================
# 18. CREATE FIGURE WITH TWO PLOTS
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(24, 18)
)

ax1 = axes[0]
ax2 = axes[1]


# ============================================================
# 19. TOP PLOT
#     MEAN MONTHLY RAINFALL ANOMALY
# ============================================================

ax1.plot(
    cauvery_mean_monthly["Date"],
    cauvery_mean_monthly[
        "Mean_Rainfall_Anomaly"
    ],
    linewidth=1.5,
    label="Cauvery Basin"
)

ax1.plot(
    cauvery_mean_monthly["Date"],
    cauvery_mean_monthly[
        "Trend"
    ],
    linewidth=2.5,
    linestyle="--",
    color="red",
    label="Linear Trend"
)

ax1.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--"
)


# ============================================================
# 20. TOP PLOT LEGEND
# ============================================================

ax1.legend(
    loc="upper right",
    fontsize=16,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white"
)


# ============================================================
# 21. TOP PLOT X-AXIS
# ============================================================

ax1.set_xlim(
    pd.Timestamp("1971-01-01"),
    pd.Timestamp("2020-12-31")
)

ax1.xaxis.set_major_locator(
    mdates.YearLocator(2)
)

ax1.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

ax1.xaxis.set_minor_locator(
    mdates.YearLocator(1)
)


# ============================================================
# 22. TOP PLOT LABELS
# ============================================================

ax1.set_xlabel(
    "Year",
    fontsize=18,
    labelpad=20
)

ax1.set_ylabel(
    "Rainfall Anomaly (mm)",
    fontsize=18
)

ax1.set_title(
    "Rainfall Anomaly of Cauvery Basin (1971–2020)",
    fontsize=22,
    fontweight="bold"
)

ax1.tick_params(
    axis="x",
    which="major",
    labelsize=20,
    rotation=90,
    colors="black",
    length=7,
    width=1.2
)

ax1.tick_params(
    axis="x",
    which="minor",
    colors="black",
    length=5,
    width=1
)

ax1.tick_params(
    axis="y",
    labelsize=20,
    colors="black"
)

ax1.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 23. BOTTOM PLOT
#     JJAS MEAN RAINFALL ANOMALY
# ============================================================

ax2.plot(
    cauvery_JJAS["Year"],
    cauvery_JJAS[
        "JJAS_Anomaly"
    ],
    linewidth=1.8,
    marker="o",
    markersize=4,
    label="JJAS Mean of Cauvery Basin"
)

ax2.plot(
    cauvery_JJAS["Year"],
    cauvery_JJAS[
        "Trend"
    ],
    linewidth=2.5,
    linestyle="--",
    color="red",
    label="Linear Trend"
)

ax2.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--"
)


# ============================================================
# 24. BOTTOM PLOT LEGEND
# ============================================================

ax2.legend(
    loc="upper right",
    fontsize=16,
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="lightgray",
    facecolor="white"
)


# ============================================================
# 25. BOTTOM PLOT X-AXIS
# ============================================================

ax2.set_xlim(
    1971,
    2020
)

ax2.set_xticks(
    range(
        1972,
        2021,
        2
    )
)

ax2.set_xticks(
    range(
        1971,
        2021,
        1
    ),
    minor=True
)


# ============================================================
# 26. BOTTOM PLOT LABELS
# ============================================================

ax2.set_xlabel(
    "Year",
    fontsize=18,
    labelpad=10
)

ax2.set_ylabel(
    "JJAS Mean Rainfall Anomaly (mm)",
    fontsize=18
)

ax2.set_title(
    "JJAS (June–September) Rainfall Anomaly of Cauvery Basin (1971–2020)",
    fontsize=22,
    fontweight="bold"
)

ax2.tick_params(
    axis="x",
    which="major",
    labelsize=20,
    rotation=90,
    colors="black",
    length=7,
    width=1.2
)

ax2.tick_params(
    axis="x",
    which="minor",
    colors="black",
    length=5,
    width=1
)

ax2.tick_params(
    axis="y",
    labelsize=20,
    colors="black"
)

ax2.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 27. CAUVERY SUB-BASIN LIST
# ============================================================

subbasins_for_legend = (
    cauvery_subbasins[
        [
            "Sub Basin Name",
            "Sub Basin ID"
        ]
    ]
    .values
    .tolist()
)


# ============================================================
# 28. CREATE NUMBERED LEGEND HANDLES
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
        label=(
            f"{number}  "
            f"{name} "
            f"({subbasin_id})"
        )
    )

    legend_handles.append(
        handle
    )


# ============================================================
# 29. SUB-BASIN LIST BELOW BOTH PLOTS
# ============================================================

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.002),
    ncol=3,
    fontsize=22,
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
# 30. MAIN FIGURE TITLE
# ============================================================

fig.suptitle(
    "Cauvery Basin Rainfall Anomaly Analysis (1971–2020)",
    fontsize=26,
    fontweight="bold",
    y=0.985
)


# ============================================================
# 31. ADJUST LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.94,
    bottom=0.30,
    hspace=0.40
)


# ============================================================
# 32. SAVE FIGURE
# ============================================================

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 33. SHOW FIGURE
# ============================================================

plt.show()


# ============================================================
# 34. FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("PLOT COMPLETED")
print("==========================================")

print(
    "Final number of Cauvery sub-basins:",
    len(cauvery_subbasins)
)

print(
    "Total monthly mean values:",
    len(cauvery_mean_monthly)
)

print(
    "Total yearly JJAS mean values:",
    len(cauvery_JJAS)
)

print(
    "Output plot:"
)

print(
    output_plot
)

print(
    "=========================================="
)