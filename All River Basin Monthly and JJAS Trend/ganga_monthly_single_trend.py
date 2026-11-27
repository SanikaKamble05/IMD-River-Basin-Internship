import pandas as pd
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
    "ganga_mean_monthly_rainfall_anomaly_1971_2020.png"
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
# 9. MONTHLY ANOMALY COLUMNS
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
# 10. CREATE MONTHLY TIME SERIES
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
# 11. SORT DATA
# ============================================================

ganga_monthly = (
    ganga_monthly
    .sort_values(
        [
            "Date",
            "Sub Basin ID"
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. CALCULATE MEAN ANOMALY OF ALL GANGA SUB-BASINS
# ============================================================

ganga_mean_monthly = (
    ganga_monthly
    .groupby(
        "Date",
        as_index=False
    )["Anomaly"]
    .mean()
)

ganga_mean_monthly = (
    ganga_mean_monthly
    .rename(
        columns={
            "Anomaly":
                "Mean_Rainfall_Anomaly"
        }
    )
)


# ============================================================
# 13. CHECK DATA
# ============================================================

print()
print("==========================================")
print("MEAN MONTHLY ANOMALY CHECK")
print("==========================================")

print(
    "Number of Ganga sub-basins:",
    ganga_data["Sub Basin ID"].nunique()
)

print(
    "Number of monthly observations:",
    len(ganga_mean_monthly)
)

print(
    "Expected monthly observations:",
    50 * 12
)

print(
    "Date range:",
    ganga_mean_monthly["Date"].min(),
    "to",
    ganga_mean_monthly["Date"].max()
)


# ============================================================
# 14. CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 12)
)


# ============================================================
# 15. PLOT SINGLE MEAN TREND LINE
# ============================================================

ax.plot(
    ganga_mean_monthly["Date"],
    ganga_mean_monthly["Mean_Rainfall_Anomaly"],
    linewidth=1.5,
    label="Mean of 17 Ganga Sub-basins"
)


# ============================================================
# 16. ZERO ANOMALY REFERENCE LINE
# ============================================================

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


# ============================================================
# 17. X-AXIS: 1971–2020
# ============================================================

ax.set_xlim(
    pd.Timestamp("1971-01-01"),
    pd.Timestamp("2020-12-31")
)

ax.xaxis.set_major_locator(
    mdates.YearLocator(1)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)


# ============================================================
# 18. AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=18
)

ax.set_ylabel(
    "Mean Monthly Rainfall Anomaly (mm)",
    fontsize=18
)

ax.set_title(
    "Mean Monthly Rainfall Anomaly of Ganga Sub-basins (1971–2020)",
    fontsize=22,
    fontweight="bold"
)


# ============================================================
# 19. X-AXIS TICK LABELS
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
# 20. GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 21. 17 GANGA SUB-BASIN NAMES AND IDs
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
# 22. CREATE COLORED LEGEND HANDLES
# ============================================================

colors = plt.cm.tab20(
    range(len(subbasins_for_legend))
)

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
# 23. CREATE LEGEND BOX BELOW THE MAP
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
# 24. ADJUST LAYOUT
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
# 25. SAVE FIGURE
# ============================================================

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 26. SHOW FIGURE
# ============================================================

plt.show()


# ============================================================
# 27. FINAL OUTPUT
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
    "Total monthly mean values:",
    len(ganga_mean_monthly)
)

print(
    "Output plot:"
)

print(
    output_plot
)