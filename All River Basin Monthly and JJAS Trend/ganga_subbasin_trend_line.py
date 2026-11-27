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
    "ganga_subbasins_monthly_rainfall_anomaly_1971_2020.png"
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
# 11. SORT TIME SERIES
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
# 12. CHECK DATA
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
# 13. CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 12)
)


# ============================================================
# 14. PLOT ONE LINE FOR EACH SUB-BASIN
# ============================================================

for _, subbasin in ganga_subbasins.iterrows():

    subbasin_data = ganga_monthly[
        ganga_monthly["Sub Basin ID"]
        == subbasin["Sub Basin ID"]
    ].copy()

    subbasin_data = (
        subbasin_data
        .sort_values("Date")
    )

    ax.plot(
        subbasin_data["Date"],
        subbasin_data["Anomaly"],
        linewidth=1.3,
        label=(
            str(subbasin["Sub Basin Name"])
            + " ("
            + str(subbasin["Sub Basin ID"])
            + ")"
        )
    )


# ============================================================
# 15. ZERO ANOMALY REFERENCE LINE
# ============================================================

ax.axhline(
    y=0,
    linewidth=1.5,
    linestyle="--"
)


# ============================================================
# 16. X-AXIS: EVERY YEAR FROM 1971 TO 2020
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
# 17. AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=18
)

ax.set_ylabel(
    "Rainfall Anomaly (mm)",
    fontsize=18
)

ax.set_title(
    "Monthly Rainfall Anomaly of Ganga Basin Sub-basins (1971–2020)",
    fontsize=22,
    fontweight="bold"
)


# ============================================================
# 18. X-AXIS TICK LABELS
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
# 19. GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)


# ============================================================
# 20. LEGEND
# ============================================================

ax.legend(
    title="Ganga Sub-basins",
    fontsize=9,
    title_fontsize=11,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)


# ============================================================
# 21. SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(
    output_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. FINAL OUTPUT
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
    "Output plot:"
)

print(
    output_plot
)