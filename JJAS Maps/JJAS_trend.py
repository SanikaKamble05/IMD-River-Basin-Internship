import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.lines import Line2D

# FILE PATHS
monthly_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_monthly_rainfall_sum.csv"
)

long_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_JJAS_rainfall_long_1971_2020.csv"
)

trend_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_JJAS_rainfall_trend_1971_2020.csv"
)

shapefile = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220.shp"
)

india_path = (
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary\India outline boundary.shp"
)

output_map = (
    r"E:\IMD Project\Final Maps"
    r"\RF_JJAS_Trend_1971_2020.png"
)

# JJAS MONTHS
months = [
    "June",
    "July",
    "August",
    "September"
]

# READ DATA
df = pd.read_csv(monthly_csv)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

for month in months:
    df[month] = pd.to_numeric(
        df[month],
        errors="coerce"
    )

df = df[
    (df["Year"] >= 1971) &
    (df["Year"] <= 2020)
].copy()

subbasins = (
    df[
        [
            "River Basin",
            "Sub Basin ID",
            "Sub Basin Name"
        ]
    ]
    .drop_duplicates(
        subset=["Sub Basin ID"],
        keep="first"
    )
    .reset_index(drop=True)
)

# CONVERT TO LONG FORMAT
long_df = df.melt(
    id_vars=[
        "River Basin",
        "Sub Basin ID",
        "Sub Basin Name",
        "Year"
    ],
    value_vars=months,
    var_name="Month",
    value_name="Rainfall"
)

month_order = {
    month: i
    for i, month in enumerate(months)
}

subbasin_order = {
    sub_id: i
    for i, sub_id in enumerate(
        subbasins["Sub Basin ID"]
    )
}

long_df["_SubBasinOrder"] = (
    long_df["Sub Basin ID"]
    .map(subbasin_order)
)

long_df["_MonthOrder"] = (
    long_df["Month"]
    .map(month_order)
)

long_df = (
    long_df
    .sort_values(
        [
            "_SubBasinOrder",
            "Year",
            "_MonthOrder"
        ]
    )
    .drop(
        columns=[
            "_SubBasinOrder",
            "_MonthOrder"
        ]
    )
)

long_df["Date"] = pd.to_datetime(
    long_df["Year"].astype(int).astype(str)
    + "-"
    + long_df["Month"]
    + "-01"
)

long_df.to_csv(
    long_csv,
    index=False
)

# CALCULATE JJAS TREND
results = []

for _, sub in subbasins.iterrows():

    sub_id = sub["Sub Basin ID"]

    sub_data = long_df[
        long_df["Sub Basin ID"] == sub_id
    ].copy()

    sub_data = sub_data.sort_values(
        "Date"
    )

    sub_data = sub_data.dropna(
        subset=["Rainfall"]
    )

    if len(sub_data) < 2:

        slope = np.nan
        intercept = np.nan
        p_value = np.nan
        category = "No Data"

    else:

        x = np.arange(
            len(sub_data),
            dtype=float
        )

        y = sub_data["Rainfall"].to_numpy(
            dtype=float
        )

        slope, intercept = np.polyfit(
            x,
            y,
            1
        )

        regression = linregress(
            x,
            y
        )

        p_value = regression.pvalue

        if slope > 0:

            if p_value < 0.05:
                category = "Significantly Increasing"
            else:
                category = "Increasing"

        elif slope < 0:

            if p_value < 0.05:
                category = "Significantly Decreasing"
            else:
                category = "Decreasing"

        else:

            category = "No Data"

    results.append({
        "River Basin": sub["River Basin"],
        "Sub Basin ID": sub["Sub Basin ID"],
        "Sub Basin Name": sub["Sub Basin Name"],
        "Start Month": "June 1971",
        "End Month": "September 2020",
        "Number of Months": len(sub_data),
        "Trend Slope": slope,
        "Intercept": intercept,
        "P-Value": p_value,
        "Trend Category": category
    })

trend_df = pd.DataFrame(
    results
)

trend_df.to_csv(
    trend_csv,
    index=False
)

# VERIFICATION
print()
print("==========================================")
print("JJAS TREND ANALYSIS COMPLETED")
print("==========================================")

print(
    "Original monthly CSV rows:",
    len(df)
)

print(
    "Long-format observations:",
    len(long_df)
)

print(
    "Expected JJAS observations:",
    len(df) * 4
)

print(
    "Unique sub-basins:",
    len(subbasins)
)

print(
    "Trend CSV rows:",
    len(trend_df)
)

print(
    "First sub-basin:",
    subbasins.iloc[0]["Sub Basin Name"]
)

print(
    "First Sub Basin ID:",
    subbasins.iloc[0]["Sub Basin ID"]
)

print()
print("Trend categories:")

print(
    trend_df["Trend Category"]
    .value_counts()
)

print()
print(
    "Long CSV saved:",
    long_csv
)

print(
    "Trend CSV saved:",
    trend_csv
)

print("==========================================")

# READ SHAPEFILES
basins = gpd.read_file(
    shapefile
)

india = gpd.read_file(
    india_path
)

# CREATE MAP IDs
trend_df["Map_ID"] = (
    trend_df["Sub Basin ID"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)

basins["Map_ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)

# CLIP TO INDIA
india = india.to_crs(
    basins.crs
)

basins = gpd.clip(
    basins,
    india
)

basins = basins[
    basins.geometry.geom_type.isin(
        [
            "Polygon",
            "MultiPolygon"
        ]
    )
].copy()

basins = basins.to_crs(
    epsg=4326
)

india = india.to_crs(
    epsg=4326
)

# MERGE TREND DATA
map_df = trend_df[
    [
        "Map_ID",
        "Trend Category"
    ]
].copy()

basins = basins.merge(
    map_df,
    on="Map_ID",
    how="left"
)

basins["Trend Category"] = (
    basins["Trend Category"]
    .fillna("No Data")
)

# TREND COLORS
category_colors = {
    "No Data": "white",
    "Significantly Increasing": "blue",
    "Increasing": "lightblue",
    "Decreasing": "lightcoral",
    "Significantly Decreasing": "red"
}

# CREATE MAP
fig, ax = plt.subplots(
    figsize=(12, 14)
)

for category, color in category_colors.items():

    selected = basins[
        basins["Trend Category"] == category
    ]

    if not selected.empty:

        selected.plot(
            ax=ax,
            color=color,
            edgecolor="black",
            linewidth=0.4
        )

india.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.5
)

# TITLE
ax.set_title(
    "INDIA METEOROLOGICAL DEPARTMENT\n"
    "JJAS MONTHLY RAINFALL TREND\n"
    "FOR THE PERIOD 1971 - 2020",
    fontsize=20,
    fontweight="bold",
    pad=15
)

# GRID
ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.5
)

ax.set_xlabel(
    "Longitude",
    fontsize=13
)

ax.set_ylabel(
    "Latitude",
    fontsize=13
)

# NORTH ARROW
ax.annotate(
    "N",
    xy=(0.94, 0.88),
    xytext=(0.94, 0.76),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=2,
        color="black"
    )
)

# LEGEND
legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=12,
        label="No Data"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="blue",
        markersize=12,
        label="Significantly Increasing"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="lightblue",
        markersize=12,
        label="Increasing"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="lightcoral",
        markersize=12,
        label="Decreasing"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="red",
        markersize=12,
        label="Significantly Decreasing"
    )
]

ax.legend(
    handles=legend_elements,
    title="JJAS TRENDS",
    title_fontsize=15,
    fontsize=12,
    loc="center",
    bbox_to_anchor=(0.62, 0.25),
    frameon=True
)

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)

plt.show()