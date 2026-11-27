import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable


# ============================================================
# FILE PATHS
# ============================================================

shapefile_path = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220.shp"
)

india_path = (
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary"
    r"\India outline boundary.shp"
)

# NEW CSV FILE
csv_path = (
    r"E:\IMD Project\Final Maps"
    r"\Subbasin_Normals_2026_198.csv"
)

output_shapefile = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220_seasonal_rainfall.shp"
)

output_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_seasonal_rainfall.csv"
)

output_map = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_seasonal_rainfall.png"
)


# ============================================================
# READ SHAPEFILE
# ============================================================

basins = gpd.read_file(
    shapefile_path
)


# ============================================================
# CREATE ID FROM SHAPEFILE UIN
# ============================================================

basins["ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# READ NEW CSV
# ============================================================

df = pd.read_csv(
    csv_path
)


# ============================================================
# CREATE ID FROM CSV UIN
# ============================================================

df["ID"] = (
    df["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)


# ============================================================
# SEASONAL COLUMNS ALREADY PRESENT IN CSV
# ============================================================

csv_seasons = [
    "JF",
    "MAM",
    "JJAS",
    "OND"
]


# ============================================================
# CONVERT SEASONAL VALUES TO NUMERIC
# ============================================================

for season in csv_seasons:

    df[season] = pd.to_numeric(
        df[season],
        errors="coerce"
    )


# ============================================================
# SELECT ONLY ID + SEASONAL COLUMNS
# ============================================================

seasonal_data = df[
    [
        "ID",
        "JF",
        "MAM",
        "JJAS",
        "OND"
    ]
].copy()


# ============================================================
# RENAME CSV SEASON COLUMNS
# ============================================================

seasonal_data = seasonal_data.rename(
    columns={
        "JF": "Winter",
        "MAM": "Pre-Monsoon",
        "JJAS": "Monsoon",
        "OND": "Post-Monsoon"
    }
)


# ============================================================
# MERGE CSV SEASONAL DATA WITH SHAPEFILE
# ============================================================

basins = basins.merge(
    seasonal_data,
    on="ID",
    how="left"
)


# ============================================================
# MATCHING INFORMATION
# ============================================================

matched = basins["Winter"].notna().sum()

unmatched = basins["Winter"].isna().sum()


print("==========================================")
print("SEASONAL RAINFALL MATCHING")
print("==========================================")

print(
    f"Total shapefile basins : {len(basins)}"
)

print(
    f"IDs matching CSV       : {matched}"
)

print(
    f"IDs not matching       : {unmatched}"
)

print("==========================================")


# ============================================================
# CREATE OUTPUT CSV
# ============================================================

seasonal_output = basins[
    [
        "ID",
        "BASIN",
        "SUBBASIN",
        "Winter",
        "Pre-Monsoon",
        "Monsoon",
        "Post-Monsoon"
    ]
].copy()


seasonal_output.to_csv(
    output_csv,
    index=False
)


# ============================================================
# KEEP ONLY POLYGON AND MULTIPOLYGON
# ============================================================

basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()


# ============================================================
# READ INDIA OUTLINE
# ============================================================

india = gpd.read_file(
    india_path
)


# ============================================================
# MATCH CRS
# ============================================================

india = india.to_crs(
    basins.crs
)


# ============================================================
# CLIP BASINS TO INDIA
# ============================================================

basins = gpd.clip(
    basins,
    india
)


# ============================================================
# REMOVE INVALID GEOMETRY TYPES
# ============================================================

basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()


# ============================================================
# SAVE UPDATED SHAPEFILE
# ============================================================

basins.to_file(
    output_shapefile,
    driver="ESRI Shapefile"
)


# ============================================================
# CONVERT TO WGS84 FOR PLOTTING
# ============================================================

basins = basins.to_crs(
    epsg=4326
)

india = india.to_crs(
    epsg=4326
)


# ============================================================
# SEASONS FOR MAP
# ============================================================

seasons = [
    "Winter",
    "Pre-Monsoon",
    "Monsoon",
    "Post-Monsoon"
]


# ============================================================
# SEASONAL RAINFALL STATISTICS
# ============================================================

print()
print("==========================================")
print("SEASONAL RAINFALL STATISTICS")
print("==========================================")


for season in seasons:

    print(
        f"{season:15s}: "
        f"{basins[season].min():.2f} - "
        f"{basins[season].max():.2f} mm"
    )


print("==========================================")


# ============================================================
# RAINFALL COLOR BOUNDARIES
# ============================================================

rainfall_boundaries = (
    list(range(0, 1001, 20))
    + list(range(1500, 3501, 500))
)


rainfall_boundaries = sorted(
    set(rainfall_boundaries)
)


# ============================================================
# NUMBER OF COLOR CLASSES
# ============================================================

num_classes = (
    len(rainfall_boundaries) - 1
)


# ============================================================
# BASE COLORMAP
# ============================================================

base_cmap = plt.get_cmap(
    "viridis_r"
)


# ============================================================
# CLASS MIDPOINTS
# ============================================================

class_midpoints = [
    (
        rainfall_boundaries[i]
        + rainfall_boundaries[i + 1]
    ) / 2
    for i in range(num_classes)
]


# ============================================================
# TRANSFORM COLORS
# ============================================================

normalized_values = [
    value / 3500
    for value in class_midpoints
]


gamma = 0.5


transformed_values = [
    value ** gamma
    for value in normalized_values
]


transformed_values = [
    min(max(value, 0), 1)
    for value in transformed_values
]


colors = [
    base_cmap(value)
    for value in transformed_values
]


# ============================================================
# CREATE DISCRETE COLORMAP
# ============================================================

cmap = ListedColormap(
    colors
)


# ============================================================
# CREATE BOUNDARY NORMALIZATION
# ============================================================

norm = BoundaryNorm(
    rainfall_boundaries,
    cmap.N,
    clip=False
)


# ============================================================
# CREATE 2 × 2 FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(26, 26)
)


# ============================================================
# FIGURE SPACING
# ============================================================

fig.subplots_adjust(
    left=0.03,
    right=0.80,
    top=0.93,
    bottom=0.09,
    wspace=0.08,
    hspace=0.12
)


# ============================================================
# PLOT FOUR SEASONS
# ============================================================

for ax, season in zip(
    axes.flat,
    seasons
):

    basins_with_data = basins[
        basins[season].notna()
    ].copy()


    if not basins_with_data.empty:

        basins_with_data.plot(
            ax=ax,
            column=season,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.45
        )


    # India boundary

    india.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5
    )


    # Season title

    ax.set_title(
        season,
        fontsize=28,
        fontweight="bold",
        pad=12
    )


    # Remove axis labels

    ax.set_xlabel("")
    ax.set_ylabel("")


    # ========================================================
    # LATITUDE AND LONGITUDE TICK APPEARANCE
    # ========================================================
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=30,
        length=12,
        width=2.5,
        direction="out",
        colors="black"
    )
    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(30)
        label.set_fontweight("normal")
        label.set_color("black")


# ============================================================
# CREATE COLORBAR
# ============================================================

sm = ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])


# ============================================================
# COLORBAR POSITION
# ============================================================

cax = fig.add_axes([
    0.82,
    0.10,
    0.035,
    0.80
])


# ============================================================
# COLORBAR
# ============================================================

cbar = fig.colorbar(
    sm,
    cax=cax,
    boundaries=rainfall_boundaries,
    spacing="uniform",
    extend="max",
    extendfrac="auto"
)


# ============================================================
# COLORBAR TICKS
# ============================================================

colorbar_ticks = [
    0,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    3500
]


colorbar_labels = [
    "0",
    "100",
    "200",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "900",
    "1000",
    "3500"
]


cbar.set_ticks(
    colorbar_ticks
)

cbar.set_ticklabels(
    colorbar_labels
)


# ============================================================
# COLORBAR TICK APPEARANCE
# ============================================================

cbar.ax.tick_params(
    labelsize=28,
    colors="black",
    length=9,
    width=1.8
)


for label in cbar.ax.get_yticklabels():

    label.set_fontsize(28)

    label.set_fontweight("bold")

    label.set_color("black")


# ============================================================
# COLORBAR LABEL
# ============================================================

cbar.set_label(
    "Rainfall (mm)",
    fontsize=24,
    fontweight="bold",
    color="black",
    labelpad=25
)


# ============================================================
# MAIN FIGURE TITLE
# ============================================================

fig.suptitle(
    "Seasonal Normal Rainfall of River Sub-Basins of India",
    fontsize=30,
    fontweight="bold",
    y=0.965
)


# ============================================================
# SAVE MAP
# ============================================================

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# DISPLAY MAP
# ============================================================

plt.show()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print()
print("==========================================")
print("PROCESS COMPLETED")
print("==========================================")

print(
    f"Seasonal CSV saved      : {output_csv}"
)

print(
    f"Updated shapefile saved : {output_shapefile}"
)

print(
    f"Seasonal map saved      : {output_map}"
)

print("==========================================")
