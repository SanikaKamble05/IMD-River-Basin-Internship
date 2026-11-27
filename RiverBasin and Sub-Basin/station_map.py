import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

shapefile_path = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220.shp"
)

india_path = (
    r"E:\IMD Project\Final Maps\India outline boundary"
    r"\India outline boundary"
    r"\India outline boundary.shp"
)

csv_path = (
    r"E:\IMD Project\Final Maps\basin_NORMALS.csv"
)

output_shapefile = (
    r"E:\IMD Project\Final Maps\SUB_BASIN_220"
    r"\SUB_BASIN_220_station_count.shp"
)

output_csv = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_station_count.csv"
)

output_map = (
    r"E:\IMD Project\Final Maps"
    r"\river_basin_station_count.png"
)

# Read sub-basin shapefile
basins = gpd.read_file(
    shapefile_path
)

# Read India boundary
india = gpd.read_file(
    india_path
)

# Match CRS
india = india.to_crs(
    basins.crs
)

# Clip sub-basins to India boundary
basins = gpd.clip(
    basins,
    india
)

# Create complete 9-digit ID
basins["ID"] = (
    basins["UIN"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str.zfill(9)
)

# Read rainfall station data
df = pd.read_csv(
    csv_path,
    header=1
)

# Extract complete 9-digit ID
df["ID"] = (
    df["sub_basin id"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .str[-9:]
)

# Convert station count to numeric
df["#stn"] = pd.to_numeric(
    df["#stn"],
    errors="coerce"
)

# Keep required station data
station_data = df[
    [
        "ID",
        "#stn"
    ]
].copy()

# Merge station data with clipped sub-basins
basins = basins.merge(
    station_data,
    on="ID",
    how="left"
)

matched = basins["#stn"].notna().sum()
unmatched = basins["#stn"].isna().sum()

print("==========================================")
print("STATION COUNT MATCHING")
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
print()
print("==========================================")
print("STATION COUNT STATISTICS")
print("==========================================")
print(
    f"Minimum stations : {basins['#stn'].min():.0f}"
)
print(
    f"Maximum stations : {basins['#stn'].max():.0f}"
)
print(
    f"Average stations : {basins['#stn'].mean():.2f}"
)
print("==========================================")

# Save station data CSV
station_output = basins[
    [
        "ID",
        "BASIN",
        "SUBBASIN",
        "#stn"
    ]
].copy()

station_output.to_csv(
    output_csv,
    index=False
)

# Keep only polygon geometries
basins = basins[
    basins.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
].copy()

# Save updated shapefile
basins.to_file(
    output_shapefile,
    driver="ESRI Shapefile"
)

# Convert to latitude/longitude
basins = basins.to_crs(
    epsg=4326
)

india = india.to_crs(
    epsg=4326
)

# Station-count boundaries
station_boundaries = [
    0,
    20,
    40,
    60,
    80,
    100,
    120,
    140
]

num_classes = (
    len(station_boundaries) - 1
)

# Blue color scheme
cmap = LinearSegmentedColormap.from_list(
    "stations",
    [
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08306b"
    ],
    N=num_classes
)

norm = BoundaryNorm(
    station_boundaries,
    cmap.N,
    clip=True
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

# Areas with station data
basins_with_data = basins[
    basins["#stn"].notna()
].copy()

# Areas without station data
basins_without_data = basins[
    basins["#stn"].isna()
].copy()

# Plot areas with station data
basins_with_data.plot(
    ax=ax,
    column="#stn",
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    linewidth=0.4
)

# Plot areas without station data in grey
basins_without_data.plot(
    ax=ax,
    color="dimgray",
    edgecolor="black",
    linewidth=0.4
)

# India boundary
india.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.2
)

# Title
ax.set_title(
    "Number of Rainfall Stations in River Sub-Basins of India",
    fontsize=16,
    fontweight="bold",
    pad=12
)

ax.set_xlabel("")
ax.set_ylabel("")

# Keep latitude and longitude numbers
ax.tick_params(
    axis="both",
    labelsize=10
)

# Colorbar
sm = ScalarMappable(
    cmap=cmap,
    norm=norm
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.035,
    pad=0.04,
    boundaries=station_boundaries,
    spacing="uniform"
)

# Boundary labels
cbar.set_ticks(
    station_boundaries
)

cbar.set_ticklabels(
    [
        "0",
        "20",
        "40",
        "60",
        "80",
        "100",
        "120",
        "140"
    ]
)

cbar.ax.tick_params(
    labelsize=10,
    colors="black"
)

for label in cbar.ax.get_yticklabels():

    label.set_fontweight(
        "bold"
    )

    label.set_color(
        "black"
    )

cbar.set_label(
    "Number of Stations",
    fontsize=12,
    fontweight="bold",
    color="black",
    labelpad=15
)

# Data unavailable note
if not basins_without_data.empty:

    legend_patch = Patch(
        facecolor="dimgray",
        edgecolor="black",
        label="Data not available"
    )

    ax.legend(
        handles=[legend_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        fontsize=18
    )

# Let the actual shapefile determine the map extent
ax.set_aspect(
    "equal"
)

plt.tight_layout()

plt.savefig(
    output_map,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("==========================================")
print("PROCESS COMPLETED")
print("==========================================")

print(
    f"Station CSV saved      : {output_csv}"
)

print(
    f"Station shapefile saved: {output_shapefile}"
)

print(
    f"Station map saved      : {output_map}"
)

print("==========================================")