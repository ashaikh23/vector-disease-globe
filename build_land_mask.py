import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

CANDIDATES = [
    "ne_50m_land/ne_50m_land.geojson",
    "ne_50m_land/ne_50m_land.shp",
    "ne_50m_land.shp",
]

def find_land_path():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find land polygons in: " + ", ".join(CANDIDATES))

def main():
    land_path = find_land_path()
    if land_path.endswith(".shp"):
        os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    print(f"[mask] Using land polygons at: {land_path}")

    # Read unique cells and force int32 for cell_id
    df = pd.read_csv("dataset.csv", usecols=["cell_id","lat_center","lon_center"]).drop_duplicates("cell_id")
    df["cell_id"] = df["cell_id"].astype("int32")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon_center"], df["lat_center"])],
        crs="EPSG:4326"
    )

    land = gpd.read_file(land_path)[["geometry"]]
    joined = gpd.sjoin(gdf, land, how="left", predicate="intersects")
    joined["is_land"] = joined["index_right"].notna().astype("int8")

    mask = joined[["cell_id","is_land"]].sort_values("cell_id").reset_index(drop=True)
    mask.to_csv("land_mask.csv", index=False)
    print(f"✅ Wrote land_mask.csv — land cells: {mask['is_land'].sum()} / {len(mask)}")

if __name__ == "__main__":
    main()
