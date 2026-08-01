"""
Patches the Senegal base network with custom lines and a substation
that are missing or incorrectly removed from the OSM data.

Added lines (225 kV AC):
  - sub 111 (base bus 9)  <-> sub 134 (base bus 26)
  - sub 5   (base bus 3)  <-> sub 13  (base bus 11)
  - sub 141 (base bus 25) <-> sub 148 (base bus 22)
  - sub 133 (base bus 21) <-> new substation (bus 28)

Added substation:
  - Bus 28 at lon=-12.183247, lat=12.553736 (225 kV AC)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiLineString

buses_path = snakemake.input.buses
lines_path = snakemake.input.lines
sentinel   = snakemake.output.sentinel

NEW_BUS_ID  = 28
NEW_BUS_LON = -12.183247
NEW_BUS_LAT =  12.553736

NEW_LINES = [
    ("custom_sub111_sub134",  9, 26),
    ("custom_sub5_sub13",     3, 11),
    ("custom_sub141_sub148", 25, 22),
    ("custom_sub133_new",    21, NEW_BUS_ID),
]

def haversine_m(lon0, lat0, lon1, lat1):
    R = 6_371_000
    dlat = np.radians(lat1 - lat0)
    dlon = np.radians(lon1 - lon0)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat0)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


buses = gpd.read_file(buses_path)
lines = gpd.read_file(lines_path)

# ── add new substation ────────────────────────────────────────────────────────
if NEW_BUS_ID not in buses["bus_id"].values:
    new_bus = {
        "bus_id": NEW_BUS_ID, "station_id": NEW_BUS_ID, "voltage": 225000,
        "dc": False, "symbol": "substation", "under_construction": False,
        "tag_substation": "transmission", "tag_area": 0.0,
        "lon": NEW_BUS_LON, "lat": NEW_BUS_LAT, "country": "SN",
        "substation_lv": False, "geometry": Point(NEW_BUS_LON, NEW_BUS_LAT),
    }
    buses = pd.concat(
        [buses, gpd.GeoDataFrame([new_bus], crs=buses.crs)], ignore_index=True
    )

# ── add custom lines ──────────────────────────────────────────────────────────
for line_id, b0, b1 in NEW_LINES:
    if line_id in lines["line_id"].values:
        continue
    r0 = buses[buses["bus_id"] == b0].iloc[0]
    r1 = buses[buses["bus_id"] == b1].iloc[0]
    L = haversine_m(r0.lon, r0.lat, r1.lon, r1.lat)
    new_line = {
        "line_id": line_id, "circuits": 1.0, "voltage": 225000,
        "bus0": b0, "bus1": b1, "length": L, "dc": False,
        "under_construction": False,
        "geometry": MultiLineString([[(r0.lon, r0.lat), (r1.lon, r1.lat)]]),
    }
    lines = pd.concat(
        [lines, gpd.GeoDataFrame([new_line], crs=lines.crs)], ignore_index=True
    )

# ── save GeoJSON ──────────────────────────────────────────────────────────────
buses.to_file(buses_path, driver="GeoJSON")
lines.to_file(lines_path, driver="GeoJSON")

# ── save CSV with geometry as WKT (required by base_network.py) ───────────────
bn_dir = str(buses_path).replace("all_buses_build_network.geojson", "")

buses_csv = buses.copy()
buses_csv["geometry"] = buses_csv["geometry"].apply(lambda g: g.wkt)
buses_csv.to_csv(bn_dir + "all_buses_build_network.csv", index=False)

lines_csv = lines.copy()
lines_csv["geometry"] = lines_csv["geometry"].apply(lambda g: g.wkt)
lines_csv.to_csv(bn_dir + "all_lines_build_network.csv", index=False)

# ── write sentinel ────────────────────────────────────────────────────────────
open(sentinel, "w").close()
