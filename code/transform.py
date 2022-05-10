from heapq import merge
import pandas as pd
from shapely.wkt import loads
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import numpy as np
import base64
import orjson
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def npfloat64_to_string(data):
    data = np.array(data, dtype=np.float64)
    return base64.b64encode(data).decode('utf-8')

def km_per_hour_to_m_per_sec(km_per_hour):
    return km_per_hour * 1000 / 3600

def interpolate_position_each_second(trajectory: LineString, total_travel_time):
    for i in range(int(total_travel_time + 2)): #yes + 2 is correct mostly
        yield trajectory.interpolate(float(i) / total_travel_time, normalized=True)

def process_lstring(geometry: LineString, speed, speed_threshold):
    total_travel_time = geometry.length / max(km_per_hour_to_m_per_sec(speed), 5)
    return geometry, total_travel_time    

def process_mlstring(geometry: MultiLineString, speed, speed_threshold):
    merged = linemerge(geometry)
    if merged.type != "LineString":
        print(merged)
        raise Exception("Merged geometry is not a LineString")
    return process_lstring(merged, speed, speed_threshold)

TIME_ON_SCREEN = 5

def process_geometry(geometry, speed, speed_threshold, time, records):
    if geometry.type == "MultiLineString":
        geom, ttt = process_mlstring(geometry, speed, speed_threshold)
    elif geometry.type == "LineString":
        geom, ttt = process_lstring(geometry, speed, speed_threshold)
    else:
        print(f"Unknown geometry type: {geometry.type}")
        return

    t = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f %z')
    t = t.second + t.minute * 60 + t.hour * 3600
    start_offset = 20 + 30 / max(speed, 1)
    start_offset_limits = 5
    length_offset_limits = 10

    for i in range(TIME_ON_SCREEN):
        points = [ p.coords[0] for p in interpolate_position_each_second(geom, ttt + random.randint(-length_offset_limits, length_offset_limits)) ]
        records.append({
            "geometry": npfloat64_to_string(points), 
            "meta": {
                "speed": speed, 
                "speed_threshold": speed_threshold, 
                "start": int(t + i * start_offset + random.randint(-start_offset_limits, start_offset_limits)),
                "metatype": "time_series"
            }})


if "__main__" == __name__:
    tqdm.pandas()
    fcd_type = "tunel"
    
    records = []

    df = pd.read_csv("./../data/csv/"+fcd_type+"_mapped.csv", encoding="utf-8")
    df["CEDA_geometry"] = df.progress_apply(lambda row: loads(row["CEDA_geometry"]), axis=1)
    df.progress_apply(lambda row: process_geometry(row["CEDA_geometry"], row["average_vehicle_speed"], row["free_flow_speed"], row["measurement_or_calculation_time"], records), axis=1)
    print(len(records))

    chunk_size = 10000
    for i in tqdm(range(0, len(records), chunk_size)):
        with open(f"./../data/out/"+fcd_type+"/chunk_{i:04d}.sim", "wb") as f:
            f.write(orjson.dumps(records[i:i+chunk_size]))
    






