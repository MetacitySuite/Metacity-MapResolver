import pandas as pd
import geopandas as gpd
from shapely.wkt import loads


def load_files(mapped_csv, roads_csv):
    mapped_csv = pd.read_csv(mapped_csv)
    roads_csv = pd.read_csv(roads_csv)
    print(mapped_csv.head())
    print(mapped_csv.info())

    mapped_csv['CEDA_geometry'] = mapped_csv['CEDA_geometry'].apply(loads)
    mapped_gpd = gpd.GeoDataFrame(mapped_csv, geometry="CEDA_geometry")
    mapped_gpd["valid"] = mapped_gpd.CEDA_geometry.apply(lambda x: not x.is_empty)
    print(mapped_gpd.valid.value_counts())

    roads_csv['geometry'] = roads_csv['geometry'].apply(loads)
    roads_gpd = gpd.GeoDataFrame(roads_csv, geometry="geometry")
    print(roads_gpd.head())

    mapped_gpd["CEDA_geometry"] = mapped_gpd["CEDA_geometry"].apply(lambda x: list(x.geoms))
    print(mapped_gpd.head())
    print(mapped_gpd.shape[0])

    roads_gpd["time"] = None
    roads_gpd["free_flow_speed"] = None
    roads_gpd["average_vehicle_speed"] = None

    return mapped_gpd, roads_gpd

def mapped_to_lines(mapped_gpd):

    mapped_lines = mapped_gpd.explode('CEDA_geometry')
    mapped_lines["segment_length"] = mapped_lines.CEDA_geometry.apply(lambda x: x.length)
    mapped_lines["CEDA_geometry"] = mapped_lines.CEDA_geometry.astype(str)
    print(mapped_lines.head())
    print(mapped_lines.shape[0])
    mapped_lines["time"] = mapped_lines.measurement_or_calculation_time.astype(str)
    mapped_lines["free_flow_speed"] = mapped_lines.free_flow_speed.astype(str)
    mapped_lines["average_vehicle_speed"] = mapped_lines.average_vehicle_speed.astype(str)

    lines_grouped = mapped_lines.groupby(['CEDA_geometry'])
    print(len(lines_grouped))
    return lines_grouped

def assign_and_save(roads_gpd, lines_grouped, filename):
    assigned_groups = []

    for i, road in roads_gpd.iterrows():
        try:
            road_information = lines_grouped.get_group(str(road.geometry))
            assigned_groups.append(road_information.CEDA_geometry.values[0])
            roads_gpd.loc[i, "time"] = ",".join(road_information.time.to_list())
            roads_gpd.loc[i, "free_flow_speed"] = ",".join(road_information.free_flow_speed.to_list())
            roads_gpd.loc[i, "average_vehicle_speed"] = ",".join(road_information.average_vehicle_speed.to_list())
        except KeyError:
        #    print("no information available for road",road.geometry)
            pass

    print(roads_gpd.info())
    #save to csv
    roads_gpd.to_file('./../data/shp/'+filename+'.shp')
    roads_gpd.to_csv('./../data/csv/'+filename+'.csv', encoding='utf-8', index=False)


mapped_gpd, roads_gpd = load_files(mapped_csv='./../data/csv/tunel_mapped.csv', roads_csv='./../data/csv/roads_shortened.csv')
lines_grouped = mapped_to_lines(mapped_gpd)
assign_and_save(roads_gpd, lines_grouped, 'roads_with_information_tunel')

