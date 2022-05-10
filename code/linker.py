from operator import le
import os

from fcd_file import FCD_File
from tmc_file import TMC_file, TMC_Points, TMC_Roads

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point


TMC_ROAD_ATTR = ['LCD', 'ROADNUMBER', 'ROADNAME', 'FIRSTNAME', 'SECONDNAME', 'AREA_REF', 'AREA_NAME', 'geometry']
TMC_POINT_ATTR = ["LCD","geometry","SJTSK_Y","SJTSK_X", "ROADNUMBER", "ROADNAME", "ROA_LCD"]


class Linker:
    def __init__(self):
        self.point_network = None
        self.road_network = None

        self.point_road_network = None
        self.fcd_road_net = None
        

    def load_files(self, directory, fcd_file_name, tmc_points, tmc_roads):
        self.fcd_file = FCD_File(fcd_file_name)
        self.fcd_file.parse_nodes()
        self.fcd_file.drop_empty_columns()
        self.tmc_points = TMC_Points(directory+tmc_points)
        self.tmc_roads = TMC_Roads(directory+tmc_roads)


    def select_area(self, stencil_file):
        #load stencil shapefile
        stencil = gpd.read_file(stencil_file)
        print(stencil.head())
        stencil_geometry = stencil.geometry.values[0]

        print("Segments before filtering", self.segment_network.shape[0])
        #remove all segments that are not in the stencil
        self.segment_network = self.segment_network[self.segment_network.geometry.apply(lambda x: x.within(stencil_geometry))]
        print("Segments after filtering", self.segment_network.shape[0])


    def show_and_save_origin_dest_points(self, df):
        df_roads = df.merge(self.tmc_roads.df, left_on="ROA_LCD", right_on="LCD", how="left", suffixes=(None, None))
        gdf_roads = gpd.GeoDataFrame(df_roads, geometry="geometry")
   
        fig, ax = plt.subplots(figsize=(15,20))
        plt.plot(df.x_start, df.y_start, 'ro', alpha=0.15, markersize=4)
        plt.plot(df.x_end, df.y_end, 'bo', alpha=0.15, markersize=4)
        #plot line from start to end
        plt.plot([df.x_start, df.x_end], [df.y_start, df.y_end], 'k-', alpha=0.15)
        #plot line from geometry_road to geometry_point
        gdf_roads.geometry.plot(alpha=0.15, ax=ax)
        
        for i, row in df.iterrows():
            plt.annotate(i, (row.x_start, row.y_start))
            plt.annotate(i, (row.x_end, row.y_end))

        plt.savefig("./../data/png/origin_dest_points.png")


    def link_fcd_to_tmc_points(self):
        tmc_points_selection = self.tmc_points.df[TMC_POINT_ATTR]

        self.point_network_start = self.fcd_file.df.merge(tmc_points_selection, left_on="start_node", right_on="LCD", how="left", suffixes=(None, "_start"))
        self.point_network_end = self.fcd_file.df.merge(tmc_points_selection, left_on="end_node", right_on="LCD", how="left", suffixes=(None, "_end"))
        
        self.point_network_start = self.point_network_start.rename(columns={"SJTSK_X": "x_start", "SJTSK_Y": "y_start"})
        self.point_network_start = self.point_network_start.reset_index(drop=False)

        self.point_network_end = self.point_network_end.rename(columns={"SJTSK_X": "x_end", "SJTSK_Y": "y_end"})
        self.point_network_end = self.point_network_end.reset_index(drop=False)

        print("Creating FCD dataset with linked TMC point coordinates and ROA_LCD identifier.")
        self.point_network = self.point_network_start.copy()
        self.point_network["x_end"] = self.point_network_end["x_end"]
        self.point_network["y_end"] = self.point_network_end["y_end"]
        self.point_network["end_node"] = self.point_network_end["end_node"]
        #self.show_and_save_origin_dest_points(self.link_network.head(20))


    def show_point_network(self):
        print(self.point_network.info())
        gpd_network = gpd.GeoDataFrame(self.point_network, geometry="geometry")

        fig, ax = plt.subplots(figsize=(15,20))
        gpd_network.plot(ax=ax, column='free_flow_speed', cmap='viridis', legend=True,  markersize=3)
        plt.savefig("./../data/png/point_network.png")


    def show_point_road_network(self):
        gpd_network_road = gpd.GeoDataFrame(self.point_road_network, geometry="geometry_road")
        gpd_network_point = gpd.GeoDataFrame(self.point_road_network, geometry="geometry_point")
        fig, ax = plt.subplots(figsize=(15,20))
        gpd_network_road.plot(ax=ax, column='CLASS', cmap='viridis', alpha=0.03, legend=True)
        gpd_network_point.plot(ax=ax, color='red', markersize=2)
        plt.savefig("./../data/png/point_road_network.png")


    def return_common_columns(self, df1, df2):
        return df1.columns.intersection(df2.columns)


    def link_points_to_roads(self):
        print("Common columns")
        print(self.return_common_columns(self.point_network, self.tmc_roads.df))
        columns_to_drop = ['CID', 'TABCD', 'CLASS', 'TCD', 'STCD', 'ROADNUMBER', 'ROADNAME',
       'FIRSTNAME', 'SECONDNAME', 'AREA_REF', 'AREA_NAME']
        tmc_points = self.tmc_points.df.copy()
        tmc_points = tmc_points.drop(columns=columns_to_drop, axis=1)
        self.point_road_network = tmc_points.merge(self.tmc_roads.df, left_on="ROA_LCD", right_on="LCD", how="left", suffixes=("_point", "_road"))


    def link_fcd_to_tmc_roads(self):
        print("Common columns", self.return_common_columns(self.point_network, self.tmc_roads.df[TMC_ROAD_ATTR]))
        print("FCD point network columns", self.point_network.columns)
        self.road_network = self.point_network.merge(self.tmc_roads.df[TMC_ROAD_ATTR], left_on="ROA_LCD", right_on="LCD", how="left", suffixes=(None, "_TMC"))


    def export_network_to_shapefile(self, gdf, filename):
        print("Exporting network to shapefile",   filename)
        print(gdf.head(2))
        gdf.reset_index(drop=True, inplace=True)
        gdf.to_file(filename, encoding='utf-8')


    def create_fcd_segment_network(self):
        self.segment_network = self.road_network.copy()
        self.segment_network["traffic_level"] = [int(l.split('level')[-1]) for l in self.segment_network.traffic_level]
        self.segment_network["geometry"] = self.segment_network.apply(lambda row: LineString([(row.x_start, row.y_start), (row.x_end, row.y_end)]), axis=1)
        self.segment_network["segment_length"] = self.segment_network.apply(lambda row: row.geometry.length, axis=1)
        print(self.segment_network.head(2))
        print("Missing values", self.segment_network.geometry.isnull().sum())

        date_column = "created_at" if "created_at" in self.segment_network.columns else "measurement_or_calculation_time"
        self.export_network_to_shapefile(gpd.GeoDataFrame(self.segment_network[[date_column,"geometry"]], geometry="geometry"), "./../data/shp/segment_network.shp")

    
    def plot_segment_attribute(self, gdf, attribute):
        print("Plotting attribute", attribute)
        fig, ax = plt.subplots(figsize=(15,20))
        gdf.plot(ax=ax, column=attribute, cmap='coolwarm', legend=True,  markersize=3, alpha=0.02)
        plt.savefig("./../data/png/segment_network_" + attribute + ".png")
    

    def show_segment_network(self):
        print(self.segment_network.info())
        gpd_network = gpd.GeoDataFrame(self.segment_network, geometry="geometry")

        self.plot_segment_attribute(gpd_network, "free_flow_speed")
        self.plot_segment_attribute(gpd_network, "traffic_level")
        self.plot_segment_attribute(gpd_network, "segment_length")
        self.plot_segment_attribute(gpd_network, "data_quality")
        self.plot_segment_attribute(gpd_network, "average_vehicle_speed")
        self.plot_segment_attribute(gpd_network, "travel_time")
        self.plot_segment_attribute(gpd_network, "free_flow_travel_time")


    def create_fcd_network(self):
        road_segments = []
        road_segments_prg = []
        print("Road network")
        print(self.road_network.info())

        for g,road in self.road_network.groupby("ROA_LCD"):
            new_road_seg = gpd.GeoDataFrame()

            new_road_seg["geometry"] = road.geometry_TMC
            s = gpd.GeoSeries(new_road_seg.geometry)
            new_road_seg["ROAD_LENGTH"] =  s.length.mean()
            new_road_seg["ROAD_LCD"] = g
            new_road_seg["ROAD_NAMES"] = str(road.ROADNAME.unique())
            new_road_seg["ROAD_AVG_SPEED"] = road.free_flow_speed.mean()
            new_road_seg["ROAD_AVG_VEH_SPEED"] = road.average_vehicle_speed.mean()
            new_road_seg["ROAD_AVG_TTIME"] = road.free_flow_travel_time.mean()
            new_road_seg["ROAD_AVG_VEH_TTIME"] = road.travel_time.mean()
            new_road_seg["ROAD_AVG_TRAFFIC"] = np.mean([int(l.split('level')[-1]) for l in road.traffic_level])
            new_road_seg["ROAD_FCD_REP"] = road.shape[0]
            new_road_seg["points"] = list(road.geometry)
            new_road_seg["x"] = road.x_start
            new_road_seg["y"] = road.y_start

            road_segments.append(new_road_seg)
            geom = road.geometry_TMC.values[0]
            if geom is None or geom.within(Polygon([(-770000, -1.07e6),(-710000,-1.07e6),(-710000, -1.02e6),(-770000,-1.02e6)])):
                road_segments_prg.append(new_road_seg)

        self.fcd_road_net = gpd.GeoDataFrame(pd.concat(road_segments))
        self.fcd_road_net_prg = gpd.GeoDataFrame(pd.concat(road_segments_prg))

        shp_columns = ["geometry","ROAD_LCD", "ROAD_NAMES", "ROAD_FCD_REP"]
        self.export_network_to_shapefile(self.fcd_road_net_prg[shp_columns], "./../data/shp/fcd_road_net_prg.shp")
        

    def show_and_save_plot(self, df, column, title, filename):
        if(df[column].unique().shape[0] < 2):
            print("Column", column, "has only one value")
            return
        fig, ax = plt.subplots(figsize=(15,12))
        df.plot(ax=ax, column=column, cmap='viridis', legend=True,  alpha=1.0)
        plt.title(title)
        print("Saving", filename)
        plt.savefig(filename)


    def show_and_save_plot_with_points(self, df, column, point_column, title, filename):
        if(df[column].unique().shape[0] < 2):
            print("Column", column, "has only one value")
            return
        fig, ax = plt.subplots(figsize=(15,12))
        df.plot(ax=ax, column=column, cmap='viridis', legend=True,  alpha=1.0)
        for p in df[point_column]:
            plt.plot(p.x, p.y, 'ro', markersize=1.0, alpha=0.05)

        plt.plot(df.x, df.y, 'bo', markersize=2.0, alpha=0.1)
        plt.title(title)
        print("Saving", filename)
        plt.savefig(filename)
        

    def show_fcd_network(self):
        self.show_and_save_plot(self.fcd_road_net, "ROAD_AVG_SPEED", "Average speed of FCD roads", "./../data/png/fcd_road_speed.png")
        self.show_and_save_plot(self.fcd_road_net, "ROAD_AVG_VEH_SPEED", "Average vehicle speed of FCD vehicles", "./../data/png/fcd_road_vehicle_speed.png")
        self.show_and_save_plot(self.fcd_road_net, "ROAD_AVG_TTIME", "Average travel time of FCD roads", "./../data/png/fcd_road_travel_time.png")
        self.show_and_save_plot(self.fcd_road_net, "ROAD_AVG_TRAFFIC", "Average traffic level of FCD roads", "./../data/png/fcd_road_traffic_level.png")
        self.show_and_save_plot(self.fcd_road_net, "ROAD_FCD_REP", "Number of FCD records on road", "./../data/png/fcd_road_record_count.png")
        self.show_and_save_plot(self.fcd_road_net, "ROAD_LENGTH", "Length of FCD roads", "./../data/png/fcd_road_length.png")


    def show_fcd_network_prague(self):
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_AVG_SPEED", "Average speed of FCD roads in Prague", "./../data/png/fcd_road_speed_prague.png")
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_AVG_VEH_SPEED", "Average vehicle speed of FCD vehicles in Prague", "./../data/png/fcd_road_vehicle_speed_prague.png")
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_AVG_TTIME", "Average travel time of FCD roads in Prague", "./../data/png/fcd_road_travel_time_prague.png")
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_AVG_TRAFFIC", "Average traffic level of FCD roads in Prague", "./../data/png/fcd_road_traffic_level_prague.png")
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_FCD_REP", "Number of FCD records on road in Prague", "./../data/png/fcd_road_record_count_prague.png")
        self.show_and_save_plot(self.fcd_road_net_prg, "ROAD_LENGTH", "Length of FCD roads in Prague", "./../data/png/fcd_road_length_prague.png")
        self.show_and_save_plot_with_points(self.fcd_road_net_prg, "ROAD_LENGTH", "points", "Average speed of FCD roads in Prague with FCD points", "./../data/png/fcd_road_speed_prague_points.png")


if __name__ == '__main__':

    linker = Linker()
    linker.load_files(directory='../data/CEDA/TMC_tabs/data/esri_format/SJTSK/' ,
                fcd_file_name='../data/FCD/fcd_sample.csv', #TODO change to fcd_sample.csv
                tmc_points='ltcze90_points_sjtsk.shp',
                tmc_roads='ltcze90_roads_sjtsk.shp')

    linker.link_fcd_to_tmc_points()
    linker.show_point_network()

    linker.link_points_to_roads() #TMC 
    linker.show_point_road_network()

    linker.link_fcd_to_tmc_roads()

    linker.create_fcd_segment_network()
    linker.show_segment_network()
    linker.segment_network.to_csv('../data/csv/FCD_segment_network.csv', index=False)

    exit()
    linker.create_fcd_network()
    linker.show_fcd_network()
    linker.show_fcd_network_prague()