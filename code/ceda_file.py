import pandas as pd
import geopandas as gpd


import numpy as np
from shapely.geometry import Polygon, LineString, Point
from multiprocessing import Pool
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from shapely import wkt

CEDA_ROADS_ATTR = ['FCC','ROAD_ID','ON','DF','FW','NC','DS','ONEWAY','METER', 'geometry']

def split_segment_recursively(point_a, point_b, cutoff):
    distance = point_a.distance(point_b)
    if distance > cutoff:
        middle_point = Point((point_a.x + point_b.x) / 2, (point_a.y + point_b.y) / 2)
        return split_segment_recursively(point_a, middle_point, cutoff)[:-1] + split_segment_recursively(middle_point, point_b, cutoff)[:]
    else:
        return [point_a, point_b]



def split_segment(args):
    i, row, cutoff = args
    new_segments = []
    distance = 0.0

    points = [Point(p[0], p[1]) for p in row.geometry.coords]
    new_points = []

    for p_before, p in zip(points[:-1], points[1:]):
        new_points.extend(split_segment_recursively(p_before, p, cutoff)[:-1])
    new_points.append(points[-1])
    points = new_points

    distances = [p_before.distance(p) for p_before, p in zip(points[:-1], points[1:])]  # distance between points
    accum_distances = pd.Series(distances).cumsum().to_list()

    segment_points = []
    segment_points.append(points[0])
    last_distance = 0.0

    for i,point in enumerate(points[1:]):
        distance = accum_distances[i] - last_distance
        segment_points.append(point)

        if(distance >= cutoff):
            new_segments.append(segment_points)
            last_distance = distance
            segment_points = []
            segment_points.append(point)
            

    if(len(segment_points) > 1):
        new_segments.append(segment_points)

    
    new_roads = pd.DataFrame(data=None, index=None,
        columns=['FCC', 'ROAD_ID', 'ON', 'DF', 'FW', 'NC', 'DS', 'ONEWAY', 'METER', 'geometry'], 
        )

    new_roads["geometry"] = [LineString(segment) for segment in new_segments]
    new_roads["ROAD_ID"] = [row.ROAD_ID+"_"+str(i) for i in range(len(new_segments))]
    new_roads["METER"] = [LineString(segment).length for segment in new_segments]

    for col in row.index:
        if col not in ['geometry',"ROAD_ID", "METER"]:
            new_roads[col] = row[col]

    return new_roads


class CEDA_file:
    def __init__(self, file_name = ""):
        self.file_name = file_name
        self.df = gpd.read_file(file_name, encoding="cp1250")

    def check_if_columns_equeal(self, column1, column2):
        return self.df[column1].equals(self.df[column2])

    def drop_columns(self, columns):
        """
        Drop columns from a dataframe
        """
        self.df = self.df.drop(columns, axis=1)

    def reload_file(self):
        self.df = gpd.read_file(self.file_name, encoding="cp1250")
    
    def filter_prague_area(self, area):
        self.df = self.df[self.df.geometry.intersects(area)]

    def filter_prague_area_name(self, name):
        """
        Keep only the area of interest
        """
        self.df = self.df[self.df.AREA_NAME.str.contains(name)]

    def get_size(self):
        print("Rows:", self.df.shape[0])





class CEDA_Roads(CEDA_file):
    def __init__(self, file_name = "", stencil=""):
        super().__init__(file_name)
        if(len(stencil) > 0):
            self.filter_area(stencil)
        self.road_identifier = "ROAD_ID"
        self.name_identifier = "ON"
        self.geometry = "geometry"
        self.geometry_type = "LineString"
        self.df = self.df[CEDA_ROADS_ATTR]
        self.df.ROAD_ID = self.df.ROAD_ID.astype(str)
        self.roads = self.df.copy()
        self.cutoff = 150.0

        #check if file exists and load it
        if(os.path.isfile("../data/csv/roads_shortened.csv")):
            self.df = pd.read_csv("../data/csv/roads_shortened.csv", index_col=0)
            self.df["geometry"] = self.df.geometry.apply(lambda x: wkt.loads(x))
            self.df = gpd.GeoDataFrame(self.df, geometry='geometry')
            self.df.ROAD_ID = self.df.ROAD_ID.astype(str)
            self.roads = self.df.copy()
            print("Loaded roads from csv")
        else:
            print("No roads csv file found")
            self.filter_car_roads()
            self.shorten_long_roads()

    def filter_area(self, area):
        area = gpd.read_file(area).geometry.values[0]
        #new_roads = ceda_roads[ceda_roads.geometry.apply(lambda x: x.within(stencil_geometry))]
        self.df = self.df[self.df.geometry.intersects(area)]


    def shorten_long_roads(self):
        new_roads = []
        
        self.df = self.df[self.df.geometry.type == self.geometry_type]
        short_paths = self.df[self.df.METER <= self.cutoff]
        long_paths = self.df[self.df.METER > self.cutoff]
    
        iterables = []
        print("Preparing long paths")
        for i, row in tqdm(long_paths.iterrows()):
            iterables.append((i, row, self.cutoff))

        with Pool(os.cpu_count()) as p:
            new_roads = p.map(split_segment, iterables)

        print("old roads", self.df.shape[0])
        print("\tshort paths", short_paths.shape[0])
        print("\tlong paths", long_paths.shape[0])
        split_paths = gpd.GeoDataFrame(pd.concat(new_roads), geometry='geometry')
        self.df = pd.concat([short_paths, split_paths])
        print("new roads", self.df.shape[0])
        self.roads = self.df.copy()
        #save as csv and shapefile
        self.export_shapefile(self.df, "../data/shp/roads_shortened.shp")
        self.df.to_csv("../data/csv/roads_shortened.csv")


    def export_shapefile(self, gdf, file_name):
        if(gdf.shape[0] > 0):
            gdf.to_file(file_name)
        return


    def filter_car_roads(self):
        print(self.df.FW.unique())
        print("number of roads:", self.df.shape[0])
        self.df = self.df[self.df.DF.isin([1,2,3])]
        self.df = self.df[self.df.FW.isin([1,2,3,4,6,7,10,11,12,13])]
        print("Number of car roads:", self.df.shape[0])

        #self.df = self.df[self.df.DS.isin([1,2,3])]
        #print("Number of car roads:", self.df.shape[0])
        self.export_shapefile(self.df,"../data/CEDA/Roads/roads_car.shp")



    def get_endpoints(self):
        self.df.loc[:,"endpoints"] = self.df.geometry.apply(lambda x: [Point(p.x, p.y) for p in x.boundary])
        print(self.df.head())
        self.df.to_csv("../data/csv/df_endpoints.csv")


    def load_endpoints(self):
        self.df = gpd.read_csv("../data/csv/df_endpoints.csv")


    def get_segment_road_ids(self, endpoints):
        partial_points = dict()
        for point in endpoints:
            df_roads = list(self.df.loc[((self.df.A_x == point[0]) & (self.df.A_y == point[1])) 
                                        | ((self.df.B_x == point[0])&(self.df.B_y == point[1])), "ROAD_ID"].values)

            partial_points[tuple(point)] = list(set(df_roads))
        print(len(endpoints))
        return partial_points

        
    def create_points_optimized(self):
        points = dict()
        self.df["A_x"] = self.df.endpoints.apply(lambda p: p[0].x)
        self.df["A_y"] = self.df.endpoints.apply(lambda p: p[0].y)
        self.df["B_x"] = self.df.endpoints.apply(lambda p: p[-1].x)
        self.df["B_y"] = self.df.endpoints.apply(lambda p: p[-1].y)

        endpoints_A = set([ (row.A_x, row.A_y) for i,row in self.df.iterrows()])
        endpoints_B = set([ (row.B_x, row.B_y) for i,row in self.df.iterrows()])

        endpoints =  np.array(list(endpoints_A.union(endpoints_B)))
        print("Number of endpoints:", endpoints.shape)

        endpoint_chunks = np.array_split(endpoints, os.cpu_count()*4)
        print("Number of endpoint chunks:", len(endpoint_chunks))

        with Pool(12) as p:
            results = p.map(self.get_segment_road_ids, endpoint_chunks)

        print("Connecting results:")
        for result in tqdm(results):
            points.update(result)

        return points
        

    def create_points(self):
        endpoints = []
        for road in self.df.endpoints.apply(lambda x: [Point(p.x, p.y) for p in x]):
            endpoints.extend(road)

        print("Number of endpoints:", len(endpoints))
        points = dict()
        for point in endpoints:
            roads = self.df.loc[self.df.endpoints.apply(lambda x: point in x)] #2n*n

            if (point.x, point.y) in points.keys():
                existing_roads = list(points[(point.x, point.y)])
                existing_roads.extend(roads.ROAD_ID.unique())
                points[(point.x, point.y)] = list(set(existing_roads))
            else:
                points[(point.x, point.y)] = list(roads.ROAD_ID.unique())
        return points

    
    def create_points_and_intersection_files(self, directory, file_name):
        """
        Create a points file from intersections in road geometry
        """

        try:
            self.load_endpoints()
        except:
            self.get_endpoints()

        self.roads = self.df.copy()
        points = self.create_points_optimized()
        print("Points extracted")

        self.points = gpd.GeoDataFrame()
        self.points['geometry'] = [Point(p[0], p[1]) for p in points.keys()]
        self.points["road_count"] = [ len(road_ids) for road_ids in points.values() ]
        self.points["roads"] = [ ",".join(road_ids) for road_ids in points.values() ]
        print(self.points.info())
        print(self.points.head())

        self.export_shapefile(self.points, directory + file_name+"_points.shp")
        print("Points created")
        print("Points:", self.points.shape[0])

        #self.intersections = self.points[self.points.road_count > 2]
        #self.intersections.reset_index(inplace=True, drop=True)
        #self.export_shapefile(self.intersections, directory + file_name+"_intersections.shp")
        #print("Intersections:", self.intersections.shape[0])
        return


    def read_point_file(self, file_name):
        self.points = gpd.read_file(file_name)

    
    def read_intersection_file(self, file_name):
        self.intersections = gpd.read_file(file_name)

        





