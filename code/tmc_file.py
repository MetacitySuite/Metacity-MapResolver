from fileinput import filename
from select import select
import pandas as pd
import geopandas as gpd

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import numpy as np


class TMC_file:
    def __init__(self, file_name = ""):
        self.file_name = filename
        self.df = gpd.read_file(file_name, encoding="cp1250")

    def check_if_columns_equeal(self, column1, column2):
        return self.df[column1].equals(self.df[column2])

    def drop_columns(self, columns):
        """
        Drop columns from a dataframe
        """
        self.df = self.df.drop(columns, axis=1)

    def reload_file(self):
        self.df = pd.read_csv(self.file_name)
    
    def filter_prague_area(self, area):
        self.df = self.df[self.df.geometry.intersects(area)]

    def filter_prague_area_name(self, name):
        """
        Keep only the area of interest
        """
        self.df = self.df[self.df.AREA_NAME.str.contains(name)]

    def get_size(self):
        print("Rows:", self.df.shape[0])




class TMC_Points(TMC_file):
    def __init__(self, file_name = ""):
        super().__init__(file_name)
        self.road_identifier = "ROA_LCD"
        self.other_road_identifier = "INT_LCD"
        self.coords_x = "SJTSK_X"
        self.coords_y = "SJTSK_Y"
        self.geometry = "geometry"
        self.geometry_type = "Point"
        self.clean_road_identifier()

    def clean_road_identifier(self):
        print("Cleaning road identifier int TMC point file")
        print("Missing values:", self.df[self.road_identifier].isnull().sum(), "out of", self.df.shape[0])

        self.df[self.road_identifier] = self.df[self.road_identifier].fillna(0)
        self.df[self.road_identifier] = self.df[self.road_identifier].astype(int)



class TMC_Roads(TMC_file):
    def __init__(self, file_name = ""):
        super().__init__(file_name)
        self.road_identifier = "LCD"
        self.geometry = "geometry"
        self.geometry_type = "LineString"






