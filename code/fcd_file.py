
import pandas as pd
import geopandas as gpd
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class FCD_File:
    def __init__(self, file_name = ""):
        self.file_name = file_name
        self.df = pd.read_csv(file_name, sep=';', encoding='cp1250')


    def drop_empty_columns(self):
        """
        Drop empty columns from a dataframe
        """
        self.df = self.df.dropna(axis=1, how='all')

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

    def parse_nodes(self):
        #print(self.df.predefined_location)
        self.df["start_node"] = self.df['source_identification'].str.split('T').str[-1].astype(int)
        self.df["end_node"] = self.df['source_identification'].str.split('T').str[1].str[1:].astype(int)
        
        for col in ["source_identification", "predefined_location"]:
            if col in self.df.columns:
                self.df = self.df.drop(col, axis=1)

    def get_size(self):
        print("Rows:", self.df.shape[0])





