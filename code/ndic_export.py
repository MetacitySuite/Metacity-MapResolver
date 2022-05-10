import pandas as pd
import geopandas as gpd
from shapely.wkt import loads


def load_data(ndic_file):
    restrictions  = pd.read_csv(ndic_file, sep=';', encoding='cp1250')
    #restrictions['geometry'] = restrictions['geometry'].apply(loads)
    restrictions = restrictions.dropna(axis=1)
    print(restrictions.head())
    print(restrictions.info())
    restrictions['geometry'] = restrictions['geom_openlr_line'].apply(loads)
    
    restrictions = gpd.GeoDataFrame(restrictions, geometry="geometry")
    restrictions = restrictions.set_crs(epsg=4326)
    restrictions = restrictions.to_crs("EPSG:5514")
    print(restrictions.head())
    print
    restrictions.to_file('./../data/shp/restrictions_ndic.shp')
    restrictions.to_csv('./../data/csv/restrictions_ndic_tunel_out.csv', sep=';', encoding='utf-8')


load_data('./../data/csv/tunel/tunel_ndic.csv')