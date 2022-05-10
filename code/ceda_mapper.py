import os

from ceda_file import CEDA_file, CEDA_Roads
from linker import Linker

import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiLineString
from shapely.wkt import loads
from multiprocessing import Pool
import itertools
import networkx as nx
from tqdm import tqdm

from scipy import spatial


TMC_ROAD_ATTR = ['LCD', 'ROADNUMBER', 'ROADNAME', 'FIRSTNAME', 'SECONDNAME', 'AREA_REF', 'AREA_NAME', 'geometry']
TMC_POINT_ATTR = ["LCD","geometry","SJTSK_Y","SJTSK_X", "ROADNUMBER", "ROADNAME", "ROA_LCD"]

def dist(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return (((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5) * 1.1

def wkt_loads(x):
    try:
        return loads(x)
    except:
        print(x)
        return None

class Mapper:
    def __init__(self):
        self.fcd_segments = None
        self.ceda_roads = None
        self.ceda_points = None
        self.ceda_network = None
        self.KDTree = None
        self.A = None
        self.matched = {}
        

    def load_files(self, directory, fcd_segments, ceda_roads_file, ceda_points_file):
        self.fcd_segments = fcd_segments
        self.ceda_roads = CEDA_Roads(os.path.join(directory, ceda_roads_file))
        self.ceda_roads.create_points_file(os.path.join(directory, ceda_points_file))
        self.ceda_points = CEDA_file(os.path.join(directory, ceda_points_file))


    def export_network(self, directory, file_name):
        nx.write_gpickle(self.ceda_network, os.path.join(directory, file_name))

    def read_network(self, directory, file_name):
        self.ceda_network = nx.read_gpickle(os.path.join(directory, file_name))

    def select_area(self, stencil_file, ceda_points, ceda_roads):
        #load stencil shapefile
        stencil = gpd.read_file(stencil_file)
        print(stencil.head())
        stencil_geometry = stencil.geometry.values[0]

        print("Points before filtering", ceda_points.shape[0])
        print("Roads before filtering", ceda_roads.shape[0])
        #remove all segments that are not in the stencil
        stencil.plot()
        ceda_points.plot()
        plt.show()
        new_points = ceda_points[ceda_points.geometry.apply(lambda x: x.within(stencil_geometry))]
        new_roads = ceda_roads[ceda_roads.geometry.apply(lambda x: x.within(stencil_geometry))]

        print("Points after filtering", new_points.shape[0])

        print("Saving ceda roads to shapefile:")
        print(new_roads.head())
        new_roads[["geometry","ROAD_ID", "METER", "ONEWAY","FW" ]].to_file("./../data/shp/ceda_roads_filtered.shp")
        return new_points


    def create_network(self, ceda_points, ceda_roads):
        """
        Create a network from the ceda points and roads geodataframes
        """
        # Create a networkx graph
        print("Creating CEDA network graph")
        #G = nx.Graph()
        G = nx.DiGraph()
        # Add nodes
        print("Adding nodes")
        for index, row in tqdm(ceda_points.iterrows()):
            point = (row.geometry.x, row.geometry.y)
            name = str(point[0]) + "," + str(point[1])
            G.add_node(name)
            nx.set_node_attributes(G, {name: point}, 'pos')

        # Add edges
        print("Adding edges")
        for index, row in tqdm(ceda_roads.iterrows()):
            points = [Point(p) for p in row.geometry.boundary]
            names = [str(p.x) + "," + str(p.y) for p in points]
            if G.has_node(names[0]) and G.has_node(names[1]):
                #TODO weight from distance to speed
                speed = row.METER

                if(row.ONEWAY == "FT"):
                    G.add_edge(names[0], names[1], weight=row.METER, segment=row.ROAD_ID)
                elif(row.ONEWAY == "TF"):
                    G.add_edge(names[1], names[0], weight=row.METER, segment=row.ROAD_ID)
                else:
                    G.add_edge(names[0], names[1], weight=row.METER, segment=row.ROAD_ID)
                    G.add_edge(names[1], names[0], weight=row.METER, segment=row.ROAD_ID)

        self.ceda_network = G
        self.build_KDTree()


    def build_KDTree(self):
        attributes = [node[1] for node in self.ceda_network.nodes.items()]
        tmp = []
        for attr in attributes:
            if 'pos' in attr.keys():
                tmp.append(attr)

        A = np.array([attr['pos'] for attr in tmp])
        self.A = A
        self.KDTree = spatial.KDTree(A)

    def find_nearest_neighbors(self, point, n=3, cutoff=100.0):
        nearest_points_query = self.KDTree.query([point.x, point.y], k=n)
        if n > 1:
            #print(nearest_points_query)
            return [Point(self.A[ix]) for ix in nearest_points_query[1]], nearest_points_query[0]
        else:
            print(nearest_points_query)
            return Point(self.A[nearest_points_query[1]]), nearest_points_query[0]


    def create_multi_line_string(self, path):
        path_segments = [self.ceda_network[path[p]][path[p+1]]['segment'] for p in range(0, len(path)-1)]
        if len(path_segments) == 0:
            return np.inf, []

        path = []
        length = 0
        for s in path_segments:
            geometries = self.ceda_roads[self.ceda_roads.ROAD_ID == s].geometry.values
            if(len(geometries) > 0):
                length += geometries[0].length
                path.append(geometries[0])

        m_path = MultiLineString(path)
        if(m_path.is_empty):
            print("empty geometry")
            print(path_segments)
        return length, m_path


    def map_segment_to_ceda(self, args):
        i, segment = args
        path = None
        best_path = []
        best_length = np.inf
        best_dists = np.inf
        #find closest points in ceda network
        N = 15 #8
        point_dists = {}
        closest_as, dists = self.find_nearest_neighbors(Point(segment.x_start, segment.y_start), n=N)
        for i, a in enumerate(closest_as):
            point_dists[str(a)] = dists[i]
        closest_bs, dists = self.find_nearest_neighbors(Point(segment.x_end, segment.y_end), n=N)
        for i, b in enumerate(closest_bs):
            point_dists[str(b)] = dists[i]
        #find shortest path between closest points in ceda network
        all_combinations = list(itertools.product(closest_as, closest_bs))
        for pair in all_combinations:
            A,B = pair
            A_name = str(A.x) + "," + str(A.y)
            B_name = str(B.x) + "," + str(B.y)

            try:
                path = nx.astar_path(self.ceda_network, A_name, B_name, heuristic=None, weight='weight')
                length, path = self.create_multi_line_string(path)
            except:
                path = []
                length = np.inf

            total_length = length + point_dists[str(A)] + point_dists[str(B)]

            if total_length < (best_length - 50*2): #TODO if difference is larger than distance from nearest point
                best_path = path
                best_length = total_length
                beest_dists = point_dists[str(A)] + point_dists[str(B)]

        if isinstance(best_path,list):
            print("No path found for segment", i, "number of point combinations", len(all_combinations))
            return ((segment.A.values[0], segment.B.values[0]), best_path)

        if best_path.is_empty:
            print("Best geometry is empty", i)
            

        return ((segment.A.values[0], segment.B.values[0]), best_path)


    def fcd_unique_segments(self, fcd_network):
        """
        Mark unique segments in the fcd segments dataset
        """
        fcd = fcd_network.copy()
        fcd['A'] = fcd.apply(lambda row: str(Point(row.x_start, row.y_start)), axis=1)
        fcd['B'] = fcd.apply(lambda row: str(Point(row.x_end, row.y_end)), axis=1)
        fcd_segments = fcd.groupby(['A', 'B'])
        print(len(fcd_network), "vs.", len(fcd_segments))
        return fcd


    def map_results_to_fcd_data(self, fcd_segments, results):
        mapped = pd.DataFrame()
        print("Mapping results back to FCD dataset:")
        groups = [(i, group) for i, group in fcd_segments.groupby(['A', 'B'])]
        for result in tqdm(results):
            ceda_path =  str(result[1])
            #print(ceda_path)
            for i,group in groups:
                if(str(i[0])==result[0][0]) and (str(i[1])==result[0][1]):
                    self.matched[i] = ceda_path
                    fcd_segments.loc[group.index.values, 'CEDA_geometry'] = [ ceda_path for l in range(0, len(group.index.values))]
                    mapped = pd.concat([mapped, fcd_segments.loc[group.index.values]])
                    groups.remove((i, group))
                    break

        #assign already matched 
        print("Assigning already matched segments")
        print("Rest of pairs",len(groups))
        for i,group in groups:
            if(i in self.matched.keys()):
                saved_path = self.matched[i]
                fcd_segments.loc[group.index.values, 'CEDA_geometry'] = [ saved_path for l in range(0, len(group.index.values))]
                mapped = pd.concat([mapped, fcd_segments.loc[group.index.values]])
                groups.remove((i, group))
        
        print(mapped.head())
        #print(mapped.info())
        return mapped


    def prepare_data_for_export(self, mapped):
        mapped["CEDA_geometry"] = mapped["CEDA_geometry"].apply(lambda x: wkt_loads(x))
        mapped = mapped.dropna(subset=["CEDA_geometry"])
        mapped = gpd.GeoDataFrame(mapped, geometry='CEDA_geometry')
        to_keep = ["index", "CEDA_geometry", "travel_time", "measurement_or_calculation_time","free_flow_speed", "start_node","end_node",
            "traffic_level", "free_flow_travel_time", "average_vehicle_speed","segment_length","LCD","ROADNAME"]
        return mapped[to_keep]


    def map_fcd_to_ceda_roads(self, fcd_network, filename="fcd_mapped", show=True):
        fcd_network["CEDA_geometry"] = None
        fcd_segments = self.fcd_unique_segments(fcd_network)

        iterables= []
        for i,group in fcd_segments.groupby(['A', 'B']):
            if i not in self.matched.keys():
                iterables.append((i,group[["A","B","x_start","x_end","y_start","y_end"]].head(1)))
   
        print("Mapping", len(iterables), "segments")
        with Pool(os.cpu_count()) as p:
            results = p.map(self.map_segment_to_ceda, iterables)

        mapped = self.map_results_to_fcd_data(fcd_segments, results)
        print('Mapped FCD samples',mapped.shape[0], "out of", fcd_network.shape[0], "in percents", (mapped.shape[0]/fcd_network.shape[0])*100)

        mapped = self.prepare_data_for_export(mapped)

        if show:
            fig, ax = plt.subplots(figsize=(10, 15))
            fcd_segments.plot(ax=ax, color='grey', alpha=0.05)
            mapped.plot(ax=ax, color='red')
            plt.savefig("./../data/png/{}.png".format(filename))

        return mapped


def get_linked_FCD_dataset(csv_path, data_path, chunk_name="", stencil_file="", show=False):
    print("Reading FCD dataset", csv_path)
    try:
        df = pd.read_csv(csv_path)
        fcd_segments = gpd.GeoDataFrame(df)
        print("FCD dataset read")
        points = []
        for l in fcd_segments.geometry.to_list():
            points.append(loads(l))

        fcd_segments["geometry"] = points
        print(fcd_segments.head())
        return fcd_segments
    except:
        print("FCD dataset not found")
        linker = Linker()
        linker.load_files(directory='./../data/CEDA/TMC_tabs/data/esri_format/SJTSK/' ,
                    fcd_file_name=data_path,
                    tmc_points='ltcze90_points_sjtsk.shp',
                    tmc_roads='ltcze90_roads_sjtsk.shp')

        linker.link_fcd_to_tmc_points()
        linker.link_points_to_roads()
        linker.link_fcd_to_tmc_roads()
        linker.create_fcd_segment_network()

        stencil_file = stencil_file
        linker.select_area(stencil_file)

        linker.segment_network.to_csv(csv_path, index=False)
        fcd_network_gpd = gpd.GeoDataFrame(linker.segment_network[["index","segment_length","travel_time","geometry"]], geometry="geometry")
        fcd_network_gpd.to_file("../data/shp/tunel/segments_"+chunk_name+".shp", driver="ESRI Shapefile")
        return linker.segment_network


def map_tunnel_data():
    fcd_files = [f for f in os.listdir('./../data/FCD/chunks/tunel') if f.endswith('.csv')]
    fcd_mapped = []
    for chunk in fcd_files[:]: #TODO
        csv_path = './../data/csv/tunel/'+chunk
        data_path = '../data/FCD/chunks/tunel/'+chunk
        fcd_network = get_linked_FCD_dataset(csv_path, data_path, chunk.split('.')[0], stencil_file="./../data/shp/tunel/tunel_small_2.shp")
        
        fcd_mapped.append(mapper.map_fcd_to_ceda_roads(fcd_network, chunk.split(".")[0], True))
    
    final_mapping = pd.concat(fcd_mapped)
    final_mapping.to_csv("./../data/csv/tunel_mapped.csv")
    final_mapping.to_file("./../data/shp/tunel_mapped.shp", driver="ESRI Shapefile")


def map_nabrezi_data():
    fcd_files = [f for f in os.listdir('./../data/FCD/chunks/nabrezi') if f.endswith('.csv')]
    fcd_mapped = []
    for chunk in fcd_files[:]: #TODO
        csv_path = './../data/csv/nabrezi/'+chunk
        data_path = '../data/FCD/chunks/nabrezi/'+chunk
        fcd_network = get_linked_FCD_dataset(csv_path, data_path, chunk.split('.')[0], stencil_file="./../data/shp/nabrezi/nabrezi.shp")
        
        fcd_mapped.append(mapper.map_fcd_to_ceda_roads(fcd_network, chunk.split(".")[0], True))
    
    final_mapping = pd.concat(fcd_mapped)
    final_mapping.to_csv("./../data/csv/nabrezi_mapped.csv")
    final_mapping.to_file("./../data/shp/nabrezi_mapped.shp", driver="ESRI Shapefile")



if __name__ == '__main__':
    stencil_tunel = "./../data/shp/tunel/tunel_small_2.shp"
    stencil_nabrezi = "./../data/shp/nabrezi/nabrezi.shp"

    fcd_type = "nabrezi"

    if fcd_type == "tunel":
        stencil = stencil_tunel
    else:
        stencil = stencil_nabrezi


    ceda = CEDA_Roads(file_name='./../data/CEDA/Roads/road.shp', stencil=stencil)
    try:
        ceda.read_point_file(file_name='./../data/CEDA/Roads/output_'+fcd_type+'_points.shp')
        print("CEDA points read")
        #ceda.read_intersection_file(file_name='./../data/CEDA/Roads/output_intersections.shp')
    except:
        print("Exporting new points and intersections from ceda roads.")
        ceda.create_points_and_intersection_files(directory='./../data/CEDA/Roads/', file_name='output_'+fcd_type)

    # CEDA network ready to be used with all fcd data
    mapper = Mapper()
    if not (os.path.exists('./../data/CEDA/Roads/ceda_graph_'+fcd_type+'.gpickle')):
        print("Creating new CEDA network graph")
        new_points = ceda.points #mapper.select_area(stencil_nabrezi, ceda.points, ceda.roads)
        mapper.create_network(new_points, ceda.roads)
        mapper.export_network(directory='./../data/CEDA/Roads/', file_name='ceda_graph_'+fcd_type+'.gpickle')
    else:
        mapper.read_network(directory='./../data/CEDA/Roads/', file_name='ceda_graph_'+fcd_type+'.gpickle')
        mapper.build_KDTree()
    #simple bidirectional network -> should be enough for now
    mapper.ceda_roads = ceda.roads.copy()

    if fcd_type == "tunel":
        map_tunnel_data()
    else:
        map_nabrezi_data()




    
