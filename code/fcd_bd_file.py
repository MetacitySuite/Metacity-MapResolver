import pandas as pd

def drop_empty_columns(df):
    """
    Drop empty columns from a dataframe
    """
    return df.dropna(axis=1, how='all', inplace=False)

def check_if_columns_equeal(df, column1, column2):
    return df[column1].equals(df[column2])

def drop_single_value_columns(df):
    """
    Drop columns with only one value
    """
    to_drop = []
    for col in df.columns:
        if len(df[col].unique()) < 2:
            to_drop.append(col)
            
    return df.drop(to_drop, axis=1)

def drop_columns(df, columns):
    """
    Drop columns from a dataframe
    """
    return df.drop(columns, axis=1)


keep_columns = ['source_identification', 'measurement_or_calculation_time', 'traffic_level', 'queue_exists', 
'queue_length', 'average_vehicle_speed','travel_time', 'free_flow_travel_time', 'free_flow_speed']  


def load_file_in_chunks(path, sep=";", identifier="", chunksize=200000):
    """
    Load a file in chunks
    """
    for i, fcd in enumerate(pd.read_csv(path, sep=sep, encoding="cp1250", chunksize=chunksize)):
        print(fcd.head(2))
        fcd = drop_empty_columns(fcd)
        fcd = drop_single_value_columns(fcd)
        
        fcd = fcd[list(set(fcd.columns).intersection(set(keep_columns)))]
        fcd.to_csv("./../data/FCD/chunks/"+identifier+"/fcd_"+str(i)+".csv", sep=";", encoding="cp1250", index=False)
        print("Exporting chunk: " + str(i), "out of " + str(int(fcd.shape[0]/chunksize)))


#load_file_in_chunks("./../data/FCD/RSD_-_FCD_-_Traffic_parameters_2022_03_12_-_vsechna_nasbirana_data.csv", identifier="tunel")
load_file_in_chunks("./../data/FCD/fcd_20220205_20220410_202204201214.csv",sep=',', identifier="nabrezi")