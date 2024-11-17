import os
import re
import pandas as pd
from datetime import datetime
import openpyxl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import geopandas as gpd
from shapely.geometry import Point

'''
TODO:
Merge benthic macrofauna dataset with sediment quality. Spionidae, and maybe Nephtyidae or ...
'''



class Preprocessor():
    def __init__(self, sedimentPath, faunaPath, savePath) -> None:
        self.sedimentPath = sedimentPath
        self.faunaPath = faunaPath
        self.savePath = savePath
        self.df = pd.read_excel(self.sedimentPath, skiprows=2)
        self.benthicMacrofauna = pd.read_excel(self.faunaPath,  header=[0, 1], index_col=0)
        # print(self.benthicMacrofauna.head())
        
    def load_dataset(self):
        
        print("loaded dataset")
        # df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        # df_cleaned = df.dropna(axis=1, how='all')
        df_cleaned = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        df_cleaned.columns = df_cleaned.columns.str.replace(' ', '', regex=True)
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        df_cleaned = df_cleaned.dropna(subset=['S'])
        df_cleaned = df_cleaned.drop(columns="J'")
        df_cleaned = df_cleaned.dropna()
        
        self.benthicMacrofauna.columns = pd.MultiIndex.from_tuples(
        [(str(col[0]).replace(' ', ''), str(col[1]).replace(' ', '')) for col in self.benthicMacrofauna.columns]
        )
        self.benthicMacrofauna.index = self.benthicMacrofauna.index.str.replace(' ', '')
        # print(self.benthicMacrofauna.head())

        # Create DataFrames based on port selection
        selected_ports = ['Cape Town', 'Mossel Bay']
        atlanticPorts = df_cleaned[df_cleaned['Port'].isin(selected_ports)]
        indianPorts = df_cleaned[~df_cleaned['Port'].isin(selected_ports)]

        sediment_fauna, fauna = self._add_benthic_macrofauna(df_cleaned, self.benthicMacrofauna, "Nephtyidae")
        sediment_fauna = self._replace_less_than(sediment_fauna)
        # print("Spionidae: \n", spionidae.head())
        
        sediment, macrofauna = self._cleanSedimentMacrofaunaData('S', df_cleaned)

        atlanticSediment, atlanticMacrofauna = self._cleanSedimentMacrofaunaData('S', atlanticPorts)
        indianSediment, indianMacrofauna = self._cleanSedimentMacrofaunaData('S', indianPorts)

        """Next part wil be updated, this is only for now:
        """

        # tempSediment = sediment.loc[:, "Totalorganiccontent":"Zn"]
        tempSediment = sediment.replace(r'<.*', 0, regex=True)
        tempSediment = tempSediment.apply(pd.to_numeric, errors='coerce')
        tempSediment = tempSediment.drop(columns="Port", axis=1) #loc[:, "S":"d"] #

        tempMacrofauna = macrofauna.drop(columns="Port", axis=1) #loc[:, "S":"d"] #
        tempMacrofauna = tempMacrofauna.apply(pd.to_numeric, errors='coerce')


        X_train, X_test, y_train, y_test  = self._trainTestSplit(tempSediment, tempMacrofauna)

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self._normalise(X_train, X_test, y_train, y_test)

        SQI  = pd.DataFrame(df_cleaned["SQILowerlimit"])
        X_train_SQI, X_test_SQI, y_train_SQI, y_test_SQI = self._trainTestSplit(tempSediment, SQI)
        X_train_SQI, X_test_SQI, y_train_SQI, y_test_SQI = self._normalise(X_train_SQI, X_test_SQI, y_train_SQI, y_test_SQI)

        X_train_fauna, X_test_fauna, y_train_fauna, y_test_fauna = self._trainTestSplit(sediment_fauna, fauna)
        X_train_fauna, X_test_fauna, y_train_fauna, y_test_fauna = self._normalise(X_train_fauna, X_test_fauna, y_train_fauna, y_test_fauna)

        X_train_fauna.to_csv(os.path.join(self.savePath,'../preprocessed/X_train_fauna.csv'), index=False)
        y_train_fauna.to_csv(os.path.join(self.savePath,'../preprocessed/y_train_fauna.csv'), index=False)
        X_test_fauna.to_csv(os.path.join(self.savePath,'../preprocessed/X_test_fauna.csv'), index=False)
        y_test_fauna.to_csv(os.path.join(self.savePath,'../preprocessed/y_test_fauna.csv'), index=False)



        X_train.to_csv(os.path.join(self.savePath,'../preprocessed/X_train.csv'), index=False)
        y_train.to_csv(os.path.join(self.savePath,'../preprocessed/y_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.savePath,'../preprocessed/X_test.csv'), index=False)
        y_test.to_csv(os.path.join(self.savePath,'../preprocessed/y_test.csv'), index=False)

        X_train_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/X_train_scaled.csv'), index=False)
        y_train_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/y_train_scaled.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/X_test_scaled.csv'), index=False)
        y_test_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/y_test_scaled.csv'), index=False)
        

        # Save the two DataFrames to CSV files
        sediment.to_csv(os.path.join(self.savePath,'../preprocessed/sedimentData.csv'), index=False)
        macrofauna.to_csv(os.path.join(self.savePath,'../preprocessed/macrofaunaData.csv'), index=False)

        atlanticPorts.to_csv(os.path.join(self.savePath,'../preprocessed/atlanticData.csv'), index=False)
        indianPorts.to_csv(os.path.join(self.savePath,'../preprocessed/indianData.csv'), index=False)

        atlanticSediment.to_csv(os.path.join(self.savePath,'../preprocessed/atlanticSedimentData.csv'), index=False)
        atlanticMacrofauna.to_csv(os.path.join(self.savePath,'../preprocessed/atlanticMacrofaunaData.csv'), index=False)
        indianSediment.to_csv(os.path.join(self.savePath,'../preprocessed/indianSedimentData.csv'), index=False)
        indianMacrofauna.to_csv(os.path.join(self.savePath,'../preprocessed/indianMacrofaunaData.csv'), index=False)
        
        
        X_train_SQI.to_csv(os.path.join(self.savePath,'../preprocessed/X_train_SQI.csv'), index=False)
        X_test_SQI.to_csv(os.path.join(self.savePath,'../preprocessed/X_test_SQI.csv'), index=False)
        y_train_SQI.to_csv(os.path.join(self.savePath,'../preprocessed/y_train_SQI.csv'), index=False)
        y_test_SQI.to_csv(os.path.join(self.savePath,'../preprocessed/y_test_SQI.csv'), index=False)
        

    def _add_benthic_macrofauna(self, df, fauna_df, fauna):
        '''
        Selects the singular index of the specific benthic macrofauna you specified

        return: the specific benthic macrofauna as a column in the harbour sediment dataset
        '''

        # df = df.loc[:,"Year":"Zn"]
        

        faunaRow =fauna_df.index.get_loc(fauna)
        
        new = fauna_df.iloc[[0,faunaRow]]     

        new = new.T
        new = new.reset_index()       
    
        new.columns = ["Year", "Location", "Station", fauna]
        

        
        # print(df.head())
        
        new["Year"] = new["Year"].astype(float)
        new["Station"] = new["Station"].str.replace(r"^DBN(\d+)$", r"DB\1", regex=True)
        new = pd.merge(new, df, how="inner", left_on=["Year", "Station"], right_on=["Year", "Station(Newnumber)"])
        new[fauna] = new[fauna].fillna(value=0.0)
        # new = new[new['Spionidae'] >= 100]
        # print(new[fauna])
        # new[fauna] = pd.to_numeric(new[fauna], errors='raise')
        # new[fauna] = np.log(new[fauna])
        # print(new["Station"])
        new["Latitude"] = new["Latitude"].apply(self._dms_to_decimal)
        new["Longitude"] = new["Longitude"].apply(self._dms_to_decimal)
        # print(new.head())
        # new = self._replace_less_than(new)
        new.to_csv(os.path.join(self.savePath,"../preprocessed/fauna.csv"), index=False)
        new['geometry'] = new.apply(lambda row: Point((row['Longitude'], row['Latitude'])), axis=1)
        # print(new.head())
        
        gdf = gpd.GeoDataFrame(new, geometry='geometry')
        gdf.to_csv(os.path.join(self.savePath,"../preprocessed/GDF_fauna.csv"), index=False)

        # Set the coordinate reference system (CRS), e.g., WGS84 (EPSG:4326) for latitude/longitude
        gdf.set_crs(epsg=4326, inplace=True)


        new = new.dropna()
        new_fauna = pd.DataFrame(new[fauna])
        # new = new.drop(columns=["Location", "Latitude", "Longitude", "Year","Station(Newnumber)"])
        new = new.drop(columns=["Location", "Latitude", "Longitude", "Year","Station(Newnumber)", "Gravel","Verycoarsegrainedsand","Coarsegrainedsand","Mediumgrainedsand","Finegrainedsand","Veryfinegrainedsand","Mud","Meanphi","Meanmm","Medianphi","Medianmm","Sorting","Skewness"])

        new_df = new.loc[:, "Port":"Zn"]#"#metalsenriched"]
        new = new.loc[:, "Totalorganiccontent":"#metalsenriched"]
        
        
        encoder = OneHotEncoder(sparse_output=False)

        encoded_ports = encoder.fit_transform(new_df[['Port']])

        df_cleaned = new_df.dropna(subset=['Zn']).reset_index(drop=True)
        # print(encoded_ports)

        encoded_df = pd.DataFrame(encoded_ports, columns=encoder.get_feature_names_out(['Port']))
        encoded_df_reset = encoded_df.reset_index(drop=True)

        df_encoded = pd.concat([df_cleaned.drop(columns=['Port']), encoded_df_reset], axis=1)  
        print(df_encoded.tail())               

        # df_encoded = self._replace_less_than(df_encoded)
        
        return df_encoded, new_fauna

    def _replace_less_than(self, df):
        df = df.replace(r"<.*", 0, regex=True)
        return df.apply(pd.to_numeric, errors="coerce")

    def _cleanSedimentMacrofaunaData(self, split, df):
        if split in df.columns:
            port = df["Port"]
            index_s = df.columns.get_loc(f"{split}")
            
            # Split the DataFrame into two parts
            df1_temp = df.iloc[:, :index_s]  # All columns up to "S"
            df2 = pd.DataFrame(df["S"])#.iloc[:, index_s :]  # All columns from "S" onwards
            df1 = df1_temp.loc[:, "Sand":"SQILowerlimit"]
            df1 = df1.replace(r"<.*", 0, regex=True)
            df1 = df1.apply(pd.to_numeric, errors="coerce")

            # selected_ports = ['Cape Town', 'Mossel Bay']
            # atlanticPorts = df_cleaned[df_cleaned['Port'].isin(selected_ports)]
            # indianPorts = df_cleaned[~df_cleaned['Port'].isin(selected_ports)]


            if len(port) == len(df2):
                df1.insert(0, "Port", port.values)
                df2.insert(0, "Port", port.values)
                print("Location column added to DataFrames.")
            else:
                print("The length of Location column does not match the macrofauna DataFrame.")
            
        else:
            print(f"Column {split} not found in the DataFrame.")

        return df1, df2
    
    def _trainTestSplit(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _normalise(self, X_train, X_test, y_train, y_test):

        scaler = MinMaxScaler()

        # # Fit the scaler on the training data and transform the training data
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        # Transform the test data using the same scaler (without refitting)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        target_scaler = MinMaxScaler()
        y_train_scaled = pd.DataFrame(target_scaler.fit_transform(y_train), columns=y_train.columns)
        y_test_scaled = pd.DataFrame(target_scaler.transform(y_test), columns=y_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def _concatDF(self, df1, df2):
        return pd.concat([df1,df2], axis=1)
    
    def _dms_to_decimal(self, dms):
        """Convert DMS to decimal degrees."""
        dms = dms.replace('"', '').strip()
        # print("DMS: ", dms)  # Remove double quotes and trim spaces
        direction = dms[-1:]
        # print("Direction: ",direction)
        degrees, minutes_seconds = dms[:-1].split("°")
        # print("Minutes & Seconds: ", minutes_seconds)
        minutes, seconds = minutes_seconds.split("'")
        # seconds = seconds_direction[:-1]
        # direction = seconds_direction[-1]

        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        
        # Adjust for southern or western hemispheres
        if direction in ['S', 'W']:
            decimal *= -1
        # print(decimal)
        return decimal