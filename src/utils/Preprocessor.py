import os
import re
import pandas as pd
from datetime import datetime
import openpyxl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split



class Preprocessor():
    def __init__(self, sedimentPath, savePath) -> None:
        self.sedimentPath = sedimentPath
        self.savePath = savePath
        
    def load_dataset(self):
        df = pd.read_excel(self.sedimentPath, skiprows=2)
        print("loaded dataset")
        # df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        # df_cleaned = df.dropna(axis=1, how='all')
        df_cleaned = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df_cleaned.columns = df_cleaned.columns.str.replace(' ', '', regex=True)
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        df_cleaned = df_cleaned.dropna(subset=['S'])
        df_cleaned = df_cleaned.drop(columns="J'")
        
        # Create DataFrames based on port selection
        selected_ports = ['Cape Town', 'Mossel Bay']
        atlanticPorts = df_cleaned[df_cleaned['Port'].isin(selected_ports)]
        indianPorts = df_cleaned[~df_cleaned['Port'].isin(selected_ports)]

        print(len(atlanticPorts))
        print(len(indianPorts))

        print(atlanticPorts.head())
        # print("Remaining columns after cleaning:")
        # print(df_cleaned.columns.tolist())
        sediment, macrofauna = self.cleanSedimentData('S', df_cleaned)

        atlanticSediment, atlanticMacrofauna = self.cleanSedimentData('S', atlanticPorts)
        indianSediment, indianMacrofauna = self.cleanSedimentData('S', indianPorts)

        """Next part wil be updated but is only for now
        """

        tempSediment = sediment.loc[:, "Totalorganiccontent":"Zn"]
        tempSediment = tempSediment.replace(r'<.*', 0, regex=True)
        tempSediment = tempSediment.apply(pd.to_numeric, errors='coerce')

        tempMacrofauna = macrofauna.drop(columns="Port", axis=1) #loc[:, "S":"d"] #
        tempMacrofauna = tempMacrofauna.apply(pd.to_numeric, errors='coerce')




        X_train, X_test, y_train, y_test  = self.trainTestSplit(tempSediment, tempMacrofauna)

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.normalise(X_train, X_test, y_train, y_test)

        X_train.to_csv(os.path.join(self.savePath,'../preprocessed/X_train.csv'), index=False)
        y_train.to_csv(os.path.join(self.savePath,'../preprocessed/y_train.csv'), index=False)
        X_test.to_csv(os.path.join(self.savePath,'../preprocessed/X_test.csv'), index=False)
        y_test.to_csv(os.path.join(self.savePath,'../preprocessed/y_test.csv'), index=False)

        X_train_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/X_train_scaled.csv'), index=False)
        y_train_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/y_train_scaled.csv'), index=False)
        X_test_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/X_test_scaled.csv'), index=False)
        y_test_scaled.to_csv(os.path.join(self.savePath,'../preprocessed/y_test_scaled.csv'), index=False)

        # Save the two DataFrames to CSV files
        sediment.to_csv(os.path.join(self.savePath,'sedimentData.csv'), index=False)
        macrofauna.to_csv(os.path.join(self.savePath,'macrofaunaData.csv'), index=False)

        atlanticPorts.to_csv(os.path.join(self.savePath,'atlanticData.csv'), index=False)
        indianPorts.to_csv(os.path.join(self.savePath,'indianData.csv'), index=False)

        atlanticSediment.to_csv(os.path.join(self.savePath,'atlanticSedimentData.csv'), index=False)
        atlanticMacrofauna.to_csv(os.path.join(self.savePath,'atlanticMacrofaunaData.csv'), index=False)
        indianSediment.to_csv(os.path.join(self.savePath,'indianSedimentData.csv'), index=False)
        indianMacrofauna.to_csv(os.path.join(self.savePath,'indianMacrofaunaData.csv'), index=False)
        
        
        

        





    def cleanSedimentData(self, split, df):
        if split in df.columns:
            port = df["Port"]
            index_s = df.columns.get_loc(f"{split}")
            
            # Split the DataFrame into two parts
            df1 = df.iloc[:, :index_s]  # All columns up to 'S'
            df2 = df.iloc[:, index_s :]  # All columns from 'S' onwards
            if len(port) == len(df2):
                df2.insert(0, "Port", port.values)
                print("Location column added to macrofauna DataFrame.")
            else:
                print("The length of Location column does not match the macrofauna DataFrame.")
            
        else:
            print(f"Column {split} not found in the DataFrame.")

        return df1, df2
    
    def trainTestSplit(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def normalise(self, X_train, X_test, y_train, y_test):

        scaler = MinMaxScaler()

        # # Fit the scaler on the training data and transform the training data
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

        # Transform the test data using the same scaler (without refitting)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        target_scaler = MinMaxScaler()
        y_train_scaled = pd.DataFrame(target_scaler.fit_transform(y_train), columns=y_train.columns)  # Reshape if y is a 1D array
        y_test_scaled = pd.DataFrame(target_scaler.transform(y_test), columns=y_test.columns)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled