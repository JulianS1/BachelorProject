import os
import re
import pandas as pd
from datetime import datetime
import openpyxl

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

        print(df_cleaned.head())
        # print("Remaining columns after cleaning:")
        # print(df_cleaned.columns.tolist())
        sediment, macrofauna = self.cleanSedimentData('S', df_cleaned)

        atlanticSediment, atlanticMacrofauna = self.cleanSedimentData('S', atlanticPorts)
        indianSediment, indianMacrofauna = self.cleanSedimentData('S', indianPorts)


        # Save the two DataFrames to CSV files
        sediment.to_csv(os.path.join(self.savePath,'sedimentData.csv'), index=False)
        macrofauna.to_csv(os.path.join(self.savePath,'macrofaunaData.csv'), index=False)

        atlanticSediment.to_csv(os.path.join(self.savePath,'atlanticSedimentData.csv'), index=False)
        atlanticMacrofauna.to_csv(os.path.join(self.savePath,'atlanticMacrofaunaData.csv'), index=False)
        indianSediment.to_csv(os.path.join(self.savePath,'indianSedimentData.csv'), index=False)
        indianMacrofauna.to_csv(os.path.join(self.savePath,'indianMacrofaunaData.csv'), index=False)
        
        #     print("DataFrames split and saved to 'data_part1.csv' and 'data_part2.csv'")
        # else:
        #     print("Column 'S' not found in the DataFrame.")

        

        





    def cleanSedimentData(self, split, df):
        if 'S' in df.columns:
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