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
        

        print(df_cleaned.head())
        # print("Remaining columns after cleaning:")
        # print(df_cleaned.columns.tolist())

        if 'S' in df_cleaned.columns:
            index_s = df_cleaned.columns.get_loc('S')
            
            # Split the DataFrame into two parts
            sediment = df_cleaned.iloc[:, :index_s]  # All columns up to 'S'
            macrofauna = df_cleaned.iloc[:, index_s :]  # All columns from 'S' onwards
            
            # Save the two DataFrames to CSV files
            sediment.to_csv(os.path.join(self.savePath,'sedimentData.csv'), index=False)
            macrofauna.to_csv(os.path.join(self.savePath,'macrofauna.csv'), index=False)
            
            print("DataFrames split and saved to 'data_part1.csv' and 'data_part2.csv'")
        else:
            print("Column 'S' not found in the DataFrame.")



    def cleanSedimentData():

        pass