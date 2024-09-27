import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualisation:
    def __init__(self, savePath) -> None:

        self.savePath = savePath

        
        self.X_train = pd.read_csv("../../data/preprocessed/X_train.csv",
                sep=",",
                encoding="utf-8")
        self.y_train = pd.read_csv("../../data/preprocessed/y_train.csv",
                sep=",",
                encoding="utf-8")
        self.X_test = pd.read_csv("../../data/preprocessed/X_test.csv",
                sep=",",
                encoding="utf-8")
        self.y_test = pd.read_csv("../../data/preprocessed/y_test.csv",
                sep=",",
                encoding="utf-8")
        
        self.X_train_scaled = pd.read_csv("../../data/preprocessed/X_train_scaled.csv",
                sep=",",
                encoding="utf-8")
        self.y_train_scaled = pd.read_csv("../../data/preprocessed/y_train_scaled.csv",
                sep=",",
                encoding="utf-8")
        self.X_test_scaled = pd.read_csv("../../data/preprocessed/X_test_scaled.csv",
                sep=",",
                encoding="utf-8")
        self.y_test_scaled = pd.read_csv("../../data/preprocessed/y_test_scaled.csv",
                sep=",",
                encoding="utf-8")
        
        self.atlanticSediment = pd.read_csv("../../data/partPreprocessed/atlanticSedimentData.csv",
                sep=",",
                encoding="utf-8")
        self.atlanticMacrofauna = pd.read_csv("../../data/partPreprocessed/atlanticMacrofaunaData.csv",
                sep=",",
                encoding="utf-8")

        self.indianSediment = pd.read_csv("../../data/partPreprocessed/indianSedimentData.csv",
                sep=",",
                encoding="utf-8")
        self.indianMacrofauna = pd.read_csv("../../data/partPreprocessed/indianMacrofaunaData.csv",
                sep=",",
                encoding="utf-8")

    def correlationAnalysis(self):

        atlanticValues = self.atlanticSediment.loc[:, "Al":"Zn"]
        atlanticValues.insert(0, "Port", self.atlanticSediment["Port"].values)
        # print(allValues.columns)

        # atlanticData = pd.merge(atlanticValues, atlanticMacrofauna, on="Port", how="inner")
        
        
        #*********Needs to be changed properly
        atlanticData = atlanticData.replace(r'<.*', 0, regex=True)
        # atlanticData = atlanticData.drop(atlanticData.columns[0], axis=1)
        atlanticData.to_csv("../../data/preprocessed/atlanticData.csv")

        

        atlanticData = atlanticData.drop(columns="Port", axis=1)
        atlanticData = atlanticData.loc[:, ~atlanticData.columns.str.contains('^Unnamed')]
        atlanticData.columns = atlanticData.columns.str.replace(' ', '', regex=True)

        corr_matrix = atlanticData.corr()

        

        # create the heatmap plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Sediment variables and bioindicators")

        # save the correlation matrix plot
        file_name = "atlantic_correlation_matrix.png"
        plt.tight_layout()
        
        plt.savefig(
            os.path.join("../../results", file_name)
        )





        indianValues = self.indianSediment.loc[:, "Al":"Zn"]
        indianValues.insert(0, "Port", self.indianSediment["Port"].values)
        # print(allValues.columns)

        # indianData = pd.merge(indianValues, indianMacrofauna, on="Port", how="inner")
        
        
        #*********Needs to be changed properly
        indianData = indianData.replace(r'<.*', 0, regex=True)
        # indianData = indianData.drop(indianData.columns[0], axis=1)
        indianData.to_csv("../../data/preprocessed/indianData.csv")

        

        indianData = indianData.drop(columns="Port", axis=1)
        indianData = indianData.loc[:, ~indianData.columns.str.contains('^Unnamed')]
        indianData.columns = indianData.columns.str.replace(' ', '', regex=True)

        corr_matrix = indianData.corr()

        

        # create the heatmap plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Sediment variables and bioindicators")

        # save the correlation matrix plot
        file_name = "indian_correlation_matrix.png"
        plt.tight_layout()
        
        plt.savefig(
            os.path.join("../../results", file_name)
        )
        # plt.show()

        