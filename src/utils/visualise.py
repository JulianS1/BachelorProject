import os
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, gamma, beta, norm, weibull_min



class Visualisation:
    def __init__(self, savePath) -> None:

        self.savePath = savePath

        self.fauna = pd.read_csv("../../data/preprocessed/fauna.csv",
                sep=",",
                encoding="utf-8")
        
        self.X_train = pd.read_csv("../../data/preprocessed/X_train_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.y_train = pd.read_csv("../../data/preprocessed/y_train_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.X_test = pd.read_csv("../../data/preprocessed/X_test_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.y_test = pd.read_csv("../../data/preprocessed/y_test_fauna.csv",
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
        
    def data_plotting(self):

        for column in self.y_train:
            plt.hist(self.y_train[column], bins=300, color='skyblue', edgecolor='black')
            plt.xticks(rotation=90)
            plt.title(column)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            ser = self.y_train[column]

            xt = plt.xticks()[0]  
            xmin, xmax = min(xt), max(xt)  
            lnspc = np.linspace(xmin, xmax, len(ser))

            m, s = norm.fit(ser) 
            pdf_g = norm.pdf(lnspc, m, s)
            plt.plot(lnspc, pdf_g, label="Norm")

            ag,bg,cg = gamma.fit(ser)  
            pdf_gamma = gamma.pdf(lnspc, ag, bg,cg)  
            plt.plot(lnspc, pdf_gamma, label="Gamma")

            ab,bb,cb,db = beta.fit(ser)  
            pdf_beta = beta.pdf(lnspc, ab, bb,cb, db)  
            plt.plot(lnspc, pdf_beta, label="Beta")

            # shape, loc, scale = weibull_min.fit(self.X_train)
            # pdf_weibull = weibull_min.pdf(lnspc, shape, loc=loc, scale=scale)
            # plt.plot(lnspc, pdf_weibull, label="Weibull")

            plt.legend(loc='best', frameon=False)
            plt.show()

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

    def faunacorrelation(self):
        faunaData = self.fauna.drop(columns=["Station", "Port","Location", "Latitude", "Longitude", "Year","Station(Newnumber)","Verycoarsegrainedsand","Coarsegrainedsand","Mediumgrainedsand","Finegrainedsand","Veryfinegrainedsand","Meanphi","Meanmm","Medianphi","Medianmm","Sorting","Skewness","Al","Fe","As","Ba","Be","Cd","Co","Cu","Cr","Mn","Hg","Ni","Pb","V","Zn"])
        faunaData = faunaData.loc[:, "Gravel":"BAPTEQ"]
        faunaData = faunaData.replace(r'<.*', 0, regex=True)
        print(faunaData)
        corrmatrix = faunaData.corr()
        plt.figure(figsize=(300, 300))
        sns.heatmap(corrmatrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Nephtyidae and predictor variables")

        # save the correlation matrix plot
        file_name = "Nephtyidae_correlation_matrix.png"
        plt.tight_layout()
        
        plt.savefig(
            os.path.join("../../results", file_name)
        )
    
    def _compare_harbours(self):
        self.fauna["residual"] = m1.u
        # Obtain the median value of residuals in each neighborhood
        medians = (
        self.fauna.groupby("neighborhood")
        .residual.median()
        .to_frame("hood_residual")
        )

        # Increase fontsize
        sns.set(font_scale=1.25)
        # Set up figure
        f = plt.figure(figsize=(15, 3))
        # Grab figure's axis
        ax = plt.gca()
        # Generate bloxplot of values by neighborhood
        # Note the data includes the median values merged on-the-fly
        sns.boxplot(
        x="neighborhood",
        y="residual",
        ax=ax,
        data=self.fauna.merge(
                medians, how="left", left_on="neighborhood", right_index=True
        ).sort_values("hood_residual"),
        palette="bwr",
        )
        # Rotate the X labels for legibility
        f.autofmt_xdate(rotation=-90)
        # Display
        plt.show()
        
    def _cluster_analysis(self):

        # self.fauna['station'] = pd.factorize(self.fauna['tation'])[0]
        features = self.fauna[["Spionidae"]]
        # stations = self.fauna["Station(Newnumber)"]
        num_clusters = 72  #72  # Change this as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        # Fit the model
        kmeans.fit(features)

        # Add the cluster labels to the original DataFrame
        self.fauna['cluster'] = kmeans.labels_


        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.fauna, x='Spionidae', y='cluster', hue='Port', palette='viridis', s=100, alpha=0.7)

        # Adding titles and labels
        plt.title('K-Means Clustering of Spionidae by Port', fontsize=16)
        plt.xlabel('Spionidae Values', fontsize=14)
        plt.ylabel('Cluster', fontsize=14)
        plt.legend(title='Port', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()
        plt.show()

        # Analyze cluster characteristics
        # cluster_summary = self.fauna.groupby('cluster').mean()
        # print(cluster_summary)

        contingency_table = pd.crosstab(self.fauna['Spionidae'], self.fauna['Port'])

        # print("Contingency Table:")
        # print(contingency_table)

        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        print(f"Chi-Squared Statistic: {chi2_stat}")
        print(f"P-Value: {p_value}")
        print(f"Degrees of Freedom: {dof}")
        print("Expected Frequencies Table:")
        print(expected)

        alpha = 0.05  # Significance level
        if p_value < alpha:
            print("Reject the null hypothesis: There is a significant correlation between Port and Spionidae.")
        else:
            print("Fail to reject the null hypothesis: No significant correlation between Port and Spionidae.")
