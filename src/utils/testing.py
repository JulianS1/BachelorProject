import os
import sklearn as sk
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt


class Tester:

    def __init__(self, save_path) -> None:
        self.data = pd.read_csv("../../data/preprocessed/fauna.csv",
                sep=",",
                encoding="utf-8")
        self.output_directory = save_path
        
    def _anova(self):
        df = self.data
        df['Port'] = df['Port'].astype('category')
        df['Station'] = df['Station'].astype('category')
        model = ols('Spionidae ~ Port * Station', data=df).fit()

        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)

        # print(model.summary())

    def _ols(self):
        df = self.data
        predictors = ['Totalorganiccontent', 'Sand', 'Al', 'Fe', 'As', 'Ba', 'Be', 'Cd', 'Co', 'Cu', 'Cr', 'Mn', 'Hg', 'Ni', 'Pb', 'V', 'Zn']
        formula = 'Spionidae ~ ' + ' + '.join(predictors)
        model = ols(formula, data=df).fit()

        
        print(model.summary())

    def _pearson(self):
        df = self.data
        columns_of_interest = ['Spionidae', 'Totalorganiccontent', 'Sand', 'Al', 'Fe', 'As', 'Ba', 'Be', 'Cd', 'Co', 'Cu', 'Cr', 'Mn', 'Hg', 'Ni', 'Pb', 'V', 'Zn']
        correlation_matrix = df[columns_of_interest].corr(method='pearson')
        print(correlation_matrix)
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Spionidae and Environmental Variables')
        plt.savefig(os.path.join(self.output_directory, "pearson_corr_heatmap"), bbox_inches='tight') 
        plt.show()