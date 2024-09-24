import os
import sklearn as sk
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


class Model:
    def __init__(self, path) -> None:
        self.path = path
        self.sediment = pd.read_csv("../../data/preprocessed/sedimentData.csv",
                sep=",",
                encoding="utf-8")
        self.macrofauna = pd.read_csv("../../data/preprocessed/macrofaunaData.csv",
                sep=",",
                encoding="utf-8")
        self.sediment = self.sediment.loc[:, "Gravel":"Zn"]
        self.sediment = self.sediment.replace(r'<.*', 0, regex=True)
        self.sediment = self.sediment.apply(pd.to_numeric, errors='coerce')

        self.macrofauna = self.macrofauna.drop(columns="Port", axis=1)
        self.macrofauna = self.macrofauna.apply(pd.to_numeric, errors='coerce')


    def linearModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.sediment, self.macrofauna, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 4: Use SHAP to explain the model's predictions
        # Initialize the SHAP explainer
        explainer = shap.Explainer(model, X_train)

        # Calculate SHAP values
        shap_values = explainer(X_test)

        # Step 5: Plot the SHAP values
        # Summary plot

        print("Shape of shap_values:", shap_values.shape)
        print("Shape of X_test:", X_test.shape)
        shap_values_array = shap_values.values

        # Take the mean SHAP values across the last dimension (outputs)
        shap_values_mean = np.mean(shap_values_array, axis=2)  # Average across outputs

        # Print the shape to confirm
        print("Shape of mean shap_values:", shap_values_mean.shape)

        # Generate summary plot for mean SHAP values
        shap.summary_plot(shap_values_mean, X_test)


        # shap.summary_plot(shap_values, X_test)

        # Optional: Force plot for a specific prediction (you can specify an index)
        shap.initjs()  # Initialize JS visualizations in Jupyter Notebooks
        shap.force_plot(explainer.expected_value, shap_values_array[0], X_test.iloc[0])