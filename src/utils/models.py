import os
import sklearn as sk
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np


class Model:
    def __init__(self, path) -> None:
        self.path = path

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
        
        


    def linearModel(self):
        X_train, X_test, y_train, y_test = train_test_split(self.sediment, self.macrofauna, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Step 2: Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print evaluation metrics
        print("\n Random Forest regression")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")
        
        # Step 4: Use SHAP to explain the model's predictions
        # Initialize the SHAP explainer
        explainer = shap.Explainer(model, X_train)

        # Calculate SHAP values
        shap_values = explainer(X_test)

        print("X_test columns", X_test.columns)
        print("shap_values Shape", shap_values.shape)  # Should match the number of rows and features
        print("X_test shape", X_test.shape)


        shap_values_single_output = shap_values[..., 0]  # or choose any output index you want (0 to 6)

        # Now plot the bar chart for the SHAP values of this specific output
        shap.plots.bar(shap_values_single_output)

        # shap.plots.bar(shap_values)

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

    def randomForest(self):
        X_train, X_test, y_train, y_test = train_test_split(self.sediment, self.macrofauna, test_size=0.2, random_state=42)
        
        # scaler = MinMaxScaler()

        # # Fit the scaler on the training data and transform the training data
        # X_train = scaler.fit_transform(X_train)

        # # Transform the test data using the same scaler (without refitting)
        # X_test = scaler.transform(X_test)

        # target_scaler = MinMaxScaler()
        # y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))  # Reshape if y is a 1D array
        # y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
        # print(y_train_scaled)

        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        print("\n Random Forest regressor")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")
        
        # Explainer object
        explainer = shap.Explainer(model, X_train)

        # Calculate SHAP values for X_test
        shap_values = explainer(X_test)

        # Print shapes to confirm
        print("Shape of shap_values:", shap_values.shape)
        print("Shape of X_test:", X_test.shape)

        # Convert SHAP values object to array
        shap_values_array = shap_values.values  # This should have shape (n_samples, n_features)

        # Print shape of the shap_values_array
        print("Shape of shap_values_array:", shap_values_array.shape)

        # Generate SHAP dependence plot
        shap.dependence_plot("Totalorganiccontent", shap_values_array, X_test, interaction_index="Zn")



    def GBoostRegressor(self):
        X_train, X_test, y_train, y_test = train_test_split(self.sediment, self.macrofauna, test_size=0.2, random_state=42)
        
        scaler = MinMaxScaler()

        # # Fit the scaler on the training data and transform the training data
        # X_train = scaler.fit_transform(X_train)

        # # Transform the test data using the same scaler (without refitting)
        # X_test = scaler.transform(X_test)

        # target_scaler = MinMaxScaler()
        # y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))  # Reshape if y is a 1D array
        # y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
        # print(y_train_scaled)

        
        base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
        # Wrap the model for multi-output regression
        model = MultiOutputRegressor(base_model)
        
        # print(f"Shape of X_train: {X_train.shape}")
        # print(f"Shape of y_train: {y_train.shape}")
        


        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\n Gradient Boosting regression")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")

        shap_values_list = []
    
    # Calculate SHAP values for each output
        for i, est in enumerate(model.estimators_):
            print(f"Explaining output {i + 1}")
            
            # Create SHAP explainer using TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(est)
            
            # Compute SHAP values for the test data
            shap_values = explainer.shap_values(X_test)
            
            # Append SHAP values to the list
            shap_values_list.append(shap_values)
            
            # Optionally, plot the summary for each output
            shap.summary_plot(shap_values, X_test, show=False, feature_names=X_train.columns)
            plt.title(f'SHAP Summary Plot for Output {i+1}')
            plt.show()
        
        # Show force plot for the first prediction of the first output (as an example)
        shap.force_plot(explainer.expected_value, shap_values_list[0][0], X_test.iloc[0, :], matplotlib=True)


    def NN(self):

        X_train, X_test, y_train, y_test = train_test_split(self.sediment, self.macrofauna, test_size=0.2, random_state=42)

        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPRegressor(hidden_layer_sizes=(50,25), activation="tanh",solver="adam" , max_iter=500, random_state=42)

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.2f}')

        # param_grid = {
        #     'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['adam', 'sgd'],
        # }

        # grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=3)
        # grid_search.fit(X_train_scaled, y_train)

        # print("Best parameters:", grid_search.best_params_)

    
    
