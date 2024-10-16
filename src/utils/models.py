import os
import sklearn as sk
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import numpy as np


'''
TODO:
Extremely random trees
NN with high bias
Bayesian model (heirarchical)
'''


class Model:
    def __init__(self, path) -> None:
        self.path = path

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

        # self.X_train_scaled = pd.read_csv("../../data/preprocessed/X_train_SQI.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.y_train_scaled = pd.read_csv("../../data/preprocessed/y_train_SQI.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.X_test_scaled = pd.read_csv("../../data/preprocessed/X_test_SQI.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.y_test_scaled = pd.read_csv("../../data/preprocessed/y_test_SQI.csv",
        #         sep=",",
        #         encoding="utf-8")
        
        # self.X_train_scaled = pd.read_csv("../../data/preprocessed/X_train_scaled.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.y_train_scaled = pd.read_csv("../../data/preprocessed/y_train_scaled.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.X_test_scaled = pd.read_csv("../../data/preprocessed/X_test_scaled.csv",
        #         sep=",",
        #         encoding="utf-8")
        # self.y_test_scaled = pd.read_csv("../../data/preprocessed/y_test_scaled.csv",
        #         sep=",",
        #         encoding="utf-8")

        self.X_train_scaled = pd.read_csv("../../data/preprocessed/X_train_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.y_train_scaled = pd.read_csv("../../data/preprocessed/y_train_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.X_test_scaled = pd.read_csv("../../data/preprocessed/X_test_fauna.csv",
                sep=",",
                encoding="utf-8")
        self.y_test_scaled = pd.read_csv("../../data/preprocessed/y_test_fauna.csv",
                sep=",",
                encoding="utf-8")
        
        


    def linearModel(self):
        model = LinearRegression()
        # self.y_train_scaled = self.y_train_scaled['S']
        # self.y_test_scaled = self.y_test_scaled['S']
        model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = model.predict(self.X_test_scaled)
        
        # Step 2: Calculate evaluation metrics
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)
        
        # Print evaluation metrics
        print("\n Linear Regression")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")
        
        # Step 4: Use SHAP to explain the model's predictions
        # Initialize the SHAP explainer
        explainer = shap.Explainer(model, self.X_train_scaled)

        # Calculate SHAP values
        shap_values = explainer(self.X_test_scaled)

        print("X_test columns", self.X_test_scaled.columns)
        print("shap_values Shape", shap_values.shape)
        print("X_test shape", self.X_test_scaled.shape)


        shap_values_single_output = shap_values[..., 0]

        shap.plots.bar(shap_values_single_output)
        file_name = "LinearRegression_SHAP"

        output_directory = os.path.join("..", "..", "results", "SHAP")
        os.makedirs(output_directory, exist_ok=True)
        plt.draw()
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, file_name), bbox_inches='tight') 
        plt.close()

        # shap.plots.bar(shap_values)

        # Step 5: Plot the SHAP values
        # Summary plot

        # print("Shape of shap_values:", shap_values.shape)
        # print("Shape of X_test:", self.X_test.shape)
        # shap_values_array = shap_values.values

        # # Take the mean SHAP values across the last dimension (outputs)
        # shap_values_mean = np.mean(shap_values_array, axis=2)  # Average across outputs

        # # Print the shape to confirm
        # print("Shape of mean shap_values:", shap_values_mean.shape)

        # Generate summary plot for mean SHAP values
        # shap.summary_plot(shap_values_mean, self.X_test)
        # plt.close()

        

        

        # shap.summary_plot(shap_values, X_test)

        # Optional: Force plot for a specific prediction (you can specify an index)
        # shap.initjs()  # Initialize JS visualizations in Jupyter Notebooks
        # shap.force_plot(explainer.expected_value, shap_values_array[0], self.X_test.iloc[0])

    def ER_Trees(self):
        model = RandomForestRegressor(n_estimators=80, random_state=42)
        model.fit(self.X_train_scaled, self.y_train_scaled)

        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)


        print("\n Extra Random Trees regressor")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")

    def randomForest(self):

        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train_scaled)

        # Make predictions
        y_pred = model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)


        print("\n Random Forest regressor")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")
        
        # Explainer object
        explainer = shap.TreeExplainer(model, self.X_train_scaled)

        # Calculate SHAP values for X_test
        shap_values = explainer(self.X_test_scaled)

        # Print shapes to confirm
        print("Shape of shap_values:", shap_values.shape)
        print("Shape of X_test:", self.X_test_scaled.shape)

        # Convert SHAP values object to array
        shap_values_array = shap_values.values  # This should have shape (n_samples, n_features)

        # Print shape of the shap_values_array
        print("Shape of shap_values_array:", shap_values_array[0].shape)

        # # Generate SHAP dependence plot
        shap_values_list = []


        
        # shap.dependence_plot("Totalorganiccontent", shap_values_array[:,:,0], self.X_test, interaction_index="Zn")
        # shap_values_list = [[] for _ in range(7)]  # Assuming 7 outputs, one list for each output

        for i, est in enumerate(model.estimators_):
            print(f"Explaining output {i + 1}")
            
            # Create SHAP explainer using TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(est)
            
            # Compute SHAP values for the test data
            shap_values = explainer.shap_values(self.X_test_scaled)
            print("num outputs: ", self.y_test_scaled.shape[1])
            # Append SHAP values to the list
            shap_values_list.append(shap_values[i])
            if self.y_test_scaled.shape[1] > 1:
                column_name = self.y_test_scaled.columns[i]
            else:
                column_name = self.y_test_scaled.columns[0]
            file_name = "RF_SHAP_" + column_name
            # Optionally, plot the summary for each output
            shap.summary_plot(shap_values, self.X_test_scaled, show=False, feature_names=self.X_train_scaled.columns)
            plt.title(f'SHAP Summary Plot for Output {self.y_test_scaled.columns[i]}')
            # plt.show()
            plt.savefig(
            os.path.join("../../results/SHAP", file_name)
            )
            plt.close()

        # # Loop over each estimator in the RandomForest model
        # for est in model.estimators_:
        #     # Create SHAP explainer using TreeExplainer for tree-based models
        #     explainer = shap.TreeExplainer(est)

        #     # Compute SHAP values for the test data (for all outputs)
        #     shap_values = explainer.shap_values(self.X_test)  # returns a list of SHAP values, one for each output

        #     # Append SHAP values for each output to the corresponding list
        #     for output_idx in range(7):  # Assuming 7 outputs
        #         shap_values_list[output_idx].append(shap_values[output_idx])

        # # Now, for each output, we need to average or sum the SHAP values across all estimators
        # mean_shap_values_list = []

        # for output_idx in range(7):  # Loop over the outputs
        #     # Average SHAP values across all estimators for the current output
        #     mean_shap_values = sum(shap_values_list[output_idx]) / len(model.estimators_)
        #     mean_shap_values_list.append(mean_shap_values)

        #     # Debugging step: Print the shapes of mean_shap_values and X_test
        #     print(f"Output {output_idx + 1} - mean_shap_values shape: {mean_shap_values.shape}")
        #     print(f"X_test shape: {self.X_test.shape}")

        #     # Check if the shapes match
        #     assert mean_shap_values.shape[0] == self.X_test.shape[0], "Mismatch in the number of samples!"
        #     assert mean_shap_values.shape[1] == self.X_test.shape[1], "Mismatch in the number of features!"

        #     # Plot SHAP summary for the current output
        #     file_name = f"RF_SHAP_Output_{output_idx + 1}.png"
            
        #     shap.summary_plot(mean_shap_values, self.X_test, feature_names=self.X_train.columns, show=False)
        #     plt.title(f'SHAP Summary Plot for Output {output_idx + 1}')
        #     plt.savefig(os.path.join("../../results/SHAP", file_name))
        #     plt.close()  



    def GBoostRegressor(self):
                
        base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
        # Wrap the model for multi-output regression
        model = MultiOutputRegressor(base_model)
        
        # print(f"Shape of X_train: {X_train.shape}")
        # print(f"Shape of y_train: {y_train.shape}")
        


        model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = model.predict(self.X_test_scaled)
        
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)

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
            shap_values = explainer.shap_values(self.X_test_scaled)
            
            # Append SHAP values to the list
            shap_values_list.append(shap_values[i])
            file_name = "GBoost_SHAP_" + self.y_test_scaled.columns[i]
            
            # Optionally, plot the summary for each output
            shap.summary_plot(shap_values, self.X_test_scaled, show=False, feature_names=self.X_train_scaled.columns)
            plt.title(f'SHAP Summary Plot for Output {self.y_test_scaled.columns[i]}')
            # plt.show()
            plt.savefig(
            os.path.join("../../results/SHAP", file_name)
            )
            plt.close()
        
        # Show force plot for the first prediction of the first output (as an example)
        # shap.force_plot(explainer.expected_value, shap_values_list[0][0], self.X_test.iloc[0, :], matplotlib=True)


    def NN(self):

        model = MLPRegressor(hidden_layer_sizes=(50,25), activation="tanh",solver="adam" , max_iter=500, random_state=42)

        # Train the model
        model.fit(self.X_train_scaled, self.y_train_scaled)

        # Make predictions
        y_pred = model.predict(self.X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)

        print("\n MLP regression")
        print(f'Mean Squared Error: {mae:.2f}')
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")

        # param_grid = {
        #     'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['adam', 'sgd'],
        # }

        # grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=3)
        # grid_search.fit(X_train_scaled, y_train)

        # print("Best parameters:", grid_search.best_params_)

    
    
