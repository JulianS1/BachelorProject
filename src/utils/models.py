import os
import sklearn as sk
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, MultiTaskLassoCV, RidgeCV, MultiTaskElasticNetCV, BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold

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

        # lasso_cv = RidgeCV(cv=5)  # 5-fold cross-validation
        # lasso_cv.fit(self.X_train_scaled, self.y_train_scaled)
        # lasso_best = Lasso(alpha=lasso_cv.alpha_)
        # lasso_best.fit(self.X_train_scaled, self.y_train_scaled)
        # y_pred = lasso_best.predict(self.X_test_scaled)

        

        model = LinearRegression()
        # self.y_train_scaled = self.y_train_scaled['S']
        # self.y_test_scaled = self.y_test_scaled['S']
        
        model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = model.predict(self.X_test_scaled)
        

        # param_grid = {
        #     'alpha_1': [1e-6, 1e-5, 1e-4],
        #     'alpha_2': [1e-6, 1e-5, 1e-4],
        #     'lambda_1': [1e-6, 1e-5, 1e-4],
        #     'lambda_2': [1e-6, 1e-5, 1e-4]
        # }

        # bayes = BayesianRidge()
        # bayes.fit(self.X_train_scaled, self.y_train_scaled)
        # y_pred = bayes.predict(self.X_test_scaled)

        # cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # grid_search = GridSearchCV(bayes, param_grid, cv=cv, scoring='neg_mean_squared_error')

        # # Fit the grid search
        # grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        # # Get the best estimator and parameters
        # best_bayes = grid_search.best_estimator_
        # best_params = grid_search.best_params_

        # print("Best Parameters:", best_params)
        # y_pred = best_bayes.predict(self.X_test_scaled)

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
        y_pred = model.predict(self.X_test_scaled)

        rf = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        # grid_search = GridSearchCV(
        #     estimator=rf,
        #     param_grid=param_grid,
        #     scoring='neg_mean_squared_error',  
        #     cv=5,
        #     n_jobs=-1,
        #     verbose=2
        # )

        # grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        # print("Best parameters found: ", grid_search.best_params_)
        # print("Best score: ", grid_search.best_score_)

        # y_pred = grid_search.best_estimator_.predict(self.X_test_scaled)

        mse = mean_squared_error(self.y_test_scaled, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)


        print("\n Random Forest regressor")
        print(f"Mean Absolute Error (MAE): {mae:.3f}")
        print(f"Mean Squared Error (MSE): {mse:.3f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        print(f"R-squared (R2): {r2:.3f}")
        
        explainer = shap.TreeExplainer(model, self.X_train_scaled)

        shap_values = explainer(self.X_test_scaled)

        print("Shape of shap_values:", shap_values.shape)
        print("Shape of X_test:", self.X_test_scaled.shape)

        shap_values_array = shap_values.values 

        print("Shape of shap_values_array:", shap_values_array[0].shape)

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

        model = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=(100, 50), learning_rate='constant', solver='adam')

        # Train the model
        model.fit(self.X_train_scaled, self.y_train_scaled)

        y_pred = model.predict(self.X_test_scaled)

        

        param_grid = {
            'hidden_layer_sizes': [(50,), (50, 25), (100,), (100, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.00001, 0.0001, 0.0005],
            'learning_rate': ['constant', 'adaptive'],
            # 'learning_rate_init': [0.001, 0.0001, 0.01],
            'warm_start': [True, False],
            # 'momentum': [0.9,0.8,0.7]
        }
        # print(self.y_train_scaled.values.reshape(-1))

        # grid_search = GridSearchCV(MLPRegressor(), param_grid, cv=3)
        # grid_search.fit(self.X_train_scaled, self.y_train_scaled.squeeze())
        # y_pred = grid_search.best_estimator_.predict(self.X_test_scaled)

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
        
        # print("Best parameters:", grid_search.best_params_)

    
    
