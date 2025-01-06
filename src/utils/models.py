import os
import sklearn as sk
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, MultiTaskLassoCV, RidgeCV, MultiTaskElasticNetCV, BayesianRidge, Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import RegressorMixin

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
        # print(self.X_test_scaled.head())

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
        # self.X_train_locations = self.X_train_scaled.loc[:, "Port_Cape Town":]
        # self.X_train_scaled = self.X_train_scaled.loc[:, :"SQILowerlimit"]
        # self.X_test_locations = self.X_test_scaled.loc[:, "Port_Cape Town":]
        # self.X_test_scaled = self.X_test_scaled.loc[:, :"SQILowerlimit"]

        self.y_train_scaled = self.y_train_scaled.values.ravel()
        self.y_test_scaled = self.y_test_scaled.values.ravel()

    def linearModel(self):

        # lasso_cv = RidgeCV(cv=5)
        # lasso_cv.fit(self.X_train_scaled, self.y_train_scaled)
        # lasso_best = Lasso(alpha=lasso_cv.alpha_)
        # lasso_best.fit(self.X_train_scaled, self.y_train_scaled)
        # y_pred = lasso_best.predict(self.X_test_scaled)

        
        ridge = Ridge()

        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }

        grid_search = GridSearchCV(
            estimator=ridge,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
        )
        grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        model = grid_search.best_estimator_
        y_pred = model.predict(self.X_test_scaled)

        

        model = LinearRegression()
        
        model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = model.predict(self.X_test_scaled)
        

        param_grid = {
            'alpha_1': [1e-6, 1e-5, 1e-4],
            'alpha_2': [1e-6, 1e-5, 1e-4],
            'lambda_1': [1e-6, 1e-5, 1e-4],
            'lambda_2': [1e-6, 1e-5, 1e-4]
        }

        bayes = BayesianRidge()
        bayes.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = bayes.predict(self.X_test_scaled)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(bayes, param_grid, cv=cv, scoring='neg_mean_squared_error')

        # Fit the grid search
        grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the best estimator and parameters
        best_bayes = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print("Best Parameters:", best_params)
        y_pred = best_bayes.predict(self.X_test_scaled)

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

        shap_values = explainer(self.X_test_scaled)


        shap_values_single_output = shap_values[..., 0]

        shap.plots.bar(shap_values_single_output, show=False)
        file_name = "LinearRegression_SHAP"

        output_directory = os.path.join("..", "..", "results", "SHAP")
        os.makedirs("../../results/SHAP", exist_ok=True)
        plt.savefig(os.path.join(output_directory, file_name), bbox_inches='tight') 
        plt.close()

        # shap.plots.bar(shap_values)
        shap.summary_plot(shap_values, self.X_test_scaled, show=False, feature_names=self.X_train_scaled.columns)
        plt.savefig(
        os.path.join("../../results/SHAP", "LR_SHAP_Spionidae.png")
        )
        plt.close()

        # self._remake_data(shap_values, 10)

        grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        model = grid_search.best_estimator_
        y_pred = model.predict(self.X_test_scaled)

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

        
        model = RandomForestRegressor(max_depth= None, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split= 2, n_estimators= 100, random_state=42)
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

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',  
            cv=5,
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(self.X_train_scaled, self.y_train_scaled)

        print("Best parameters found: ", grid_search.best_params_)
        print("Best score: ", grid_search.best_score_)

        y_pred = grid_search.best_estimator_.predict(self.X_test_scaled)

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

        shap_values = explainer(self.X_test_scaled, check_additivity=False)
        # print("RF VALUES \n", shap_values)

        # print("Shape of shap_values:", shap_values.shape)
        # print("Shape of X_test:", self.X_test_scaled.shape)

        # shap_values_array = shap_values.values 

        # print("Shape of shap_values_array:", shap_values_array[0].shape)

        shap_values_list = []
        shap.summary_plot(shap_values, self.X_test_scaled, show=False, feature_names=self.X_train_scaled.columns)
        plt.savefig(
        os.path.join("../../results/SHAP", "RF_SHAP_Spionidae.png")
        )
        plt.close()

        

        # Calculate mean absolute SHAP values for each feature
        
        
        # self._remake_data(shap_values, 5)

        model = RandomForestRegressor(max_depth= None, max_features= 'log2', min_samples_leaf= 4, min_samples_split= 10, n_estimators= 100, random_state=42)
        model.fit(self.X_train_scaled, self.y_train_scaled)
        y_pred = model.predict(self.X_test_scaled)
        
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

        model = MLPRegressor(random_state=42, activation='relu', alpha=0.0001, hidden_layer_sizes=(100, 50), learning_rate='constant', solver='adam')

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

        # explainer = shap.KernelExplainer(model.predict, self.X_train_scaled)

        # shap_values = explainer(self.X_test_scaled)

        # self._remake_data(shap_values, 20)
        
        # model = MLPRegressor(random_state=42, activation='relu', alpha=0.0001, hidden_layer_sizes=(100, 50), learning_rate='constant', solver='adam')

        # model.fit(self.X_train_scaled, self.y_train_scaled)

        # y_pred = model.predict(self.X_test_scaled)

        # mse = mean_squared_error(self.y_test_scaled, y_pred)
        # rmse = np.sqrt(mse)
        # mae = mean_absolute_error(self.y_test_scaled, y_pred)
        # r2 = r2_score(self.y_test_scaled, y_pred)

        # print("\n MLP regression")
        # print(f'Mean Squared Error: {mae:.2f}')
        # print(f"Mean Squared Error (MSE): {mse:.3f}")
        # print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
        # print(f"R-squared (R2): {r2:.3f}")
        
        # print("Best parameters:", grid_search.best_params_)

    def _remake_data(self, shap_values, num_values):

        shap_df = pd.DataFrame(shap_values.values, columns=self.X_test_scaled.columns)

        feature_importance = shap_df.abs().mean().sort_values(ascending=False)
        ranked_features = feature_importance.reset_index()
        ranked_features.columns = ['Feature', 'Mean Absolute SHAP Value']
        ranked_features['Rank'] = ranked_features['Mean Absolute SHAP Value'].rank(ascending=False)
        print("Ranked Features DataFrame:\n", ranked_features)

        # print(ranked_features)

        top_features = ranked_features['Feature'].head(num_values).tolist()
        self.X_train_scaled = self.X_train_scaled[top_features]
        # self.y_train_scaled = self.y_train_scaled.values.ravel()
        self.X_test_scaled = self.X_test_scaled[top_features]
        # self.y_test_scaled = self.y_test_scaled.values.ravel()
        
        # self.X_train_scaled = self._concatDF(self.X_train_scaled, self.X_train_locations)
        # self.X_test_scaled = self._concatDF(self.X_test_scaled, self.X_test_locations)
        print(self.X_train_scaled)
        

    def _concatDF(self, df1, df2):
        return pd.concat([df1,df2], axis=1)