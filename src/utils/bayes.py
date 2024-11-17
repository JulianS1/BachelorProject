import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt

class Bayes:

    def __init__(self) -> None:
        
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
        self.y_train_scaled = self.y_train_scaled.values.ravel()
        self.y_test_scaled = self.y_test_scaled.values.ravel()

        

        model = BayesianRidge()
        model.fit(self.X_train_scaled, self.y_train_scaled)

        y_pred = model.predict(self.X_test_scaled)
        mae = mean_absolute_error(self.y_test_scaled, y_pred)
        mse = mean_squared_error(self.y_test_scaled, y_pred)
        r2 = r2_score(self.y_test_scaled, y_pred)

        # Print the metrics
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("R² Score:", r2)



        # Set the random seed for reproducibility
        np.random.seed(42)

        # Get the number of samples and features
        n_samples = len(self.X_train_scaled)  # Number of samples
        n_features = len(self.X_train_scaled.columns)  # Number of features

        with pm.Model() as model:
            # Define priors for each feature's coefficient
            coefficients = pm.Normal('coefficients', mu=0, sigma=1, shape=n_features)
            
            # Define a prior for the intercept
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            
            # Linear combination of features to estimate the mean of y
            y_est = pm.math.dot(self.X_train_scaled, coefficients) + intercept
            
            # Define a prior for the noise term
            noise_sigma = pm.HalfNormal('noise_sigma', sigma=1)
            
            # Define the likelihood of the data (i.e., the observed data y_obs)
            y_obs = pm.Normal('y_obs', mu=y_est, sigma=noise_sigma, observed=self.y_train_scaled)

            # Inference: Sample from the posterior distribution using NUTS
            trace = pm.sample(tune=10000, draws=10000, step=pm.NUTS(target_accept=0.95, max_treedepth=15))

        # Debugging: Check the keys in the posterior of the InferenceData object
        print("Posterior keys:", trace.posterior.keys())  # This will print the variable names in the posterior

        # Extract the posterior samples
        posterior_coefficients = trace.posterior["coefficients"]
        posterior_intercept = trace.posterior["intercept"]
        posterior_noise_sigma = trace.posterior["noise_sigma"]

        # Average over the chains and draws (axis 0 for chains, axis 1 for draws)
        mean_coefficients = posterior_coefficients.mean(axis=(0, 1)).values  # shape: (38,)
        mean_intercept = posterior_intercept.mean(axis=(0, 1)).values  # shape: (1,)
        mean_noise_sigma = posterior_noise_sigma.mean(axis=(0, 1)).values  # shape: (1,)

        # Compute predictions using the mean posterior coefficients and intercept
        predictions = np.dot(self.X_test_scaled, mean_coefficients) + mean_intercept  # Shape: (70,)

        # Compute 95% credible intervals for the predictions using posterior predictive samples
        # Reshape posterior coefficients for each prediction sample (along axis 0 for samples)
        # posterior_coefficients_reshaped = posterior_coefficients.mean(axis=0).values  # shape: (1000, 38)
        # posterior_intercept_reshaped = posterior_intercept.mean(axis=0).values  # shape: (1000,)
        # predictions_samples = np.dot(self.X_test_scaled, posterior_coefficients_reshaped.T) + posterior_intercept_reshaped.T  # Shape: (1000, 70)

        with model:
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

        # Extract the posterior predictive samples for y_obs
        posterior_predictive_data = posterior_predictive["posterior_predictive"]
        y_obs_samples = posterior_predictive_data["y_obs"]

        # Check the shape of the posterior predictive samples
        print(f"Shape of posterior_predictive['y_obs']: {y_obs_samples.shape}")
        # Expected shape should be (chains, draws, test_samples), i.e., (4, 10, 70)

        # Get the number of test samples (70 in this case)
        n_test_samples = len(self.y_test_scaled)

        # If the shape is still (4, 10, 280), slice it to match the number of test samples (70)
        # Slice the posterior predictive samples to select only the test set samples
        y_obs_samples_test = y_obs_samples[:, :, :n_test_samples]

        # Compute the 95% credible intervals (percentiles) for the test set
        credible_intervals = np.percentile(y_obs_samples_test, [2.5, 97.5], axis=0)

        # Check the shape of credible intervals
        print(f"Shape of credible_intervals: {credible_intervals.shape}")
        # Expected shape: (2, 70), i.e., lower and upper bounds for each test sample

        # Compute coverage: Check if the actual values in the test set are within the credible intervals
        coverage = np.mean((self.y_test_scaled >= credible_intervals[0, :]) & (self.y_test_scaled <= credible_intervals[1, :]))

        print(f"Coverage of 95% credible intervals: {coverage * 100:.2f}%")

        # # Calculate the 95% credible intervals for the predictions
        # credible_intervals = np.percentile(y_obs_samples, [2.5, 97.5], axis=0)  # Shape: (2, 280)

        # # Compute coverage for the credible intervals
        # coverage = np.mean((self.y_test_scaled >= credible_intervals[0, :]) & (self.y_test_scaled <= credible_intervals[1, :]))
        # print(f"Coverage of 95% credible intervals: {coverage * 100:.2f}%")


        # Performance Metrics
        mse = mean_squared_error(self.y_test_scaled, predictions)
        print("Mean Squared Error:", mse)

        r_squared = r2_score(self.y_test_scaled, predictions)
        print("R-squared:", r_squared)

        # Calculate credible intervals for the test set samples
        credible_intervals = np.percentile(posterior_predictive_data['y_obs'], [2.5, 97.5], axis=0)

        # Now, credible_intervals should have the shape (2, 10, 280)
        # Slice to match the number of test samples (n_test_samples = 50)
        credible_intervals = credible_intervals[:, :, :n_test_samples]  # Shape will be (2, 10, 50)

        # Now we need to calculate the lower and upper bounds for each test sample
        # We calculate the lower and upper bounds for each of the 50 test samples (axis 0 for percentiles)
        lower_bound = credible_intervals[0, :, :]  # Lower bound for each posterior sample (2.5% percentile)
        upper_bound = credible_intervals[1, :, :]  # Upper bound for each posterior sample (97.5% percentile)

        # We now reduce the axis 0 (across chains and draws) to obtain the final bounds
        # Calculate the mean of the bounds across posterior draws and chains
        mean_lower_bound = np.mean(lower_bound, axis=(0, 1))  # Mean of lower bounds across chains and draws
        mean_upper_bound = np.mean(upper_bound, axis=(0, 1))  # Mean of upper bounds across chains and draws

        # Plot Results
        plt.figure(figsize=(10, 5))
        plt.plot(self.y_test_scaled, predictions, 'o', label="Predicted vs Actual")
        plt.fill_between(range(len(self.y_test_scaled)), mean_lower_bound, mean_upper_bound, color='gray', alpha=0.3, label="95% Credible Interval")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.show()