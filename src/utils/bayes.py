import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import BayesianRidge, LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv
import pytensor as pt
import arviz as az
from scipy.stats import gamma

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



    def zero_inflated_Bayes(self):

        pass

        
    def bayes(self):

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
            # coefficients = pm.HalfNormal('coefficients', sigma=1, shape=n_features)


            # Define a prior for the intercept
            intercept = pm.Normal('intercept', mu=0, sigma=1)
            
            # Linear combination of features to estimate the mean of y
            y_est = (pm.math.dot(self.X_train_scaled, coefficients)) + intercept
            
            # Define a prior for the noise term
            noise_sigma = pm.HalfNormal('noise_sigma', sigma=1)
            
            # Define the likelihood of the data (i.e., the observed data y_obs)
            y_obs = pm.Normal('y_obs', mu=y_est, sigma=noise_sigma, observed=self.y_train_scaled)

            # Inference: Sample from the posterior distribution using NUTS
            # trace = pm.sample(tune=10000, draws=10000, step=pm.NUTS(target_accept=0.95, max_treedepth=15))
            trace = pm.sample(1000, return_inferencedata=True )

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
        # plt.fill_between(range(len(self.y_test_scaled)), mean_lower_bound, mean_upper_bound, color='gray', alpha=0.3, label="95% Credible Interval")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.legend()
        plt.show()

    def informed_bayes_gamma(self):
        means = np.mean(self.X_train_scaled, axis=0)
        variances = np.var(self.X_train_scaled, axis=0)

        betas = means /( variances)
        alphas = means * betas


        # betas = []
        # alphas = []

        # for column in self.X_train_scaled:
        #     ser = self.X_train_scaled[column]
        #     ag,bg,cg = gamma.fit(ser)
        #     alphas.append(ag)
        #     betas.append(cg)

        with pm.Model() as model:
            
            beta_priors = []
            for i in range(self.X_train_scaled.shape[1]):
                beta_priors.append(pm.Gamma(f"beta_{i}", alpha=alphas[i], beta=betas[i]))
                # beta_priors.append(pm.Beta(f"beta_{i}", alpha=1, beta = 1))
            
            betas_vector = pm.math.stack(beta_priors)

            # betas = []
            # for i in range(self.X_train_scaled.shape[1]):
            #     betas.append(pm.Normal(f"beta_{i}", mu=0, sigma=10))
            
            # betas_vector = pm.math.stack(betas)
    
            mu = pm.math.exp(pm.math.dot(self.X_train_scaled, betas_vector))
            
            Y_obs = pm.Poisson("y_obs", mu=mu, observed=self.y_train_scaled)
            
            # observed_zeros = pm.Bernoulli('observed_zeros', p=p_zero_inflation, observed=(self.y_train_scaled == 0))

            # y_mean = np.mean(self.y_train_scaled)
            # y_var = np.var(self.X_train_scaled)
            # y_beta = y_mean /(y_var)
            # y_alpha = y_mean * y_beta

            # Y_obs = pm.Gamma("y_obs", alpha=1, beta=1, observed=self.y_train_scaled)

            
            trace = pm.sample(1000, return_inferencedata=True)

            # az.plot_trace(trace)

    

        with model:
            posterior_predictive = pm.sample_posterior_predictive(trace)

        print(posterior_predictive.posterior_predictive.keys())

        posterior_y_obs_mean = posterior_predictive.posterior_predictive['y_obs'].mean(axis=(0, 1))  # Average over both chain and draw dimensions

        if len(posterior_y_obs_mean) == len(self.y_train_scaled):
            plt.scatter(self.y_train_scaled, posterior_y_obs_mean)
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title('Posterior Predictive Check')
            plt.show()
        else:
            print(f"Shape mismatch: Observed values have shape {self.y_train_scaled.shape}, but predicted values have shape {posterior_y_obs_mean.shape}")
        
        y_pred = posterior_predictive.posterior_predictive['y_obs'].mean(axis=0)

        y_mean = np.mean(self.y_train_scaled)
        RSS = np.sum((self.y_train_scaled - y_pred) ** 2)
        TSS = np.sum((self.y_train_scaled - y_mean) ** 2)
        R_squared = 1 - (RSS / TSS)
        print(f"R-squared: {R_squared:.4f}")


    def informed_bayes_separate(self):
        """
        TODO:
        Implement Zero Boosting
        """
        print(self.y_train_scaled == 0)

        #Temporary check, removing zero values:
        rows_to_drop = self.y_train_scaled == 0
        self.y_train_scaled = self.y_train_scaled[~rows_to_drop]
        self.X_train_scaled = self.X_train_scaled.values[~rows_to_drop]
        print(f"Shape of X_train_scaled: {self.X_train_scaled.shape}")
        print(f"Shape of y_train_scaled: {self.y_train_scaled.shape}")
        print(self.X_train_scaled)

        ols_model = LinearRegression()
        ols_model.fit(self.X_train_scaled, self.y_train_scaled)

        ols_coeffs = ols_model.coef_  # Coefficients for the predictors
        ols_intercept = ols_model.intercept_  # Intercept
        ols_residuals = self.y_train_scaled - ols_model.predict(self.X_train_scaled)  # Residuals
        # print(ols_residuals)

        ols_sigma_squared = np.var(ols_residuals)
        ols_sigma = np.sqrt(ols_sigma_squared)
        # print(ols_sigma)
        X_train_scaled = np.asarray(self.X_train_scaled)
        n, p = X_train_scaled.shape  # n: number of samples, p: number of predictors
        X_design = np.hstack([np.ones((n, 1)), X_train_scaled])  # Add intercept to design matrix
        cov_matrix = np.linalg.inv(X_design.T @ X_design) * ols_sigma_squared

        # Extract standard errors (sqrt of diagonal elements)
        standard_errors = np.sqrt(np.diag(cov_matrix))

        # Separate intercept and coefficients' standard errors
        intercept_se = standard_errors[0]
        coeffs_se = standard_errors[1:]

        print("Intercept Standard Error:", intercept_se)
        print("Coefficients Standard Errors:", coeffs_se)
        with pm.Model() as model:
            # Prior for intercept (based on OLS estimate)
            intercept_prior = pm.Gamma('intercept', mu=ols_intercept, sigma=intercept_se*2)
            
            # Ensure self.X_train_scaled is a numpy array (if it's a DataFrame, convert it)
            X_train_scaled = np.asarray(self.X_train_scaled)  # Convert to numpy array if necessary
            
            coeffs_priors = []
            for i in range(X_train_scaled.shape[1]):  # Number of features
                coeffs_priors.append(pm.Normal(f'coeff_{i}', mu=ols_coeffs[i], sigma=coeffs_se[i]))
            
            mu = intercept_prior + pm.math.dot(X_train_scaled, coeffs_priors)            
            sigma = pm.HalfNormal('sigma', sigma=ols_sigma)

            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.y_train_scaled)

            trace = pm.sample(1000, return_inferencedata=True)

            # az.plot_trace(trace)

        # After sampling, you can get the posterior predictive samples
        with model:
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y_obs"])

        # Check the available keys in posterior predictive
        print(posterior_predictive.posterior_predictive.keys())

        posterior_y_obs_mean = posterior_predictive.posterior_predictive['y_obs'].mean(axis=(0, 1))  # Average over both chain and draw dimensions

        # Ensure that posterior_y_obs_mean has the same shape as y_train_scaled
        if len(posterior_y_obs_mean) == len(self.y_train_scaled):
            plt.scatter(self.y_train_scaled, posterior_y_obs_mean)
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title('Posterior Predictive Check')
            plt.show()
        else:
            print(f"Shape mismatch: Observed values have shape {self.y_train_scaled.shape}, but predicted values have shape {posterior_y_obs_mean.shape}")

        # # If 'y_obs' exists in posterior predictive, use it for the scatter plot
        # if 'y_obs' in posterior_predictive.posterior_predictive:
        #     posterior_y_obs = posterior_predictive.posterior_predictive['y_obs']
        #     plt.scatter(self.y_train_scaled, posterior_y_obs.mean(axis=0))
        #     plt.xlabel('Observed')
        #     plt.ylabel('Predicted')
        #     plt.title('Posterior Predictive Check')
        #     plt.show()
        # else:
        #     print("y_obs not found in posterior predictive!")



    def  informed_Bayes(self):
        n_samples, n_features = self.X_train_scaled.shape
        ols_model = LinearRegression()
        ols_model.fit(self.X_train_scaled, self.y_train_scaled)

        # Get the OLS estimates
        ols_coeffs = ols_model.coef_  # Coefficients for the predictors
        ols_intercept = ols_model.intercept_  # Intercept
        ols_residuals = self.y_train_scaled - ols_model.predict(self.X_train_scaled)  # Residuals

        # Estimate the residual variance (sigma^2)
        ols_sigma_squared = np.var(ols_residuals)
        ols_sigma = np.sqrt(ols_sigma_squared)
        
        with pm.Model() as model:
            # Informative priors based on the OLS estimates
            intercept_prior = pm.Normal('intercept', mu=ols_intercept, sigma=10)  # Use OLS intercept, large sigma for flexibility
            coeffs_prior = pm.Normal('coeffs', mu=ols_coeffs, sigma=10, shape=self.X_train_scaled.shape[1])  # Use OLS coefficients
            
            # Likelihood (linear regression model)
            mu = intercept_prior + pm.math.dot(self.X_train_scaled.values, coeffs_prior)  # Linear model: y = X * coeffs + intercept
            
            # Prior on the error term (noise variance) based on OLS residual variance
            sigma = pm.HalfNormal('sigma', sigma=ols_sigma)  # Prior for noise term based on OLS residuals
            
            # Likelihood (data)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=self.y_train_scaled)  # Observed target variable
            
            # Sample from the posterior using MCMC
            trace = pm.sample(1000, return_inferencedata=True)

        with model:
            posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y_obs"])
        print(posterior_predictive.keys())

        plt.scatter(self.y_train_scaled, posterior_predictive.posterior_predictive['y_obs'].mean(axis=0))
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title('Posterior Predictive Check')
        plt.show()


        # # Fit Linear Regression
        # model = LinearRegression().fit(self.X_train_scaled, self.y_train_scaled)
        # beta_hat = model.coef_  # Coefficient estimates
        # residuals = self.y_train_scaled - model.predict(self.X_train_scaled)
        # sigma_hat = np.std(residuals)
        # XTX_inv = inv(self.X_train_scaled.T @ self.X_train_scaled)
        # beta_var = sigma_hat**2 * np.diag(XTX_inv)



        # with pm.Model() as bayesian_model:
        #     # Priors for coefficients
        #     beta = pm.Normal("beta", mu=beta_hat, sigma=np.sqrt(beta_var), shape=n_features)
            
        #     # Prior for residual standard deviation
        #     sigma = pm.HalfNormal("sigma", sigma=1.0)
            
        #     # Likelihood
        #     y_obs = pm.Normal("y_obs", mu=pm.math.dot(self.X_train_scaled, beta), sigma=sigma, observed=self.y_train_scaled)
            
        #     # Sample from the posterior
        #     trace = pm.sample(10, return_inferencedata=True)

        # print("Posterior keys:", trace.posterior.keys())

        # with bayesian_model:
        #     # Posterior predictive sampling
        #     posterior_predictive = pm.sample_posterior_predictive(
        #         # samples=1000,  # Number of samples to draw
        #         var_names=["beta", "sigma"],  # Variables to draw predictions for
        #         trace = trace  # Specify posterior object
        #     )

        # print(trace.posterior["beta"])

        # # Extract posterior samples for coefficients
        # beta_samples = posterior_predictive["beta"]  # Shape: (samples, n_features)
        # sigma_samples = posterior_predictive["sigma"]  # Shape: (samples, )

        # # Make predictions for the test data
        # y_pred_samples = np.dot(self.X_test_scaled, beta_samples.T)  # Shape: (samples, n_test_samples)
        # y_pred_samples += sigma_samples[:, np.newaxis]
        # y_pred_mean = y_pred_samples.mean(axis=0)  # Mean across samples for each test point

        # # 95% Credible interval (lower and upper bounds)
        # y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)  # Lower 2.5%
        # y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

        # y_pred_mean = y_pred_samples.mean(axis=0)

        # # Calculate MSE, MAE, and R-squared
        # mse = mean_squared_error(self.y_test_scaled, y_pred_mean)
        # mae = mean_absolute_error(self.y_test_scaled, y_pred_mean)
        # r2 = r2_score(self.y_test_scaled, y_pred_mean)

        # print(f"MSE: {mse}")
        # print(f"MAE: {mae}")
        # print(f"R-squared: {r2}")

        # plt.scatter(self.y_test_scaled, y_pred_mean)
        # plt.plot([min(self.y_test_scaled), max(self.y_test_scaled)], [min(self.y_test_scaled), max(self.y_test_scaled)], color='red')  # Ideal line
        # plt.xlabel("Observed values")
        # plt.ylabel("Predicted values")
        # plt.title("Observed vs Predicted")
        # plt.show()

        # # Distribution of residuals
        # residuals = self.y_test_scaled - y_pred_mean
        # plt.hist(residuals, bins=30, density=True)
        # plt.title("Posterior Predictive Check - Residuals")
        # plt.show()


    def zip_Bayes(self):
        n_samples, n_predictors = self.X_train_scaled.shape

        with pm.Model() as zip_model:
            # Priors for zero-inflation component
            alpha = pm.Normal('alpha', mu=0, sigma=10, shape=n_predictors + 1)  # Intercept + predictors

            # Priors for count component
            beta = pm.Normal('beta', mu=0, sigma=10, shape=n_predictors + 1)  # Intercept + predictors

            # Zero-inflation linear predictor
            logit_pi = alpha[0] + pm.math.dot(self.X_train_scaled, alpha[1:])
            pi = pm.math.sigmoid(logit_pi)  # Convert log-odds to probability

            # Count linear predictor
            log_lambda = beta[0] + pm.math.dot(self.X_train_scaled, beta[1:])
            lambda_ = pm.math.exp(log_lambda)  # Convert log-scale to Poisson rate

            # Zero-Inflated Poisson likelihood
            Y_obs = pm.ZeroInflatedPoisson('Y_obs', mu=lambda_, psi=pi, observed=self.y_train_scaled)

            # Sampling
            trace_zip_no_site = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)
        with zip_model:
            ppc_no_site = pm.sample_posterior_predictive(trace_zip_no_site, var_names=["Y_obs"])

        # Visualization: Compare observed and predicted distributions
        import matplotlib.pyplot as plt

        plt.hist(self.y_train_scaled, bins=30, alpha=0.5, label='Observed', density=True)
        plt.hist(ppc_no_site.posterior_predictive['Y_obs'].mean(axis=0), bins=30, alpha=0.5, label='Predicted', density=True)
        plt.legend()
        plt.show()

