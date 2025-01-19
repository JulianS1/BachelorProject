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
import seaborn as sns

import pymc_bart as pmb


X_train = pd.read_csv("Nephytyidae/X_train_fauna.csv",
        sep=",",
        encoding="utf-8")
y_train = pd.read_csv("Nephytyidae/y_train_fauna.csv",
        sep=",",
        encoding="utf-8")
X_test = pd.read_csv("Nephytyidae/X_test_fauna.csv",
        sep=",",
        encoding="utf-8")
y_test = pd.read_csv("Nephytyidae/y_test_fauna.csv",
        sep=",",
        encoding="utf-8")

# X_train = pd.read_csv("X_train_fauna.csv",
#         sep=",",
#         encoding="utf-8")
# y_train = pd.read_csv("y_train_fauna.csv",
#         sep=",",
#         encoding="utf-8")

y_train=y_train.values.ravel()


#Drop negative values. 

X_train = X_train.loc[:, (X_train >= 0).all(axis=0)]

labels = [f"coeff_{i}" for i in range(32)]  # Names for each coefficient

# Define coords to match the number of coefficients
coords = {"coeffs": labels}

# X_train = np.asarray(X_train)
# y_train = np.asarray(y_train)

# print("X_train shape: ",X_train.shape)
# print("y_train shape: ",y_train.shape)

prop_zero = np.mean(y_train==0.0)
print("Proportion of zeros:", prop_zero)
# print(y_train)
alpha = prop_zero * 10 
beta = (1 - prop_zero) * 10

# with pm.Model() as bart_g:
#     X = pm.Data("X", X_train)
#     y = pm.Data("y", y_train)
#     σ = pm.Gamma("σ", mu=np.mean(y.get_value()), sigma=np.var(y.get_value()*2))
#     μ= pmb.BART("μ_raw", X, y.get_value(), m=100)
#     # μ = pm.Deterministic("μ", pm.math.log(1 + pm.math.exp(μ_raw)))
#     y_obs = pm.Gamma("y_obs", mu=np.abs(μ), sigma=σ, observed=y)
#     trace = pm.sample(10, return_inferencedata=True)

with pm.Model() as bart_g:
    X = pm.Data("X", X_train)
    y = pm.Data("y", y_train + 0.00001)
    π = pm.Beta("π", alpha=alpha, beta=beta)  # π is the probability of zero
    
    # Zero-inflation: Bernoulli likelihood
    # zero_inflation = pm.Bernoulli("zero_inflation", p=π, observed=(y == 0.0))

    
    σ = pm.HalfNormal("σ", sigma=10)
    μ_raw= pmb.BART("μ_raw", X, y.get_value(), m=100)
    # μ = pm.Deterministic("μ", pm.math.log(1 + pm.math.exp(μ_raw)))
    μ = pm.Deterministic("μ", pm.math.abs(μ_raw))
    # y_obs = pm.Normal("y_obs", mu=μ, sigma=σ, observed=y)
    y_obs = pm.Mixture(
    "y_obs",
    w=[π, 1 - π],  # Zero-inflated weight and regular distribution weight
    comp_dists=[
        pm.Normal.dist(mu=0, sigma=1e-6),  # Continuous approximation of zero
        pm.Gamma.dist(mu=μ, sigma=σ),  # Normal for non-zero observations
    ],
    observed=y
)

    trace = pm.sample(50000, tune=1000, target_accept=0.9, return_inferencedata=True)

# pm.model_to_graphviz(model)

# with model:
#     trace = pm.sample()


"""
with bart_g:
    pm.set_data({"X": X_train, "y": np.zeros(X_train.shape[0])})
    # pm.set_data({"X": X_test, "y": y_test})
    temptrace = trace
    trace.extend(pm.sample_posterior_predictive(temptrace))


p_train_pred = trace.posterior_predictive["y_obs"].mean(dim=["chain", "draw"])

mse = mean_squared_error(y_train, p_train_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squareds
r2 = r2_score(y_train, p_train_pred)
print(f"R-squared (R²): {r2}")

plt.scatter(y_train, p_train_pred)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.legend()
plt.show()

"""


with bart_g:

    pm.set_data({"X": X_test, "y": y_test.values.ravel()})
    pp = pm.sample_posterior_predictive(trace, predictions=True)


p_test_pred = pp.predictions["y_obs"].mean(dim=["chain", "draw"])

mse = mean_squared_error(y_test, p_test_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate R-squareds
r2 = r2_score(y_test, p_test_pred)
print(f"R-squared (R²): {r2}")


plt.scatter(y_test, p_test_pred)
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.legend()
plt.show()

sns.histplot(y_test, kde=True, label="Observed")
sns.histplot(p_test_pred, kde=True, label="Predicted", color="orange")
plt.legend()
plt.show()
az.plot_posterior(trace)
