library(brms)
library(bayesplot)

X_train <- read.csv("/home2/s4711297/data/X_train_fauna_Unscaled.csv")
y_train <- read.csv("/home2/s4711297/data/y_train_fauna_Unscaled.csv")
X_test <- read.csv("/home2/s4711297/data/X_test_fauna_Unscaled.csv")
y_test <- read.csv("/home2/s4711297/data/y_test_fauna_Unscaled.csv")
y_train$Nephtyidae <- pmax(0, round(y_train$Nephtyidae))

data <- cbind(y_train, X_train)
colnames(data)[1] <- "abundance"

formula <- bf(
  abundance ~ .,#Gravel + Mud + Totalorganiccontent + Sand + Fe.1 + As.1 + Ba.1 + Be.1 + Cd.1 + Co.1 + Cu.1 + Cr.1 + Mn.1 + Hg.1 + Ni.1 + Pb.1 + V.1 + Zn.1 + X.metalsenriched + Cumulativeenrichmentfactor,  # Count component with all predictors
  hu ~ .#Gravel + Mud + Totalorganiccontent + Sand + Fe.1 + As.1 + Ba.1 + Be.1 + Cd.1 + Co.1 + Cu.1 + Cr.1 + Mn.1 + Hg.1 + Ni.1 + Pb.1 + V.1 + Zn.1 + X.metalsenriched + Cumulativeenrichmentfactor           # Hurdle component with all predictors
)

priors <- c(
  prior(normal(0, 2), class = "b"),  # Coefficients for predictors
  prior(normal(log(100), 1), class = "Intercept")  # Intercept
)

hurdle_model <- brm(
  formula,
  data = data,
  family = hurdle_poisson(),
  prior = priors,
  chains = 4,
  iter = 2000,
  cores = 8,
  seed = 123,
  init = 0,
  control = list(max_treedepth = 15),
)

# Print model summary
summary(hurdle_model)

library(bayesplot)
r2 <- bayes_R2(hurdle_model)
print(r2)

new_data <- cbind(y_test, X_test)
colnames(new_data)[1] <- "abundance"
posterior_predictions <- posterior_predict(hurdle_model, newdata = new_data)

sum(is.na(posterior_predictions))


# Compute the mean of posterior predictions for each observation
predicted_means <- colMeans(posterior_predictions)

# Print the predicted means
print(predicted_means)

# Calculate R-squared on test data
observed <- new_data$abundance
ss_total <- sum((observed - mean(observed))^2)
ss_residual <- sum((observed - predicted_means)^2)
r_squared <- 1 - (ss_residual / ss_total)

print(paste("Test Data R-squared:", r_squared))

