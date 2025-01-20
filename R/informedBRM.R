library(brms)
library(bayesplot)


X_train <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/X_train_fauna_Unscaled.csv")
y_train <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/y_train_fauna_Unscaled.csv")
X_test <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/X_test_fauna_Unscaled.csv")
y_test <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/y_test_fauna_Unscaled.csv")
# X_train <- read.csv("/home2/s4711297/data/X_train_fauna_Unscaled.csv")
# y_train <- read.csv("/home2/s4711297/data/y_train_fauna_Unscaled.csv")
# X_test <- read.csv("/home2/s4711297/data/X_test_fauna_Unscaled.csv")
# y_test <- read.csv("/home2/s4711297/data/y_test_fauna_Unscaled.csv")

y_train$Nephtyidae <- pmax(0, round(y_train$Nephtyidae))
y_test$Nephtyidae <- pmax(0, round(y_test$Nephtyidae))
data <- cbind(y_train, X_train)
colnames(data)[1] <- "abundance"
temp <- cbind(y_test,X_test)
colnames(temp)[1] <- "abundance"
data <- rbind(data, temp)

formula <- bf(
  abundance ~ .,
  hu ~ .
)
priors <- c(
  prior(normal(1, 2), class = "b", coef="Ba.1"),  # Coefficients for predictors
  prior(normal(-1, 2), class = "b", coef="Pb1"),
  prior(normal(1, 1), class = "b", coef="Mud"),
  prior(normal(-1, 2), class = "b", coef="Naphthalene"),
  prior(normal(-1, 2), class = "b", coef="Perylene"),
  prior(normal(log(100), 2), class = "Intercept")  # Intercept
)


hurdle_model <- brm(
  formula,
  data = data,
  # family = hurdle_poisson(),
  family = hurdle_poisson(),
  prior = priors,
  chains = 4,
  iter = 2000,
  cores = 7,
  seed = 42,
  init = 0,
  # control = list(adapt_delta = 0.99)
  control = list(max_treedepth = 15),
)

saveRDS(hurdle_model, file="informed_linear_model_home.rds")

# Print model summary
summary(hurdle_model)

library(bayesplot)
r2 <- bayes_R2(hurdle_model)
print(r2)

loo(hurdle_model)
