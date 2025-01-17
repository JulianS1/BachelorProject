library(brms)
library(bayesplot)


# X_train <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/X_train_fauna_Unscaled.csv")
# y_train <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/y_train_fauna_Unscaled.csv")
# X_test <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/X_test_fauna_Unscaled.csv")
# y_test <- read.csv("/home/julian/Documents/BachelorProject/Bayes_test/Nephtyidae/y_test_fauna_Unscaled.csv")
X_train <- read.csv("/home2/s4711297/data/X_train_fauna_Unscaled.csv")
y_train <- read.csv("/home2/s4711297/data/y_train_fauna_Unscaled.csv")
X_test <- read.csv("/home2/s4711297/data/X_test_fauna_Unscaled.csv")
y_test <- read.csv("/home2/s4711297/data/y_test_fauna_Unscaled.csv")
y_train$Nephtyidae <- pmax(0, round(y_train$Nephtyidae))
y_test$Nephtyidae <- pmax(0, round(y_test$Nephtyidae))
data <- cbind(y_train, X_train)
colnames(data)[1] <- "abundance"
temp <- cbind(y_test,X_test)
colnames(temp)[1] <- "abundance"
data <- rbind(data, temp)

# formula_string_main <- paste(
#   "abundance ~", 
#   paste(
#     ifelse(colnames(predictors) %in% c("NumberofMetalsExceedingLevel1", "NumberofMetalsExceedingLevel2"),
#            colnames(predictors), 
#            paste0("s(", colnames(predictors), ")")
#     ), 
#     collapse = " + "
#   )
# )
# 
# formula_string_hu <- paste(
#   "hu ~", 
#   paste(
#     ifelse(colnames(predictors) %in% c("NumberofMetalsExceedingLevel1", "NumberofMetalsExceedingLevel2"),
#            colnames(predictors), 
#            paste0("s(", colnames(predictors), ")")
#     ), 
#     collapse = " + "
#   )
# )
selecting <- c("abundance","Perylene", "Gravel", "Zn.1", "Fe.1", "Cu.1", "USEPASumPAH",
                "LevelIIquotient", "Fluoranthene", "SumHMWPAH", "Indeno.1.2.3.cd.pyrene",
                "Chrysene", "Cd.1", "Dibenz.a.h.anthracene", "Benzo.b.fluoranthene",
                "V.1", "X.metalsenriched", "Benz.a.anthracene", "Acenaphthylene",
                "Hg.1", "Totalorganiccontent")

# Extract those columns from the data frame
selected_data <- data[, selecting, drop = FALSE]


formula_string_main <- paste("abundance ~", paste("s(", c("Perylene", "Gravel", "Zn.1", "Fe.1", "Cu.1", "USEPASumPAH",
                                                     "LevelIIquotient", "Fluoranthene", "SumHMWPAH", "Indeno.1.2.3.cd.pyrene",
                                                     "Chrysene", "Cd.1", "Dibenz.a.h.anthracene", "Benzo.b.fluoranthene", "V.1", "X.metalsenriched", "Benz.a.anthracene", "Acenaphthylene", "Hg.1",
                                                     "Totalorganiccontent"), ")", collapse = " + "))

formula_string_hu <- paste("hu ~", paste("s(", c("Perylene", "Gravel", "Zn.1", "Fe.1", "Cu.1", "USEPASumPAH",
                                                          "LevelIIquotient", "Fluoranthene", "SumHMWPAH", "Indeno.1.2.3.cd.pyrene",
                                                          "Chrysene", "Cd.1", "Dibenz.a.h.anthracene", "Benzo.b.fluoranthene", "V.1", "X.metalsenriched", "Benz.a.anthracene", "Acenaphthylene", "Hg.1",
                                                          "Totalorganiccontent"), ")", collapse = " + "))


formula_main <- bf(
  as.formula(formula_string_main),
  as.formula(formula_string_hu)
)

formula <- bf(
  abundance ~ s(Perylene) + s(Gravel),
  hu ~ s(Perylene) + s(Gravel)
)

priors <- c(
  prior(normal(0, 5), class = "b"),  # Coefficients for predictors
  prior(normal(log(100), 1), class = "Intercept"),  # Intercept
  prior(normal(0, 1), class = "b", dpar = "hu"),
  prior(normal(0, 1), class = "Intercept", dpar = "hu")
)

hurdle_model <- brm(
  formula,
  data = selected_data,
  family = hurdle_poisson(),
  prior = priors,
  chains = 4,
  iter = 2000,
  cores = 4,
  seed = 123,
  init = 0,
  control = list(max_treedepth = 15),
)



pp_check(hurdle_model)

# Print model summary
summary(hurdle_model)

library(bayesplot)
r2 <- bayes_R2(hurdle_model)
print(r2)

loo(hurdle_model)

mcmc_trace(hurdle_model, window=c(100, 130))

