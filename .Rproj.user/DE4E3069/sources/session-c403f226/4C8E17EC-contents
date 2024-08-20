# Data Imputation Analysis Script
#
# This script performs a comprehensive analysis of missing data and applies
# various imputation techniques to handle missing values in a dataset.
# It includes data exploration, visualization, multiple imputation methods,
# and evaluation of imputation quality.
#
# The script is divided into the following sections:
# 1. Setup and Data Generation
# 2. Missing Data Exploration
# 3. Simple Imputation Methods
# 4. Multiple Imputation using MICE
# 5. Visualization of Imputed Data
# 6. Imputation Quality Evaluation
# 7. Sensitivity Analysis
# 8. Data Export and Reporting

# Section 1: Setup and Data Generation
# =====================================

# Load required libraries
# List of required packages
required_packages <- c("mice", "VIM", "ggplot2", "dplyr", "tidyr")

# Function to check and install packages
check_and_install <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Check and install each package
invisible(sapply(required_packages, check_and_install))

# Set seed for reproducibility
set.seed(123)

# Create a sample dataset with missing values
n <- 1000
data <- data.frame(
  age = rnorm(n, mean = 40, sd = 10),
  income = rnorm(n, mean = 50000, sd = 15000),
  education = sample(c("High School", "Bachelor's", "Master's", "PhD"), n, replace = TRUE),
  satisfaction = rnorm(n, mean = 7, sd = 1.5)
)

# Introduce missing values
# We're artificially creating missing data to simulate real-world scenarios
data$age[sample(1:n, 100)] <- NA
data$income[sample(1:n, 150)] <- NA
data$education[sample(1:n, 80)] <- NA
data$satisfaction[sample(1:n, 120)] <- NA

# Section 2: Missing Data Exploration
# ===================================

# Explore missing data patterns
# This helps understand the structure of missingness in the data
missing_pattern <- md.pattern(data, plot = TRUE)
print(missing_pattern)

# Visualize missing data
# This plot shows the pattern of missing data across variables
png("missing_data_plot.png", width = 800, height = 600)
aggr(data, col = c('navyblue', 'red'), numbers = TRUE, sortVars = TRUE,
     labels = names(data), cex.axis = .7, gap = 3,
     ylab = c("Missing data", "Pattern"))
dev.off()

# Section 3: Simple Imputation Methods
# ====================================

# 3.1 Mean imputation for numeric variables
# This is a simple method where missing values are replaced by the mean of the variable
data_mean_imputed <- data
data_mean_imputed$age[is.na(data_mean_imputed$age)] <- mean(data_mean_imputed$age, na.rm = TRUE)
data_mean_imputed$income[is.na(data_mean_imputed$income)] <- mean(data_mean_imputed$income, na.rm = TRUE)
data_mean_imputed$satisfaction[is.na(data_mean_imputed$satisfaction)] <- mean(data_mean_imputed$satisfaction, na.rm = TRUE)

# 3.2 Mode imputation for categorical variables
# For categorical variables, we use the most frequent category to impute missing values
mode_impute <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

data_mean_imputed$education[is.na(data_mean_imputed$education)] <- mode_impute(data_mean_imputed$education)

# Section 4: Multiple Imputation using MICE
# =========================================

# MICE (Multivariate Imputation by Chained Equations) is a more sophisticated method
# It creates multiple imputations for multivariate missing data
imp <- mice(data, m = 5, maxit = 50, method = 'pmm', seed = 500)
data_mice <- complete(imp)

# Section 5: Visualization of Imputed Data
# ========================================

# 5.1 Density plot for age
# This plot compares the distribution of age before and after imputation
png("age_density_plot.png", width = 800, height = 600)
ggplot() +
  geom_density(data = data, aes(x = age), color = "red", alpha = 0.5) +
  geom_density(data = data_mice, aes(x = age), color = "blue", alpha = 0.5) +
  labs(title = "Density Plot: Original vs Imputed Age",
       x = "Age", y = "Density") +
  theme_minimal()
dev.off()

# 5.2 Boxplot for income
# This plot compares the distribution of income before and after imputation
png("income_boxplot.png", width = 800, height = 600)
ggplot() +
  geom_boxplot(data = data, aes(x = "Original", y = income), fill = "red", alpha = 0.5) +
  geom_boxplot(data = data_mice, aes(x = "Imputed", y = income), fill = "blue", alpha = 0.5) +
  labs(title = "Boxplot: Original vs Imputed Income",
       x = "", y = "Income") +
  theme_minimal()
dev.off()

# 5.3 Bar plot for education
# This plot compares the distribution of education levels before and after imputation
data_long <- data %>%
  mutate(source = "Original") %>%
  bind_rows(data_mice %>% mutate(source = "Imputed")) %>%
  group_by(source, education) %>%
  summarise(count = n(), .groups = "drop")

png("education_barplot.png", width = 800, height = 600)
ggplot(data_long, aes(x = education, y = count, fill = source)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Bar Plot: Original vs Imputed Education Levels",
       x = "Education", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
dev.off()

# Section 6: Imputation Quality Evaluation
# ========================================

# 6.1 Compare means and standard deviations
# This helps assess if the imputation preserved the overall distribution of the data
summary_stats <- data.frame(
  Variable = c("Age", "Income", "Satisfaction"),
  Original_Mean = c(mean(data$age, na.rm = TRUE), mean(data$income, na.rm = TRUE), mean(data$satisfaction, na.rm = TRUE)),
  Imputed_Mean = c(mean(data_mice$age), mean(data_mice$income), mean(data_mice$satisfaction)),
  Original_SD = c(sd(data$age, na.rm = TRUE), sd(data$income, na.rm = TRUE), sd(data$satisfaction, na.rm = TRUE)),
  Imputed_SD = c(sd(data_mice$age), sd(data_mice$income), sd(data_mice$satisfaction))
)

print(summary_stats)

# 6.2 Compare correlations
# This helps assess if the imputation preserved the relationships between variables
cor_original <- cor(data[c("age", "income", "satisfaction")], use = "pairwise.complete.obs")
cor_imputed <- cor(data_mice[c("age", "income", "satisfaction")])

print("Original correlation matrix:")
print(cor_original)
print("Imputed correlation matrix:")
print(cor_imputed)

# Section 7: Sensitivity Analysis
# ===============================

# 7.1 Compare different imputation methods
# This helps assess how sensitive the results are to the choice of imputation method
imp_pmm <- mice(data, m = 5, maxit = 50, method = 'pmm', seed = 500)
imp_norm <- mice(data, m = 5, maxit = 50, method = 'norm', seed = 500)
imp_cart <- mice(data, m = 5, maxit = 50, method = 'cart', seed = 500)

data_pmm <- complete(imp_pmm)
data_norm <- complete(imp_norm)
data_cart <- complete(imp_cart)

# 7.2 Compare means and standard deviations across different methods
sensitivity_stats <- data.frame(
  Variable = c("Age", "Income", "Satisfaction"),
  Original_Mean = c(mean(data$age, na.rm = TRUE), mean(data$income, na.rm = TRUE), mean(data$satisfaction, na.rm = TRUE)),
  Original_SD = c(sd(data$age, na.rm = TRUE), sd(data$income, na.rm = TRUE), sd(data$satisfaction, na.rm = TRUE)),
  PMM_Mean = c(mean(data_pmm$age), mean(data_pmm$income), mean(data_pmm$satisfaction)),
  PMM_SD = c(sd(data_pmm$age), sd(data_pmm$income), sd(data_pmm$satisfaction)),
  Norm_Mean = c(mean(data_norm$age), mean(data_norm$income), mean(data_norm$satisfaction)),
  Norm_SD = c(sd(data_norm$age), sd(data_norm$income), sd(data_norm$satisfaction)),
  CART_Mean = c(mean(data_cart$age), mean(data_cart$income), mean(data_cart$satisfaction)),
  CART_SD = c(sd(data_cart$age), sd(data_cart$income), sd(data_cart$satisfaction))
)

print(sensitivity_stats)

# 7.3 Compare correlations across different methods
cor_pmm <- cor(data_pmm[c("age", "income", "satisfaction")])
cor_norm <- cor(data_norm[c("age", "income", "satisfaction")])
cor_cart <- cor(data_cart[c("age", "income", "satisfaction")])

print("PMM correlation matrix:")
print(cor_pmm)
print("Norm correlation matrix:")
print(cor_norm)
print("CART correlation matrix:")
print(cor_cart)

# 7.4 Visualize differences in imputation methods
# Create a long format dataset for plotting
data_long <- rbind(
  data.frame(Method = "Original", Variable = "Age", Value = data$age),
  data.frame(Method = "PMM", Variable = "Age", Value = data_pmm$age),
  data.frame(Method = "Norm", Variable = "Age", Value = data_norm$age),
  data.frame(Method = "CART", Variable = "Age", Value = data_cart$age),
  data.frame(Method = "Original", Variable = "Income", Value = data$income),
  data.frame(Method = "PMM", Variable = "Income", Value = data_pmm$income),
  data.frame(Method = "Norm", Variable = "Income", Value = data_norm$income),
  data.frame(Method = "CART", Variable = "Income", Value = data_cart$income),
  data.frame(Method = "Original", Variable = "Satisfaction", Value = data$satisfaction),
  data.frame(Method = "PMM", Variable = "Satisfaction", Value = data_pmm$satisfaction),
  data.frame(Method = "Norm", Variable = "Satisfaction", Value = data_norm$satisfaction),
  data.frame(Method = "CART", Variable = "Satisfaction", Value = data_cart$satisfaction)
)

# Create density plots for each variable
png("imputation_methods_comparison.png", width = 1200, height = 800)
ggplot(data_long, aes(x = Value, color = Method)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ Variable, scales = "free") +
  labs(title = "Comparison of Imputation Methods",
       x = "Value", y = "Density") +
  theme_minimal()
dev.off()

# 7.5 Analyze the impact of imputation on a simple linear model
# We'll use age to predict income in each imputed dataset

# Function to fit model and extract coefficients
fit_model <- function(data) {
  model <- lm(income ~ age, data = data)
  return(coef(model))
}

# Fit models
coef_original <- fit_model(data)
coef_pmm <- fit_model(data_pmm)
coef_norm <- fit_model(data_norm)
coef_cart <- fit_model(data_cart)

# Compare coefficients
coef_comparison <- rbind(
  Original = coef_original,
  PMM = coef_pmm,
  Norm = coef_norm,
  CART = coef_cart
)

print("Comparison of regression coefficients:")
print(coef_comparison)

# Section 8: Data Export and Reporting
# ====================================

# Save imputed dataset
write.csv(data_mice, "imputed_data.csv", row.names = FALSE)

# Final report
cat("
Imputation Report:
1. Missing data patterns and visualizations have been saved as PNG files.
2. Multiple imputation using MICE (Predictive Mean Matching) was performed.
3. Imputed data visualizations (density plot, boxplot, and bar plot) have been saved as PNG files.
4. Summary statistics and correlation matrices for original and imputed data have been printed.
5. Sensitivity analysis comparing different imputation methods has been conducted:
   - Compared means and standard deviations across methods (PMM, Norm, CART)
   - Compared correlation matrices across methods
   - Visualized differences in imputation methods (saved as 'imputation_methods_comparison.png')
   - Analyzed the impact of imputation on a simple linear model (age predicting income)
6. The final imputed dataset has been saved as 'imputed_data.csv'.

Key findings from sensitivity analysis:
- [Insert key observations about differences between imputation methods]
- [Comment on which method seems most appropriate for this dataset and why]
- [Discuss any potential biases or limitations introduced by the imputation methods]

Please review the generated plots and printed statistics to assess the quality of the imputation.
")