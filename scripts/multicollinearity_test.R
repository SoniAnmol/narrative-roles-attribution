# Multicollinearity Diagnostics
# This script tests for multicollinearity among demographic predictors

library(MASS)
library(dplyr)
library(car)

# Load data
data <- read.csv("./data/subset_for_regression.csv", stringsAsFactors = FALSE)

# ============================================================================
# DATA CLEANING: REMOVE RESPONDENTS UNDER 18 YEARS OLD
# ============================================================================

cat("Total respondents before filtering:", nrow(data), "\n")
cat("Respondents under 18 (age = 1):", sum(data$age == 1, na.rm = TRUE), "\n")

# Remove respondents with age = 1 (under 18)
data <- data %>% filter(age != 1)

cat("Total respondents after removing under 18:", nrow(data), "\n\n")

# Data preparation
data$flooded <- factor(data$flooded, levels = c("False", "True"))
data$subjective_attribution <- factor(data$subjective_attribution, levels = 1:5, ordered = TRUE)
data$gender <- as.factor(data$gender)
# Age factor excluding category 1 (under 18, already filtered out)
data$age <- factor(data$age, levels = c(2, 3, 4, 5, 6, 7, 8))
data$income <- as.factor(data$income)
data$source_of_info <- as.factor(data$source_of_info)

cat("\n")
cat(strrep("=", 80), "\n")
cat("MULTICOLLINEARITY DIAGNOSTICS\n")
cat(strrep("=", 80), "\n\n")

# ============================================================================
# 1. VARIANCE INFLATION FACTOR (VIF)
# ============================================================================

cat("--- Variance Inflation Factors (VIF) for Demographics ---\n")
cat("Testing: flooded, gender, age, income, source_of_info\n\n")

# Create a linear model (VIF doesn't depend on outcome, we use numeric proxy)
vif_model <- lm(as.numeric(subjective_attribution) ~ flooded + gender + age + income + source_of_info, 
                data = data)

# Calculate VIF/GVIF
vif_results <- vif(vif_model)
print(vif_results)

cat("\n--- Interpretation ---\n")
cat("For binary/simple variables: VIF values\n")
cat("  VIF < 5: No multicollinearity concern\n")
cat("  VIF 5-10: Moderate multicollinearity\n")
cat("  VIF > 10: High multicollinearity (problematic)\n\n")
cat("For multi-category variables: GVIF^(1/(2*Df)) values\n")
cat("  GVIF^(1/(2*Df)) < 2: No multicollinearity concern\n")
cat("  GVIF^(1/(2*Df)) > 2: Potential multicollinearity concern\n\n")

# ============================================================================
# 2. CRAMÉR'S V (Association between categorical variables)
# ============================================================================

cat("\n")
cat("--- Cramér's V Matrix (Pairwise Associations) ---\n\n")

# Function to calculate Cramér's V
cramers_v <- function(x, y) {
  tbl <- table(x, y)
  # Remove empty rows/columns
  tbl <- tbl[rowSums(tbl) > 0, colSums(tbl) > 0, drop = FALSE]
  
  # Check if table has sufficient data
  if (min(dim(tbl)) < 2 || sum(tbl) < 5) {
    return(NA)
  }
  
  chi2 <- tryCatch({
    suppressWarnings(chisq.test(tbl, correct = FALSE)$statistic)
  }, error = function(e) {
    return(NA)
  })
  
  if (is.na(chi2)) return(NA)
  
  n <- sum(tbl)
  min_dim <- min(dim(tbl)) - 1
  if (min_dim == 0) return(0)
  v <- sqrt(chi2 / (n * min_dim))
  return(as.numeric(v))
}

# Calculate Cramér's V for all pairs
predictors <- c("flooded", "gender", "age", "income", "source_of_info")
cramers_matrix <- matrix(NA, nrow = length(predictors), ncol = length(predictors))
rownames(cramers_matrix) <- predictors
colnames(cramers_matrix) <- predictors

for (i in 1:length(predictors)) {
  for (j in 1:length(predictors)) {
    if (i == j) {
      cramers_matrix[i, j] <- 1.0
    } else if (i < j) {
      cramers_matrix[i, j] <- cramers_v(data[[predictors[i]]], data[[predictors[j]]])
      cramers_matrix[j, i] <- cramers_matrix[i, j]
    }
  }
}

print(round(cramers_matrix, 3))

cat("\n--- Interpretation ---\n")
cat("Cramér's V ranges from 0 (no association) to 1 (perfect association)\n")
cat("  V < 0.10: Weak/negligible association\n")
cat("  V 0.10-0.30: Moderate association\n")
cat("  V > 0.30: Strong association (potential multicollinearity concern)\n\n")

# ============================================================================
# 3. IDENTIFY PROBLEMATIC PAIRS
# ============================================================================

cat("\n--- High Association Pairs (Cramér's V > 0.30) ---\n")
high_pairs <- which(cramers_matrix > 0.30 & cramers_matrix < 1, arr.ind = TRUE)
if (nrow(high_pairs) > 0) {
  high_pairs <- high_pairs[high_pairs[,1] < high_pairs[,2], ]
  if (is.matrix(high_pairs) && nrow(high_pairs) > 0) {
    for (i in 1:nrow(high_pairs)) {
      var1 <- predictors[high_pairs[i, 1]]
      var2 <- predictors[high_pairs[i, 2]]
      value <- cramers_matrix[high_pairs[i, 1], high_pairs[i, 2]]
      cat(sprintf("%s <-> %s: %.3f\n", var1, var2, value))
    }
  } else {
    cat("None found\n")
  }
} else {
  cat("None found\n")
}

cat("\n")
cat(strrep("=", 80), "\n")
cat("DIAGNOSTICS COMPLETE\n")
cat(strrep("=", 80), "\n\n")

# ============================================================================
# ADDITIONAL TEST FOR MODEL 7b
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("MULTICOLLINEARITY DIAGNOSTICS FOR MODEL 7b\n")
cat("(Model with subjective_attribution as predictor)\n")
cat(strrep("=", 80), "\n\n")

# Prepare data for Model 7b (already filtered for age, just need to add reported_mitigation_action)
data$reported_mitigation_action <- factor(data$reported_mitigation_action, 
                                         levels = 1:5, 
                                         ordered = TRUE)

cat("Sample size for Model 7b:", nrow(data), "\n")
cat("--- Variance Inflation Factors (VIF) for Model 7b ---\n")
cat("Testing: subjective_attribution, flooded, source_of_info, gender, age, income\n\n")

# VIF model for Model 7b
vif_model_7b <- lm(as.numeric(reported_mitigation_action) ~ subjective_attribution + flooded + source_of_info + gender + age + income, 
                   data = data)

vif_results_7b <- vif(vif_model_7b)
print(vif_results_7b)

cat("\n--- Interpretation ---\n")
cat("For binary/simple variables: VIF values\n")
cat("  VIF < 5: No multicollinearity concern\n")
cat("  VIF 5-10: Moderate multicollinearity\n")
cat("  VIF > 10: High multicollinearity (problematic)\n\n")
cat("For multi-category variables: GVIF^(1/(2*Df)) values\n")
cat("  GVIF^(1/(2*Df)) < 2: No multicollinearity concern\n")
cat("  GVIF^(1/(2*Df)) > 2: Potential multicollinearity concern\n\n")

# Cramér's V for Model 7b
cat("\n--- Cramér's V Matrix for Model 7b ---\n\n")

predictors_7b <- c("subjective_attribution", "flooded", "gender", "age", "income", "source_of_info")
cramers_matrix_7b <- matrix(NA, nrow = length(predictors_7b), ncol = length(predictors_7b))
rownames(cramers_matrix_7b) <- predictors_7b
colnames(cramers_matrix_7b) <- predictors_7b

for (i in 1:length(predictors_7b)) {
  for (j in 1:length(predictors_7b)) {
    if (i == j) {
      cramers_matrix_7b[i, j] <- 1.0
    } else if (i < j) {
      cramers_matrix_7b[i, j] <- cramers_v(data[[predictors_7b[i]]], data[[predictors_7b[j]]])
      cramers_matrix_7b[j, i] <- cramers_matrix_7b[i, j]
    }
  }
}

print(round(cramers_matrix_7b, 3))

cat("\n--- High Association Pairs (Cramér's V > 0.30) ---\n")
high_pairs_7b <- which(cramers_matrix_7b > 0.30 & cramers_matrix_7b < 1, arr.ind = TRUE)
if (nrow(high_pairs_7b) > 0) {
  high_pairs_7b <- high_pairs_7b[high_pairs_7b[,1] < high_pairs_7b[,2], ]
  if (is.matrix(high_pairs_7b) && nrow(high_pairs_7b) > 0) {
    for (i in 1:nrow(high_pairs_7b)) {
      var1 <- predictors_7b[high_pairs_7b[i, 1]]
      var2 <- predictors_7b[high_pairs_7b[i, 2]]
      value <- cramers_matrix_7b[high_pairs_7b[i, 1], high_pairs_7b[i, 2]]
      cat(sprintf("%s <-> %s: %.3f\n", var1, var2, value))
    }
  } else {
    cat("None found\n")
  }
} else {
  cat("None found\n")
}

cat("\n")
cat(strrep("=", 80), "\n")
cat("MODEL 7b DIAGNOSTICS COMPLETE\n")
cat(strrep("=", 80), "\n")
