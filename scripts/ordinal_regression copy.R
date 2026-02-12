# Ordinal Regression Analysis
# Comparing flooded vs unflooded municipalities on subjective attribution and mitigation actions

# Load required libraries
library(MASS)        # for polr (proportional odds logistic regression)
library(broom)       # for tidy model outputs
library(dplyr)       # for data manipulation
library(tidyr)       # for data manipulation

# Load data
data <- read.csv("./data/subset_for_regression.csv", stringsAsFactors = FALSE)

# ============================================================================
# DATA CLEANING: REMOVE RESPONDENTS UNDER 18 YEARS OLD
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("DATA CLEANING: AGE RESTRICTION\n")
cat(strrep("=", 80), "\n\n")

cat("Total respondents before filtering:", nrow(data), "\n")
cat("Respondents under 18 (age = 1):", sum(data$age == 1, na.rm = TRUE), "\n")

# Remove respondents with age = 1 (under 18)
data <- data %>% filter(age != 1)

cat("Total respondents after removing under 18:", nrow(data), "\n")
cat(strrep("=", 80), "\n\n")

# Data preparation
# Convert flooded to factor
data$flooded <- factor(data$flooded, levels = c("False", "True"))

# Convert ordinal outcomes to ordered factors
data$subjective_attribution <- factor(data$subjective_attribution, 
                                      levels = 1:5, 
                                      ordered = TRUE)
data$reported_mitigation_action <- factor(data$reported_mitigation_action, 
                                          levels = 1:5, 
                                          ordered = TRUE)

# Convert categorical controls to factors
data$gender <- as.factor(data$gender)
# Age factor excluding category 1 (under 18, already filtered out)
data$age <- factor(data$age, levels = c(2, 3, 4, 5, 6, 7, 8))
data$occupation <- as.factor(data$occupation)
data$income <- as.factor(data$income)
data$source_of_info <- as.factor(data$source_of_info)

# Create new column for Model 7: source of information (newspaper preferred over tv_channel)
# Prioritize newspaper if available, otherwise use tv_channel, drop if both NA
data$preferred_source <- NA

# Map newspaper values with descriptive names
newspaper_map <- c('1' = 'Il Corriere della Sera', 
                   '2' = 'LaRepiblica', 
                   '3' = 'IlSole24', 
                   '4' = 'Il resto del carlino', 
                   '5' = 'other')

# Map tv_channel values with descriptive names
tv_channel_map <- c('1' = 'Rai 1 o Rai 2', 
                    '2' = 'Mediaset', 
                    '3' = 'La7', 
                    '4' = 'Rai3', 
                    '5' = 'SkyTG24', 
                    '6' = 'Other')

# First, map newspaper values where newspaper is not NA
data$preferred_source[!is.na(data$newspaper)] <- newspaper_map[as.character(data$newspaper[!is.na(data$newspaper)])]

# Then, map tv_channel values where newspaper is NA but tv_channel is not NA
data$preferred_source[is.na(data$preferred_source) & !is.na(data$tv_channel)] <- 
  tv_channel_map[as.character(data$tv_channel[is.na(data$preferred_source) & !is.na(data$tv_channel)])]

# Create dataset for Model 4a/4b (dropping rows where both newspaper and tv_channel are NA)
data_model7 <- data %>% filter(!is.na(preferred_source))
data_model7$preferred_source <- as.factor(data_model7$preferred_source)

# ============================================================================
# ANALYSIS FOR SUBJECTIVE ATTRIBUTION
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("ORDINAL REGRESSION MODELS: SUBJECTIVE ATTRIBUTION\n")
cat(strrep("=", 80), "\n\n")

# MODEL 1a: Treatment + Demographics
cat("\n--- MODEL 1a: Treatment + Demographics ---\n")
model2_subj <- polr(subjective_attribution ~ flooded + gender + age + occupation + income, 
                    data = data, 
                    Hess = TRUE)
summary(model2_subj)

ctable2_subj <- coef(summary(model2_subj))
p2_subj <- pnorm(abs(ctable2_subj[, "t value"]), lower.tail = FALSE) * 2
ctable2_subj <- cbind(ctable2_subj, "p value" = p2_subj)
print(ctable2_subj)

# MODEL 2a: Treatment + Demographics + Source of Information
cat("\n--- MODEL 2a: Treatment + Demographics + Source of Information ---\n")
model3_subj <- polr(subjective_attribution ~ flooded + gender + age + occupation + income + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model3_subj)

ctable3_subj <- coef(summary(model3_subj))
p3_subj <- pnorm(abs(ctable3_subj[, "t value"]), lower.tail = FALSE) * 2
ctable3_subj <- cbind(ctable3_subj, "p value" = p3_subj)
print(ctable3_subj)

# MODEL 3a: Newspaper subset
cat("\n--- MODEL 3a: Newspaper subset (newspaper only) ---\n")
data_newspaper <- data %>% filter(!is.na(newspaper))
model4a1_subj <- polr(subjective_attribution ~ flooded + newspaper, 
                      data = data_newspaper, 
                      Hess = TRUE)
summary(model4a1_subj)

ctable4a1_subj <- coef(summary(model4a1_subj))
p4a1_subj <- pnorm(abs(ctable4a1_subj[, "t value"]), lower.tail = FALSE) * 2
ctable4a1_subj <- cbind(ctable4a1_subj, "p value" = p4a1_subj)
print(ctable4a1_subj)

# MODEL 4a: Treatment + Preferred Source (newspaper or tv_channel combined)
cat("\n--- MODEL 4a: Treatment + Preferred Source (newspaper or tv_channel) ---\n")
model7_subj <- polr(subjective_attribution ~ flooded + preferred_source, 
                    data = data_model7, 
                    Hess = TRUE)
summary(model7_subj)

ctable7_subj <- coef(summary(model7_subj))
p7_subj <- pnorm(abs(ctable7_subj[, "t value"]), lower.tail = FALSE) * 2
ctable7_subj <- cbind(ctable7_subj, "p value" = p7_subj)
print(ctable7_subj)

# MODEL 5a: Treatment + Source of Information
# Report N for Model 5a
cat("N for Model 5a: ", nrow(data), "\n")
cat("\n--- MODEL 5a: Treatment + Source of Information ---\n")
model8_subj <- polr(subjective_attribution ~ flooded + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model8_subj)

ctable8_subj <- coef(summary(model8_subj))
p8_subj <- pnorm(abs(ctable8_subj[, "t value"]), lower.tail = FALSE) * 2
ctable8_subj <- cbind(ctable8_subj, "p value" = p8_subj)
print(ctable8_subj)

# MODEL 6a: Treatment + Source of Information + Demographics (No Occupation)
cat("\n--- MODEL 6a: Treatment + Source of Information + Demographics (No Occupation) ---\n")

# Report N for Model 6a
cat("N for Model 6a: ", nrow(data), "\n")

model8c_subj <- polr(subjective_attribution ~ flooded + source_of_info + gender + age + income, 
                     data = data, 
                     Hess = TRUE)
summary(model8c_subj)

ctable8c_subj <- coef(summary(model8c_subj))
p8c_subj <- pnorm(abs(ctable8c_subj[, "t value"]), lower.tail = FALSE) * 2
ctable8c_subj <- cbind(ctable8c_subj, "p value" = p8c_subj)
print(ctable8c_subj)

# ============================================================================
# ANALYSIS FOR REPORTED MITIGATION ACTION
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("ORDINAL REGRESSION MODELS: REPORTED MITIGATION ACTION\n")
cat(strrep("=", 80), "\n\n")

# MODEL 1b: Treatment + Demographics
cat("\n--- MODEL 1b: Treatment + Demographics ---\n")
model2_mit <- polr(reported_mitigation_action ~ flooded + gender + age + occupation + income, 
                   data = data, 
                   Hess = TRUE)
summary(model2_mit)

ctable2_mit <- coef(summary(model2_mit))
p2_mit <- pnorm(abs(ctable2_mit[, "t value"]), lower.tail = FALSE) * 2
ctable2_mit <- cbind(ctable2_mit, "p value" = p2_mit)
print(ctable2_mit)

# MODEL 2b: Treatment + Demographics + Source of Information
cat("\n--- MODEL 2b: Treatment + Demographics + Source of Information ---\n")
model3_mit <- polr(reported_mitigation_action ~ flooded + gender + age + occupation + income + source_of_info, 
                   data = data, 
                   Hess = TRUE)
summary(model3_mit)

ctable3_mit <- coef(summary(model3_mit))
p3_mit <- pnorm(abs(ctable3_mit[, "t value"]), lower.tail = FALSE) * 2
ctable3_mit <- cbind(ctable3_mit, "p value" = p3_mit)
print(ctable3_mit)

# MODEL 3b: Newspaper subset
cat("\n--- MODEL 3b: Newspaper subset (newspaper only) ---\n")
model4a1_mit <- polr(reported_mitigation_action ~ flooded + newspaper, 
                     data = data_newspaper, 
                     Hess = TRUE)
summary(model4a1_mit)

ctable4a1_mit <- coef(summary(model4a1_mit))
p4a1_mit <- pnorm(abs(ctable4a1_mit[, "t value"]), lower.tail = FALSE) * 2
ctable4a1_mit <- cbind(ctable4a1_mit, "p value" = p4a1_mit)
print(ctable4a1_mit)


# MODEL 4b: Treatment + Preferred Source (newspaper or tv_channel combined)
cat("\n--- MODEL 4b: Treatment + Preferred Source (newspaper or tv_channel) ---\n")
model7b_mit <- polr(reported_mitigation_action ~ flooded + preferred_source, 
                    data = data_model7, 
                    Hess = TRUE)
summary(model7b_mit)

ctable7b_mit <- coef(summary(model7b_mit))
p7b_mit <- pnorm(abs(ctable7b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable7b_mit <- cbind(ctable7b_mit, "p value" = p7b_mit)
print(ctable7b_mit)

# MODEL 5b: Treatment + Source of Information
cat("\n--- MODEL 5b: Treatment + Source of Information ---\n")
model8b_mit <- polr(reported_mitigation_action ~ flooded + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model8b_mit)

ctable8b_mit <- coef(summary(model8b_mit))
p8b_mit <- pnorm(abs(ctable8b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable8b_mit <- cbind(ctable8b_mit, "p value" = p8b_mit)
print(ctable8b_mit)

# MODEL 6b: Treatment + Source of Information + Demographics (No Occupation)
cat("\n--- MODEL 6b: Treatment + Source of Information + Demographics (No Occupation) ---\n")
model8m_mit <- polr(reported_mitigation_action ~ flooded + source_of_info + gender + age + income, 
                    data = data, 
                    Hess = TRUE)
summary(model8m_mit)

ctable8m_mit <- coef(summary(model8m_mit))
p8m_mit <- pnorm(abs(ctable8m_mit[, "t value"]), lower.tail = FALSE) * 2
ctable8m_mit <- cbind(ctable8m_mit, "p value" = p8m_mit)
print(ctable8m_mit)



# MODEL 7b: Attribution + Treatment + Source of Information + Demographics
cat("\n--- MODEL 7b: Attribution + Treatment + Source of Information + Demographics ---\n")
cat("N for Model 7b: ", nrow(data), "\n")
model10_mit <- polr(reported_mitigation_action ~ subjective_attribution + flooded + source_of_info + gender + age + income, 
                    data = data, 
                    Hess = TRUE)
summary(model10_mit)

ctable10_mit <- coef(summary(model10_mit))
p10_mit <- pnorm(abs(ctable10_mit[, "t value"]), lower.tail = FALSE) * 2
ctable10_mit <- cbind(ctable10_mit, "p value" = p10_mit)
print(ctable10_mit)


# ============================================================================
# MODEL COMPARISON (AIC/BIC)
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("MODEL COMPARISON (AIC/BIC)\n")
cat(strrep("=", 80), "\n\n")

cat("\nSubjective Attribution Models:\n")
cat("Model 1a (flooded + demographics) AIC:", AIC(model2_subj), "BIC:", BIC(model2_subj), "\n")
cat("Model 2a (flooded + demographics + source_of_info) AIC:", AIC(model3_subj), "BIC:", BIC(model3_subj), "\n")
cat("Model 3a (flooded + newspaper) AIC:", AIC(model4a1_subj), "BIC:", BIC(model4a1_subj), "\n")
cat("Model 4a (flooded + preferred_source) AIC:", AIC(model7_subj), "BIC:", BIC(model7_subj), "\n")
cat("Model 5a (flooded + source_of_info) AIC:", AIC(model8_subj), "BIC:", BIC(model8_subj), "\n")
cat("Model 6a (flooded + source_of_info + demographics) AIC:", AIC(model8c_subj), "BIC:", BIC(model8c_subj), "\n")

cat("\nReported Mitigation Action Models:\n")
cat("Model 1b (flooded + demographics) AIC:", AIC(model2_mit), "BIC:", BIC(model2_mit), "\n")
cat("Model 2b (flooded + demographics + source_of_info) AIC:", AIC(model3_mit), "BIC:", BIC(model3_mit), "\n")
cat("Model 3b (flooded + newspaper) AIC:", AIC(model4a1_mit), "BIC:", BIC(model4a1_mit), "\n")
cat("Model 4b (flooded + preferred_source) AIC:", AIC(model7b_mit), "BIC:", BIC(model7b_mit), "\n")
cat("Model 5b (flooded + source_of_info) AIC:", AIC(model8b_mit), "BIC:", BIC(model8b_mit), "\n")
cat("Model 6b (flooded + source_of_info + demographics) AIC:", AIC(model8m_mit), "BIC:", BIC(model8m_mit), "\n")
cat("Model 7b (attribution + flooded + source_of_info + demographics) AIC:", AIC(model10_mit), "BIC:", BIC(model10_mit), "\n")


# ============================================================================
# SAVE MODEL RESULTS TO CSV
# ============================================================================

# Function to extract coefficient table with p-values
extract_coef_table <- function(model, model_name, outcome) {
  ctable <- coef(summary(model))
  p_values <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
  result <- as.data.frame(cbind(ctable, "p value" = p_values))
  result$model <- model_name
  result$outcome <- outcome
  result$coefficient <- rownames(result)
  return(result)
}

# Combine all results
all_results_subj <- rbind(
  extract_coef_table(model2_subj, "Model 1a (flooded+demographics)", "subjective_attribution"),
  extract_coef_table(model3_subj, "Model 2a (flooded+demo+source_of_info)", "subjective_attribution"),
  extract_coef_table(model4a1_subj, "Model 3a (flooded+newspaper)", "subjective_attribution"),
  extract_coef_table(model7_subj, "Model 4a (flooded+preferred_source)", "subjective_attribution"),
  extract_coef_table(model8_subj, "Model 5a (flooded+source_of_info)", "subjective_attribution"),
  extract_coef_table(model8c_subj, "Model 6a (flooded+source_of_info+demo)", "subjective_attribution")
)

all_results_mit <- rbind(
  extract_coef_table(model2_mit, "Model 1b (flooded+demographics)", "reported_mitigation_action"),
  extract_coef_table(model3_mit, "Model 2b (flooded+demo+source_of_info)", "reported_mitigation_action"),
  extract_coef_table(model4a1_mit, "Model 3b (flooded+newspaper)", "reported_mitigation_action"),
  extract_coef_table(model7b_mit, "Model 4b (flooded+preferred_source)", "reported_mitigation_action"),
  extract_coef_table(model8b_mit, "Model 5b (flooded+source_of_info)", "reported_mitigation_action"),
  extract_coef_table(model8m_mit, "Model 6b (flooded+source_of_info+demo)", "reported_mitigation_action"),
  extract_coef_table(model10_mit, "Model 7b (attribution+flooded+source_of_info+demo)", "reported_mitigation_action")
)

all_results <- rbind(all_results_subj, all_results_mit)

# Save to CSV
write.csv(all_results, "./ordinal_regression_results.csv", row.names = FALSE)

cat("\n\nResults saved to: ordinal_regression_results.csv\n")

cat("\n")
cat(strrep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 80), "\n")
