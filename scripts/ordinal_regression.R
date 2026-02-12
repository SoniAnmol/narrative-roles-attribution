# Ordinal Regression Analysis
# Comparing flooded vs unflooded municipalities on subjective attribution and mitigation actions

# Load required libraries
library(MASS)        # for polr (proportional odds logistic regression)
library(broom)       # for tidy model outputs
library(dplyr)       # for data manipulation
library(tidyr)       # for data manipulation

# Load data
data <- read.csv("./data/subset_for_regression.csv", stringsAsFactors = FALSE)

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
data$age <- as.factor(data$age)
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

# Create dataset for Model 7 (dropping rows where both newspaper and tv_channel are NA)
data_model7 <- data %>% filter(!is.na(preferred_source))
data_model7$preferred_source <- as.factor(data_model7$preferred_source)

# Create dataset for Model 9: subset with high climate attribution (subjective_attribution 4 or 5)
data_model9 <- data %>% filter(subjective_attribution %in% c('4', '5'))

# Create dataset for Model 9a: high attribution subset with preferred_source available
data_model9a <- data_model9 %>% filter(!is.na(preferred_source))
data_model9a$preferred_source <- as.factor(data_model9a$preferred_source)

# ============================================================================
# ANALYSIS FOR SUBJECTIVE ATTRIBUTION
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("ORDINAL REGRESSION MODELS: SUBJECTIVE ATTRIBUTION\n")
cat(strrep("=", 80), "\n\n")

# MODEL 1: No controls (just treatment)
cat("\n--- MODEL 1: Treatment only ---\n")
model1_subj <- polr(subjective_attribution ~ flooded, 
                    data = data, 
                    Hess = TRUE)
summary(model1_subj)

# Calculate p-values
ctable1_subj <- coef(summary(model1_subj))
p1_subj <- pnorm(abs(ctable1_subj[, "t value"]), lower.tail = FALSE) * 2
ctable1_subj <- cbind(ctable1_subj, "p value" = p1_subj)
print(ctable1_subj)

# MODEL 2: Treatment + Demographics
cat("\n--- MODEL 2: Treatment + Demographics ---\n")
model2_subj <- polr(subjective_attribution ~ flooded + gender + age + occupation + income, 
                    data = data, 
                    Hess = TRUE)
summary(model2_subj)

ctable2_subj <- coef(summary(model2_subj))
p2_subj <- pnorm(abs(ctable2_subj[, "t value"]), lower.tail = FALSE) * 2
ctable2_subj <- cbind(ctable2_subj, "p value" = p2_subj)
print(ctable2_subj)

# MODEL 3: Treatment + Demographics + Source of Information
cat("\n--- MODEL 3: Treatment + Demographics + Source of Information ---\n")
model3_subj <- polr(subjective_attribution ~ flooded + gender + age + occupation + income + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model3_subj)

ctable3_subj <- coef(summary(model3_subj))
p3_subj <- pnorm(abs(ctable3_subj[, "t value"]), lower.tail = FALSE) * 2
ctable3_subj <- cbind(ctable3_subj, "p value" = p3_subj)
print(ctable3_subj)

# MODEL 4a: Newspaper subset
cat("\n--- MODEL 4a: Newspaper subset (newspaper only) ---\n")
data_newspaper <- data %>% filter(!is.na(newspaper))
model4a1_subj <- polr(subjective_attribution ~ flooded + newspaper, 
                      data = data_newspaper, 
                      Hess = TRUE)
summary(model4a1_subj)

ctable4a1_subj <- coef(summary(model4a1_subj))
p4a1_subj <- pnorm(abs(ctable4a1_subj[, "t value"]), lower.tail = FALSE) * 2
ctable4a1_subj <- cbind(ctable4a1_subj, "p value" = p4a1_subj)
print(ctable4a1_subj)

# MODEL 4b: TV Channel subset
cat("\n--- MODEL 4b: TV Channel subset (tv_channel only) ---\n")
data_tv <- data %>% filter(!is.na(tv_channel))
model4b1_subj <- polr(subjective_attribution ~ flooded + tv_channel, 
                      data = data_tv, 
                      Hess = TRUE)
summary(model4b1_subj)

ctable4b1_subj <- coef(summary(model4b1_subj))
p4b1_subj <- pnorm(abs(ctable4b1_subj[, "t value"]), lower.tail = FALSE) * 2
ctable4b1_subj <- cbind(ctable4b1_subj, "p value" = p4b1_subj)
print(ctable4b1_subj)

cat("\n--- MODEL 4b: TV Channel subset (tv_channel + demographics) ---\n")
model4b2_subj <- polr(subjective_attribution ~ flooded + tv_channel + gender + age + occupation + income, 
                      data = data_tv, 
                      Hess = TRUE)
summary(model4b2_subj)

ctable4b2_subj <- coef(summary(model4b2_subj))
p4b2_subj <- pnorm(abs(ctable4b2_subj[, "t value"]), lower.tail = FALSE) * 2
ctable4b2_subj <- cbind(ctable4b2_subj, "p value" = p4b2_subj)
print(ctable4b2_subj)

# MODEL 5a: Newspaper subset with all demographics
cat("\n--- MODEL 5a: Newspaper subset (flooded + demographics + newspaper) ---\n")
model5a_subj <- polr(subjective_attribution ~ flooded + gender + age + occupation + income + newspaper, 
                     data = data_newspaper, 
                     Hess = TRUE)
summary(model5a_subj)

ctable5a_subj <- coef(summary(model5a_subj))
p5a_subj <- pnorm(abs(ctable5a_subj[, "t value"]), lower.tail = FALSE) * 2
ctable5a_subj <- cbind(ctable5a_subj, "p value" = p5a_subj)
print(ctable5a_subj)

# MODEL 7: Treatment + Preferred Source (newspaper or tv_channel combined)
cat("\n--- MODEL 7: Treatment + Preferred Source (newspaper or tv_channel) ---\n")
model7_subj <- polr(subjective_attribution ~ flooded + preferred_source, 
                    data = data_model7, 
                    Hess = TRUE)
summary(model7_subj)

ctable7_subj <- coef(summary(model7_subj))
p7_subj <- pnorm(abs(ctable7_subj[, "t value"]), lower.tail = FALSE) * 2
ctable7_subj <- cbind(ctable7_subj, "p value" = p7_subj)
print(ctable7_subj)

# MODEL 8b: Treatment + Source of Information

# Report N for Model 8b
cat("N for Model 8b: ", nrow(data), "\n")


# Report N for Model 8
cat("N for Model 8: ", nrow(data), "\n")

cat("\n--- MODEL 8: Treatment + Source of Information ---\n")
model8_subj <- polr(subjective_attribution ~ flooded + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model8_subj)

ctable8_subj <- coef(summary(model8_subj))
p8_subj <- pnorm(abs(ctable8_subj[, "t value"]), lower.tail = FALSE) * 2
ctable8_subj <- cbind(ctable8_subj, "p value" = p8_subj)
print(ctable8_subj)

# MODEL 8c: Treatment + Source of Information + Demographics
cat("\n--- MODEL 8c: Treatment + Source of Information + Demographics ---\n")

# Report N for Model 8c
cat("N for Model 8c: ", nrow(data), "\n")

model8c_subj <- polr(subjective_attribution ~ flooded + source_of_info + gender + age + occupation + income, 
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

# MODEL 1: No controls (just treatment)
cat("\n--- MODEL 1: Treatment only ---\n")
model1_mit <- polr(reported_mitigation_action ~ flooded, 
                   data = data, 
                   Hess = TRUE)
summary(model1_mit)

ctable1_mit <- coef(summary(model1_mit))
p1_mit <- pnorm(abs(ctable1_mit[, "t value"]), lower.tail = FALSE) * 2
ctable1_mit <- cbind(ctable1_mit, "p value" = p1_mit)
print(ctable1_mit)

# MODEL 2: Treatment + Demographics
cat("\n--- MODEL 2: Treatment + Demographics ---\n")
model2_mit <- polr(reported_mitigation_action ~ flooded + gender + age + occupation + income, 
                   data = data, 
                   Hess = TRUE)
summary(model2_mit)

ctable2_mit <- coef(summary(model2_mit))
p2_mit <- pnorm(abs(ctable2_mit[, "t value"]), lower.tail = FALSE) * 2
ctable2_mit <- cbind(ctable2_mit, "p value" = p2_mit)
print(ctable2_mit)

# MODEL 3: Treatment + Demographics + Source of Information
cat("\n--- MODEL 3: Treatment + Demographics + Source of Information ---\n")
model3_mit <- polr(reported_mitigation_action ~ flooded + gender + age + occupation + income + source_of_info, 
                   data = data, 
                   Hess = TRUE)
summary(model3_mit)

ctable3_mit <- coef(summary(model3_mit))
p3_mit <- pnorm(abs(ctable3_mit[, "t value"]), lower.tail = FALSE) * 2
ctable3_mit <- cbind(ctable3_mit, "p value" = p3_mit)
print(ctable3_mit)

# MODEL 4a: Newspaper subset
cat("\n--- MODEL 4a: Newspaper subset (newspaper only) ---\n")
model4a1_mit <- polr(reported_mitigation_action ~ flooded + newspaper, 
                     data = data_newspaper, 
                     Hess = TRUE)
summary(model4a1_mit)

ctable4a1_mit <- coef(summary(model4a1_mit))
p4a1_mit <- pnorm(abs(ctable4a1_mit[, "t value"]), lower.tail = FALSE) * 2
ctable4a1_mit <- cbind(ctable4a1_mit, "p value" = p4a1_mit)
print(ctable4a1_mit)

# MODEL 4b: TV Channel subset
cat("\n--- MODEL 4b: TV Channel subset (tv_channel only) ---\n")
model4b1_mit <- polr(reported_mitigation_action ~ flooded + tv_channel, 
                     data = data_tv, 
                     Hess = TRUE)
summary(model4b1_mit)

ctable4b1_mit <- coef(summary(model4b1_mit))
p4b1_mit <- pnorm(abs(ctable4b1_mit[, "t value"]), lower.tail = FALSE) * 2
ctable4b1_mit <- cbind(ctable4b1_mit, "p value" = p4b1_mit)
print(ctable4b1_mit)

cat("\n--- MODEL 4b: TV Channel subset (tv_channel + demographics) ---\n")
model4b2_mit <- polr(reported_mitigation_action ~ flooded + tv_channel + gender + age + occupation + income, 
                     data = data_tv, 
                     Hess = TRUE)
summary(model4b2_mit)

ctable4b2_mit <- coef(summary(model4b2_mit))
p4b2_mit <- pnorm(abs(ctable4b2_mit[, "t value"]), lower.tail = FALSE) * 2
ctable4b2_mit <- cbind(ctable4b2_mit, "p value" = p4b2_mit)
print(ctable4b2_mit)

# MODEL 5b: Newspaper subset with all demographics
cat("\n--- MODEL 5b: Newspaper subset (flooded + demographics + newspaper) ---\n")
model5b_mit <- polr(reported_mitigation_action ~ flooded + gender + age + occupation + income + newspaper, 
                    data = data_newspaper, 
                    Hess = TRUE)
summary(model5b_mit)

ctable5b_mit <- coef(summary(model5b_mit))
p5b_mit <- pnorm(abs(ctable5b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable5b_mit <- cbind(ctable5b_mit, "p value" = p5b_mit)
print(ctable5b_mit)

# MODEL 7b: Treatment + Preferred Source (newspaper or tv_channel combined)
cat("\n--- MODEL 7b: Treatment + Preferred Source (newspaper or tv_channel) ---\n")
model7b_mit <- polr(reported_mitigation_action ~ flooded + preferred_source, 
                    data = data_model7, 
                    Hess = TRUE)
summary(model7b_mit)

ctable7b_mit <- coef(summary(model7b_mit))
p7b_mit <- pnorm(abs(ctable7b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable7b_mit <- cbind(ctable7b_mit, "p value" = p7b_mit)
print(ctable7b_mit)

# MODEL 8b: Treatment + Source of Information
cat("\n--- MODEL 8b: Treatment + Source of Information ---\n")
model8b_mit <- polr(reported_mitigation_action ~ flooded + source_of_info, 
                    data = data, 
                    Hess = TRUE)
summary(model8b_mit)

ctable8b_mit <- coef(summary(model8b_mit))
p8b_mit <- pnorm(abs(ctable8b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable8b_mit <- cbind(ctable8b_mit, "p value" = p8b_mit)
print(ctable8b_mit)

# MODEL 8-m: Treatment + Source of Information + Demographics
cat("\n--- MODEL 8-m: Treatment + Source of Information + Demographics ---\n")
model8m_mit <- polr(reported_mitigation_action ~ flooded + source_of_info + gender + age + occupation + income, 
                    data = data, 
                    Hess = TRUE)
summary(model8m_mit)

ctable8m_mit <- coef(summary(model8m_mit))
p8m_mit <- pnorm(abs(ctable8m_mit[, "t value"]), lower.tail = FALSE) * 2
ctable8m_mit <- cbind(ctable8m_mit, "p value" = p8m_mit)
print(ctable8m_mit)

# MODEL 9: Treatment effect on mitigation among high climate attribution population
cat("\n--- MODEL 9: Treatment effect on mitigation (high climate attribution subset) ---\n")
cat("Subset: subjective_attribution == 4 or 5\n")
cat("Sample size:", nrow(data_model9), "\n")
model9_mit <- polr(reported_mitigation_action ~ flooded, 
                   data = data_model9, 
                   Hess = TRUE)
summary(model9_mit)

ctable9_mit <- coef(summary(model9_mit))
p9_mit <- pnorm(abs(ctable9_mit[, "t value"]), lower.tail = FALSE) * 2
ctable9_mit <- cbind(ctable9_mit, "p value" = p9_mit)
print(ctable9_mit)

# MODEL 9a: Treatment + Preferred Source (high climate attribution subset)
cat("\n--- MODEL 9a: Treatment + Preferred Source (high climate attribution subset) ---\n")
cat("Subset: subjective_attribution == 4 or 5, with preferred_source\n")
cat("Sample size:", nrow(data_model9a), "\n")
model9a_mit <- polr(reported_mitigation_action ~ flooded + preferred_source, 
                    data = data_model9a, 
                    Hess = TRUE)
summary(model9a_mit)

ctable9a_mit <- coef(summary(model9a_mit))
p9a_mit <- pnorm(abs(ctable9a_mit[, "t value"]), lower.tail = FALSE) * 2
ctable9a_mit <- cbind(ctable9a_mit, "p value" = p9a_mit)
print(ctable9a_mit)

# MODEL 9b: Treatment + Source of Information (high climate attribution subset)
cat("\n--- MODEL 9b: Treatment + Source of Information (high climate attribution subset) ---\n")
cat("Subset: subjective_attribution == 4 or 5\n")
cat("Sample size:", nrow(data_model9), "\n")
model9b_mit <- polr(reported_mitigation_action ~ flooded + source_of_info, 
                    data = data_model9, 
                    Hess = TRUE)
summary(model9b_mit)

ctable9b_mit <- coef(summary(model9b_mit))
p9b_mit <- pnorm(abs(ctable9b_mit[, "t value"]), lower.tail = FALSE) * 2
ctable9b_mit <- cbind(ctable9b_mit, "p value" = p9b_mit)
print(ctable9b_mit)

# MODEL 10: Attribution + Treatment + Source of Information + Demographics
cat("\n--- MODEL 10: Attribution + Treatment + Source of Information + Demographics ---\n")
cat("N for Model 10: ", nrow(data), "\n")
model10_mit <- polr(reported_mitigation_action ~ subjective_attribution + flooded + source_of_info + gender + age + occupation + income, 
                    data = data, 
                    Hess = TRUE)
summary(model10_mit)

ctable10_mit <- coef(summary(model10_mit))
p10_mit <- pnorm(abs(ctable10_mit[, "t value"]), lower.tail = FALSE) * 2
ctable10_mit <- cbind(ctable10_mit, "p value" = p10_mit)
print(ctable10_mit)

# MODEL 11: Attribution * Treatment + Source of Information + Demographics (with interaction)
cat("\n--- MODEL 11: Attribution * Treatment + Source of Information + Demographics (with interaction) ---\n")
cat("N for Model 11: ", nrow(data), "\n")
model11_mit <- polr(reported_mitigation_action ~ subjective_attribution * flooded + source_of_info + gender + age + occupation + income, 
                    data = data, 
                    Hess = TRUE)
summary(model11_mit)

ctable11_mit <- coef(summary(model11_mit))
p11_mit <- pnorm(abs(ctable11_mit[, "t value"]), lower.tail = FALSE) * 2
ctable11_mit <- cbind(ctable11_mit, "p value" = p11_mit)
print(ctable11_mit)

# ============================================================================
# MODEL COMPARISON (AIC/BIC)
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("MODEL COMPARISON (AIC/BIC)\n")
cat(strrep("=", 80), "\n\n")

cat("\nSubjective Attribution Models:\n")
cat("Model 1 AIC:", AIC(model1_subj), "BIC:", BIC(model1_subj), "\n")
cat("Model 2 AIC:", AIC(model2_subj), "BIC:", BIC(model2_subj), "\n")
cat("Model 3 AIC:", AIC(model3_subj), "BIC:", BIC(model3_subj), "\n")
cat("Model 4a (newspaper only) AIC:", AIC(model4a1_subj), "BIC:", BIC(model4a1_subj), "\n")
cat("Model 4b (tv_channel only) AIC:", AIC(model4b1_subj), "BIC:", BIC(model4b1_subj), "\n")
cat("Model 4b (tv_channel + demographics) AIC:", AIC(model4b2_subj), "BIC:", BIC(model4b2_subj), "\n")
cat("Model 5a (flooded + demographics + newspaper) AIC:", AIC(model5a_subj), "BIC:", BIC(model5a_subj), "\n")
cat("Model 7 (flooded + preferred_source) AIC:", AIC(model7_subj), "BIC:", BIC(model7_subj), "\n")
cat("Model 8 (flooded + source_of_info) AIC:", AIC(model8_subj), "BIC:", BIC(model8_subj), "\n")

cat("\nReported Mitigation Action Models:\n")
cat("Model 1 AIC:", AIC(model1_mit), "BIC:", BIC(model1_mit), "\n")
cat("Model 2 AIC:", AIC(model2_mit), "BIC:", BIC(model2_mit), "\n")
cat("Model 3 AIC:", AIC(model3_mit), "BIC:", BIC(model3_mit), "\n")
cat("Model 4a (newspaper only) AIC:", AIC(model4a1_mit), "BIC:", BIC(model4a1_mit), "\n")
cat("Model 4b (tv_channel only) AIC:", AIC(model4b1_mit), "BIC:", BIC(model4b1_mit), "\n")
cat("Model 4b (tv_channel + demographics) AIC:", AIC(model4b2_mit), "BIC:", BIC(model4b2_mit), "\n")
cat("Model 5b (flooded + demographics + newspaper) AIC:", AIC(model5b_mit), "BIC:", BIC(model5b_mit), "\n")
cat("Model 7b (flooded + preferred_source) AIC:", AIC(model7b_mit), "BIC:", BIC(model7b_mit), "\n")
cat("Model 8b (flooded + source_of_info) AIC:", AIC(model8b_mit), "BIC:", BIC(model8b_mit), "\n")
cat("Model 9 (treatment on high attribution subset) AIC:", AIC(model9_mit), "BIC:", BIC(model9_mit), "\n")
cat("Model 9a (high attribution + preferred_source) AIC:", AIC(model9a_mit), "BIC:", BIC(model9a_mit), "\n")
cat("Model 9b (high attribution + source_of_info) AIC:", AIC(model9b_mit), "BIC:", BIC(model9b_mit), "\n")


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
  extract_coef_table(model1_subj, "Model 1", "subjective_attribution"),
  extract_coef_table(model2_subj, "Model 2", "subjective_attribution"),
  extract_coef_table(model3_subj, "Model 3", "subjective_attribution"),
  extract_coef_table(model4a1_subj, "Model 4a (newspaper)", "subjective_attribution"),
  extract_coef_table(model4b1_subj, "Model 4b (tv_channel)", "subjective_attribution"),
  extract_coef_table(model4b2_subj, "Model 4b (tv_channel+demo)", "subjective_attribution"),
  extract_coef_table(model5a_subj, "Model 5a (flooded+demo+newspaper)", "subjective_attribution"),
  extract_coef_table(model7_subj, "Model 7 (flooded+preferred_source)", "subjective_attribution"),
  extract_coef_table(model8_subj, "Model 8 (flooded+source_of_info)", "subjective_attribution")
)

all_results_mit <- rbind(
  extract_coef_table(model1_mit, "Model 1", "reported_mitigation_action"),
  extract_coef_table(model2_mit, "Model 2", "reported_mitigation_action"),
  extract_coef_table(model3_mit, "Model 3", "reported_mitigation_action"),
  extract_coef_table(model4a1_mit, "Model 4a (newspaper)", "reported_mitigation_action"),
  extract_coef_table(model4b1_mit, "Model 4b (tv_channel)", "reported_mitigation_action"),
  extract_coef_table(model4b2_mit, "Model 4b (tv_channel+demo)", "reported_mitigation_action"),
  extract_coef_table(model5b_mit, "Model 5b (flooded+demo+newspaper)", "reported_mitigation_action"),
  extract_coef_table(model7b_mit, "Model 7b (flooded+preferred_source)", "reported_mitigation_action"),
  extract_coef_table(model8b_mit, "Model 8b (flooded+source_of_info)", "reported_mitigation_action"),
  extract_coef_table(model9_mit, "Model 9 (high attribution subset)", "reported_mitigation_action"),
  extract_coef_table(model9a_mit, "Model 9a (high attrib+preferred_source)", "reported_mitigation_action"),
  extract_coef_table(model9b_mit, "Model 9b (high attrib+source_of_info)", "reported_mitigation_action")
)

all_results <- rbind(all_results_subj, all_results_mit)

# Save to CSV
write.csv(all_results, "./ordinal_regression_results.csv", row.names = FALSE)

cat("\n\nResults saved to: ordinal_regression_results.csv\n")

cat("\n")
cat(strrep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 80), "\n")
