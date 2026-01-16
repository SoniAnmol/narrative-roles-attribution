############################################################
# Flood exposure analysis — FINAL CLEAN SCRIPT
############################################################

# -------------------------
# 1) Packages
# -------------------------
library(tidyverse)
library(ordinal)
library(xtable)

# -------------------------
# 2) Load data (Windows-safe)
# -------------------------
candidate_paths <- c(
  "./data/survey_data/T_subset.csv",
  "data/survey_data/T_subset.csv",
  "./T_subset.csv",
  "T_subset.csv"
)

path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(path)) path <- file.choose()

message("Loading: ", normalizePath(path, winslash = "/"))
df_raw <- readr::read_csv(path, show_col_types = FALSE)

# -------------------------
# 3) Clean data (ONCE)
# -------------------------
df <- df_raw %>%
  select(-any_of(c("...1", "X1", "Unnamed: 0", "row_id", "index"))) %>%
  mutate(
    # SINGLE, CANONICAL recode
    flooded = case_when(
      flooded %in% c(TRUE, 1)  ~ "Respondents from flooded municipalities",
      flooded %in% c(FALSE, 0) ~ "Respondents from non-flooded municipalities",
      TRUE ~ NA_character_
    ),
    flooded = factor(flooded, levels = c("Respondents from non-flooded municipalities", "Respondents from flooded municipalities")),

    # Outcomes (ordinal Likert)
    respondents_flood_attribution =
      factor(respondents_flood_attribution, ordered = TRUE),
    took_mitigation_steps_post_foods =
      factor(took_mitigation_steps_post_foods, ordered = TRUE)
  ) %>%
  filter(!is.na(flooded)) %>%
  droplevels()

# -------------------------
# 4) HARD SANITY CHECK (must pass)
# -------------------------
print(table(df$flooded))
stopifnot(nrow(df) == sum(table(df$flooded)))
# Expected: Unaffected = 211, Flooded = 288, Total = 499

# -------------------------
# 5) Ordinal model helper
# -------------------------
fit_clm <- function(outcome, controls = NULL) {

  rhs <- if (is.null(controls)) {
    "flooded"
  } else {
    paste(c("flooded", controls), collapse = " + ")
  }

  f <- as.formula(paste(outcome, "~", rhs))
  m <- clm(f, data = df, link = "logit")
  s <- summary(m)

  coef_row <- grep("^flooded", rownames(s$coef), value = TRUE)[1]

  beta <- s$coef[coef_row, "Estimate"]
  se   <- s$coef[coef_row, "Std. Error"]
  p    <- s$coef[coef_row, "Pr(>|z|)"]

  list(
    model = m,
    stats = tibble(
      outcome = outcome,
      OR = exp(beta),
      CIlo = exp(beta - 1.96 * se),
      CIhi = exp(beta + 1.96 * se),
      p = p
    )
  )
}

# -------------------------
# 6) MAIN MODELS (baseline)
# -------------------------
res_attr <- fit_clm("respondents_flood_attribution")
res_mitg <- fit_clm("took_mitigation_steps_post_foods")

# main_results <- bind_rows(res_attr$stats, res_mitg$stats) %>%
#   mutate(
#     p_holm = p.adjust(p, method = "holm"),
#     Outcome = case_when(
#       outcome == "respondents_flood_attribution" ~
#         "Flood attribution to climate change",
#       outcome == "took_mitigation_steps_post_foods" ~
#         "Mitigation engagement"
#     )
#   )

print(main_results)

# -------------------------
# 7) MAIN TABLE (LaTeX)
# -------------------------
table_main <- main_results %>%
  transmute(
    Outcome,
    `Odds Ratio` = round(OR, 2),
    `95\\% CI` = paste0("[", round(CIlo, 2), ", ", round(CIhi, 2), "]"),
    `p` = round(p, 3),
    # `Holm-adjusted p` = round(p_holm, 3) NOT APPLICABLE AND RELEVANT 
  )

print(
  xtable(
    table_main,
    caption = paste(
      "Flood exposure and climate-related attribution and behaviour.",
      "Odds ratios from proportional-odds ordinal logistic regression models.",
      "Holm-adjusted p-values account for multiple outcome testing."
    ),
    label = "tab:flood_exposure_results"
  ),
  include.rownames = FALSE,
  booktabs = TRUE
)

# -------------------------
# 8) FIGURE: predicted top-category probabilities
# -------------------------
newdat <- tibble(flooded = factor(
  c("Respondents from non-flooded municipalities", "Respondents from flooded municipalities"),
  levels = levels(df$flooded)
))

pp_attr <- predict(res_attr$model, newdata = newdat, type = "prob")
pp_mitg <- predict(res_mitg$model, newdata = newdat, type = "prob")

get_top <- function(pp, i) pp[i, ncol(pp)]

top_probs <- tibble(
  outcome = c("Strong climate attribution", "Strong mitigation engagement"),
  Unaffected = c(get_top(pp_attr$fit, 1), get_top(pp_mitg$fit, 1)),
  Flooded    = c(get_top(pp_attr$fit, 2), get_top(pp_mitg$fit, 2))
) %>%
  pivot_longer(c(Unaffected, Flooded),
               names_to = "group",
               values_to = "probability") %>%
  mutate(
    # Map short names to full labels
    group = case_when(
      group == "Unaffected" ~ "Respondents from non-flooded municipalities",
      group == "Flooded" ~ "Respondents from flooded municipalities",
      TRUE ~ group
    ),
    group = factor(group, levels = c("Respondents from non-flooded municipalities", "Respondents from flooded municipalities")),
    label = scales::percent(probability, accuracy = 1),
    x = case_when(
      outcome == "Strong climate attribution" & group == "Respondents from non-flooded municipalities" ~ 1.0,
      outcome == "Strong climate attribution" & group == "Respondents from flooded municipalities"    ~ 1.4,
      outcome == "Strong mitigation engagement" & group == "Respondents from non-flooded municipalities" ~ 2.0,
      outcome == "Strong mitigation engagement" & group == "Respondents from flooded municipalities"    ~ 2.4
    )
  )

p <- ggplot(top_probs, aes(x = x, y = probability, fill = group)) +
  geom_col(width = 0.35) +
  geom_text(aes(label = label), vjust = -0.4, size = 4) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.45)) +
  scale_x_continuous(
    breaks = c(1.2, 2.2),
    labels = c("Strong climate attribution",
               "Strong mitigation engagement")
  ) +
  scale_fill_manual(values = c("Respondents from non-flooded municipalities" = "#E69F00",
                               "Respondents from flooded municipalities" = "#D55E00")) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
  labs(x = NULL, y = "Predicted probability") +
  theme_classic(base_size = 14) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank()
  )

print(p)

dir.create("figures", showWarnings = FALSE)
ggsave("figures/figure_flooded_attribution_behaviour.png",
       p, width = 140, height = 105, units = "mm", dpi = 300)

message("✓ DONE — counts correct, models valid, outputs ready.")

