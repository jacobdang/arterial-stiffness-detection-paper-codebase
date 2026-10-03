#!/usr/bin/env Rscript
# Retinal-structure score associations and exploratory discrimination.
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) || any(args %in% c("--help", "-h"))) {
  cat("Usage: Rscript analysis_code/retinal_structure/run_analysis.R CONFIG.yaml [--prepare-only] [--discrimination]\n")
  quit(status = 0)
}
suppressPackageStartupMessages(library(yaml))
config <- yaml::read_yaml(args[[1]])
output_dir <- config$output_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
read_table <- function(path) {
  if (grepl("\\.xlsx?$", path, ignore.case = TRUE)) {
    return(as.data.frame(readxl::read_excel(path)))
  }
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}
rename_columns <- function(data, mapping) {
  for (canonical in names(mapping)) {
    source <- mapping[[canonical]]
    if (!source %in% names(data)) stop("Required input column: ", source)
    names(data)[match(source, names(data))] <- canonical
  }
  data
}
check_ids <- function(data) {
  if (!"participant_id" %in% names(data) || anyNA(data$participant_id) ||
      anyDuplicated(data$participant_id)) stop("Use unique participant identifiers")
  data$participant_id <- as.character(data$participant_id)
  data
}
scores <- check_ids(rename_columns(read_table(config$inputs$scores), config$columns$scores))
clinical <- check_ids(rename_columns(read_table(config$inputs$clinical), config$columns$clinical))
if (!setequal(scores$participant_id, clinical$participant_id)) {
  stop("Score and clinical inputs must describe the same analysis cohort")
}
clinical <- clinical[match(scores$participant_id, clinical$participant_id), , drop = FALSE]
score_names <- unlist(config$scores, use.names = FALSE)
if (!all(score_names %in% names(scores))) stop("Required retinal-structure scores are absent")
if (any(!is.finite(as.matrix(scores[score_names])))) stop("Scores must be complete and finite")
if ("as_event" %in% names(scores) && "as_event" %in% names(clinical) &&
    !identical(as.numeric(scores$as_event), as.numeric(clinical$as_event))) {
  stop("Outcomes differ between the aligned inputs")
}
if (!"as_event" %in% names(clinical)) clinical$as_event <- scores$as_event
if (!all(clinical$as_event %in% c(0, 1))) stop("Outcome must be 0 or 1")
if (!is.null(config$inputs$completed_clinical)) {
  completed <- check_ids(rename_columns(read_table(config$inputs$completed_clinical),
                                       config$columns$completed_clinical))
  if (!setequal(clinical$participant_id, completed$participant_id)) {
    stop("Completed clinical input must use the same cohort")
  }
  completed <- completed[match(clinical$participant_id, completed$participant_id), , drop = FALSE]
  for (name in names(config$completion$continuous)) {
    values <- as.numeric(completed[[name]])
    recovery <- config$completion$continuous[[name]]
    restored <- (values + recovery$offset) / recovery$divisor
    missing <- is.na(clinical[[name]])
    clinical[[name]][missing] <- restored[missing]
  }
  for (name in unlist(config$completion$categorical, use.names = FALSE)) {
    clinical[[name]] <- completed[[name]]
  }
}
clinical$coronary_heart_disease[is.na(clinical$coronary_heart_disease)] <- 0
continuous <- unlist(config$continuous_covariates, use.names = FALSE)
for (name in continuous) {
  values <- as.numeric(clinical[[name]])
  observed_mean <- mean(values, na.rm = TRUE)
  if (!is.finite(observed_mean)) stop("No observed values for ", name)
  values[is.na(values)] <- observed_mean
  if (any(!is.finite(values))) stop("Nonfinite clinical values for ", name)
  clinical[[name]] <- values
}
for (name in c("sex", "smoking", "drinking", "coronary_heart_disease")) {
  if (anyNA(clinical[[name]])) stop("Categorical covariate requires an assignment: ", name)
  clinical[[name]] <- factor(clinical[[name]])
}
data <- clinical
for (name in score_names) data[[name]] <- as.numeric(scale(scores[[name]]))
for (name in c("hba1c", "uacr")) data[[name]] <- as.numeric(scale(data[[name]]))
if (any(!is.finite(as.matrix(data[c(score_names, "hba1c", "uacr")])))) {
  stop("Standardized scores and covariates must have positive standard deviations")
}
if (nrow(data) != config$cohort$participants || sum(data$as_event) != config$cohort$events) {
  stop("Input cohort size or event count differs from the configured analysis cohort")
}
standardization <- data.frame(score = score_names,
  mean = vapply(scores[score_names], mean, numeric(1)),
  sample_sd = vapply(scores[score_names], sd, numeric(1)), row.names = NULL)
write.csv(standardization, file.path(output_dir, "score_standardization.csv"), row.names = FALSE)
write.csv(data, file.path(output_dir, "analysis_input.csv"), row.names = FALSE)
if ("--prepare-only" %in% args) quit(status = 0)
suppressPackageStartupMessages(library(broom))
model_covariates <- list(model0 = character(), model1 = c("age_years", "sex"),
  model2 = c("age_years", "sex", "diabetes_duration", "systolic_bp", "bmi",
             "smoking", "drinking", "coronary_heart_disease", "hba1c", "uacr"))
fits <- list()
results <- list()
for (model in names(model_covariates)) {
  for (score in score_names) {
    fit <- glm(reformulate(c(score, model_covariates[[model]]), response = "as_event"),
               data = data, family = binomial(), na.action = na.fail)
    fit_rows <- broom::tidy(fit, conf.int = TRUE, conf.level = 0.95)
    row <- fit_rows[fit_rows$term == score, , drop = FALSE]
    result <- data.frame(score = score, model = model, n = nobs(fit), events = sum(data$as_event),
      estimate = row$estimate, std_error = row$std.error, statistic = row$statistic,
      p_value = row$p.value, conf_low_log = row$conf.low, conf_high_log = row$conf.high,
      odds_ratio = exp(row$estimate), ci_lower = exp(row$conf.low), ci_upper = exp(row$conf.high))
    results[[paste(model, score)]] <- result
    fits[[paste(model, score)]] <- fit
  }
}
results <- do.call(rbind, results)
results$q_value <- NA_real_
for (model in names(model_covariates)) {
  selected <- results$model == model
  results$q_value[selected] <- p.adjust(results$p_value[selected], method = "BH")
}
results <- results[order(match(results$score, score_names), match(results$model, names(model_covariates))), ]
write.csv(results, file.path(output_dir, "figure5_associations.csv"), row.names = FALSE)
if ("--discrimination" %in% args) {
  suppressPackageStartupMessages(library(pROC))
  baseline <- glm(reformulate(model_covariates$model2, response = "as_event"),
                  data = data, family = binomial(), na.action = na.fail)
  base_roc <- pROC::roc(data$as_event, predict(baseline, type = "response"),
                         levels = c(0, 1), direction = "<", quiet = TRUE)
  rows <- lapply(score_names, function(score) {
    fit <- fits[[paste("model2", score)]]
    roc <- pROC::roc(data$as_event, predict(fit, type = "response"),
                      levels = c(0, 1), direction = "<", quiet = TRUE)
    ci <- as.numeric(pROC::ci.auc(roc, method = "delong"))
    base_ci <- as.numeric(pROC::ci.auc(base_roc, method = "delong"))
    data.frame(score = score, n = nrow(data), auc = as.numeric(pROC::auc(roc)),
      ci_lower = ci[1], ci_upper = ci[3], baseline_auc = as.numeric(pROC::auc(base_roc)),
      baseline_ci_lower = base_ci[1], baseline_ci_upper = base_ci[3],
      paired_delong_p = pROC::roc.test(roc, base_roc, method = "delong", paired = TRUE)$p.value,
      likelihood_ratio_p = anova(baseline, fit, test = "LRT")$`Pr(>Chi)`[2])
  })
  write.csv(do.call(rbind, rows), file.path(output_dir, "table_s10_discrimination.csv"), row.names = FALSE)
}
cat("Retinal-structure analysis complete.\n")
