#!/usr/bin/env Rscript
# Figure 3 and Table S3 performance from existing participant predictions.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2 || any(args %in% c("--help", "-h"))) {
  cat("Usage: Rscript analysis_code/statistics/subgroup_analysis.R INPUT.csv OUTPUT_DIR\n")
  quit(status = 0)
}
suppressPackageStartupMessages(library(pROC))
data <- read.csv(args[[1]], check.names = FALSE, stringsAsFactors = FALSE)
required <- c("participant_id", "outcome", "fusion", "age_years", "sex_code", "hba1c_percent")
if (!all(required %in% names(data))) stop("Required columns: ", paste(required, collapse = ", "))
if (anyNA(data$participant_id) || anyDuplicated(data$participant_id)) stop("Use one row per participant")
if (!all(data$outcome %in% c(0, 1)) || any(!is.finite(data$fusion)) ||
    any(data$fusion < 0 | data$fusion > 1)) stop("Use binary outcomes and complete probability scores")
mean_hba1c <- mean(data$hba1c_percent, na.rm = TRUE)
if (!is.finite(mean_hba1c)) stop("At least one observed HbA1c measurement is required")
data$hba1c_completed_percent <- data$hba1c_percent
data$hba1c_completed_percent[is.na(data$hba1c_completed_percent)] <- mean_hba1c
assignments <- list(age = ifelse(data$age_years < 45, "<45", "≥45"),
  sex = ifelse(data$sex_code == 1, "Male", ifelse(data$sex_code == 2, "Female", NA)),
  hba1c = ifelse(data$hba1c_completed_percent < 7, "<7%", "≥7%"))
if ("ethnicity_code" %in% names(data)) {
  data$ethnicity_code[is.na(data$ethnicity_code)] <- 1
  assignments$ethnicity <- ifelse(data$ethnicity_code == 1, "Han", "Non-Han")
}
rows <- list()
comparisons <- list()
for (variable in names(assignments)) {
  groups <- assignments[[variable]]
  rocs <- list()
  for (group in unique(groups[!is.na(groups)])) {
    selected <- !is.na(groups) & groups == group
    y <- data$outcome[selected]
    pred <- data$fusion[selected]
    roc <- pROC::roc(y, pred, levels = c(0, 1), direction = "<", quiet = TRUE)
    set.seed(123)
    ci <- as.numeric(pROC::ci.auc(roc, method = "bootstrap", boot.n = 1000, boot.stratified = TRUE))
    cutoff <- pROC::coords(roc, x = "best", best.method = "youden", ret = "threshold", transpose = FALSE)$threshold[1]
    positive <- pred > cutoff
    tp <- sum(positive & y == 1); tn <- sum(!positive & y == 0)
    sensitivity_ci <- binom.test(tp, sum(y == 1))$conf.int
    specificity_ci <- binom.test(tn, sum(y == 0))$conf.int
    rows[[paste(variable, group)]] <- data.frame(subgroup_variable = variable, group = group,
      n = length(y), events = sum(y), auc = as.numeric(pROC::auc(roc)), ci_lower = ci[1], ci_upper = ci[3],
      threshold = cutoff, sensitivity = tp / sum(y == 1), sensitivity_lower = sensitivity_ci[1],
      sensitivity_upper = sensitivity_ci[2], specificity = tn / sum(y == 0),
      specificity_lower = specificity_ci[1], specificity_upper = specificity_ci[2])
    rocs[[group]] <- roc
  }
  if (length(rocs) == 2) {
    comparison <- pROC::roc.test(rocs[[1]], rocs[[2]], method = "delong", paired = FALSE)
    comparisons[[variable]] <- data.frame(subgroup_variable = variable,
      group_a = names(rocs)[1], group_b = names(rocs)[2], p_value = comparison$p.value)
  }
}
dir.create(args[[2]], recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), file.path(args[[2]], "subgroup_performance.csv"), row.names = FALSE)
write.csv(do.call(rbind, comparisons), file.path(args[[2]], "between_group_delong.csv"), row.names = FALSE)
write.csv(data.frame(n = nrow(data), hba1c_mean = mean_hba1c,
                    hba1c_completed = sum(is.na(data$hba1c_percent))),
          file.path(args[[2]], "subgroup_definition.csv"), row.names = FALSE)
cat("Subgroup performance complete.\n")
