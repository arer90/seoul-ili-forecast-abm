#!/usr/bin/env Rscript
# 03_dm_test_canonical.R
# Canonical Diebold-Mariano test via forecast::dm.test.
# Cross-validates the Python DM implementation (phase5_dm_test.py).
# Deps: forecast
# Usage:
#   Rscript 03_dm_test_canonical.R [input.csv] [output.csv]
# Input columns:  model, week_start, y_true, y_pred, regime
# Output columns: model_a, model_b, regime, n, dm_stat, dm_p, better_model

suppressPackageStartupMessages({
  library(forecast)
})

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else "../results/post_E/model_predictions_long.csv"
out_path <- if (length(args) >= 2) args[2] else "results/03_dm_canonical.csv"

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) {
  cat(sprintf("[SKIP] input not found: %s\n", in_path))
  quit(status = 1)
}

df <- read.csv(in_path, stringsAsFactors = FALSE)
df$residual <- df$y_true - df$y_pred
models  <- sort(unique(df$model))
regimes <- unique(df$regime)
if (length(regimes) == 0) regimes <- "global"
cat(sprintf("[03] %d models, %d regimes\n", length(models), length(regimes)))

out_rows <- list()
for (rg in regimes) {
  sub <- if (rg == "global") df else df[df$regime == rg, ]
  # align by week_start: inner join residuals per model
  wide <- reshape(
    sub[, c("model", "week_start", "residual")],
    idvar = "week_start", timevar = "model", direction = "wide"
  )
  for (i in seq_along(models)) {
    for (j in seq_along(models)) {
      if (i >= j) next
      a <- models[i]; b <- models[j]
      ca <- paste0("residual.", a); cb <- paste0("residual.", b)
      if (!all(c(ca, cb) %in% names(wide))) next
      ea <- wide[[ca]]; eb <- wide[[cb]]
      keep <- is.finite(ea) & is.finite(eb)
      if (sum(keep) < 20) next  # relaxed from 30 to allow seasonal sub-regimes
      ea <- ea[keep]; eb <- eb[keep]
      res <- tryCatch(
        dm.test(ea, eb, alternative = "two.sided", h = 1, power = 2),
        error = function(e) NULL
      )
      if (is.null(res)) next
      out_rows[[length(out_rows) + 1]] <- data.frame(
        model_a      = a, model_b = b, regime = rg, n = sum(keep),
        dm_stat      = unname(res$statistic),
        dm_p         = res$p.value,
        better_model = ifelse(res$p.value < 0.05,
                              ifelse(mean(ea^2) < mean(eb^2), a, b),
                              "tie"),
        stringsAsFactors = FALSE
      )
    }
  }
}
out <- if (length(out_rows)) do.call(rbind, out_rows) else data.frame()
print(head(out, 20))
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("[03] OK %d pairs -> %s\n", nrow(out), out_path))
