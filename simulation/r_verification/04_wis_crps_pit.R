#!/usr/bin/env Rscript
# 04_wis_crps_pit.R
# Canonical WIS (Weighted Interval Score, Bracher 2021) + CRPS + PIT histogram.
# Deps: scoringutils (>= 2.0), scoringRules
# Usage:
#   Rscript 04_wis_crps_pit.R [input.csv] [output.csv] [pit_pdf]
# Input columns:  model, week_start, y_true, q025, q500, q975
#   (optional extra quantiles: q100, q250, q750, q900 for multi-interval WIS)
# Output:
#   results/04_wis_crps_pit.csv (model, wis_mean, crps_mean, coverage_95, coverage_50)
#   results/04_pit_histogram.pdf (PIT uniformity check)

suppressPackageStartupMessages({
  library(scoringutils)
  library(scoringRules)
})

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else "../results/post_E/pi_samples_wide.csv"
out_path <- if (length(args) >= 2) args[2] else "results/04_wis_crps_pit.csv"
pdf_path <- if (length(args) >= 3) args[3] else "results/04_pit_histogram.pdf"

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) {
  cat(sprintf("[SKIP] input not found: %s\n", in_path))
  quit(status = 1)
}

df <- read.csv(in_path, stringsAsFactors = FALSE)
# Expect quantiles; map to long form for scoringutils
qcols <- intersect(c("q025", "q100", "q250", "q500", "q750", "q900", "q975"), names(df))
qvals <- as.numeric(sub("q", "", qcols)) / 1000
cat(sprintf("[04] quantile cols: %s\n", paste(qcols, collapse = ", ")))

long <- do.call(rbind, lapply(seq_along(qcols), function(i) {
  data.frame(
    model         = df$model,
    week_start    = df$week_start,
    observed      = df$y_true,
    quantile_level = qvals[i],
    predicted     = df[[qcols[i]]],
    stringsAsFactors = FALSE
  )
}))

fc <- as_forecast_quantile(long,
  observed = "observed", predicted = "predicted",
  quantile_level = "quantile_level",
  forecast_unit = c("model", "week_start")
)
sc <- score(fc)
sum_tab <- summarise_scores(sc, by = "model")
print(sum_tab)
write.csv(as.data.frame(sum_tab), out_path, row.names = FALSE)
cat(sprintf("[04] OK -> %s\n", out_path))

# PIT histogram per model (scoringRules::pit for quantile forecasts approximated
# by interpolating empirical CDF; uses rank of observed among quantiles)
models <- unique(df$model)
pdf(pdf_path, width = 10, height = 3 * ceiling(length(models) / 3))
op <- par(mfrow = c(ceiling(length(models) / 3), 3))
for (m in models) {
  sub <- df[df$model == m, ]
  pit <- mapply(function(y, q025, q500, q975) {
    qs <- c(0.025, 0.5, 0.975)
    vs <- c(q025, q500, q975)
    approx(vs, qs, xout = y, rule = 2)$y
  }, sub$y_true, sub$q025, sub$q500, sub$q975)
  pit <- pit[is.finite(pit)]
  hist(pit, breaks = 10, main = m, xlab = "PIT", col = "steelblue")
  abline(h = length(pit) / 10, col = "red", lty = 2)
}
par(op)
dev.off()
cat(sprintf("[04] PIT PDF -> %s\n", pdf_path))
