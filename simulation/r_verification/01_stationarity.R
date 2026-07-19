#!/usr/bin/env Rscript
# 01_stationarity.R
# ADF / KPSS on Seoul weekly ILI series.
# Deps: tseries
# Usage:
#   Rscript 01_stationarity.R [input.csv] [output.csv]
# Input columns:  week_start, ili_rate
# Output columns: test, statistic, p_value, null_hypothesis, conclusion

suppressPackageStartupMessages({
  library(tseries)
})

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else "../results/post_E/ili_series.csv"
out_path <- if (length(args) >= 2) args[2] else "results/01_stationarity.csv"

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) {
  cat(sprintf("[SKIP] input not found: %s\n", in_path))
  cat("  Expected columns: week_start, ili_rate\n")
  cat("  Run post_E_comprehensive_eval.py first to generate.\n")
  quit(status = 1)
}

df <- read.csv(in_path, stringsAsFactors = FALSE)
y  <- as.numeric(df$ili_rate)
y  <- y[is.finite(y)]
cat(sprintf("[01] n = %d weeks\n", length(y)))

adf  <- suppressWarnings(adf.test(y, alternative = "stationary"))
kpss <- suppressWarnings(kpss.test(y, null = "Level"))

out <- data.frame(
  test            = c("ADF", "KPSS-level"),
  statistic       = c(unname(adf$statistic),  unname(kpss$statistic)),
  p_value         = c(adf$p.value,            kpss$p.value),
  null_hypothesis = c("non-stationary",       "stationary"),
  conclusion      = c(
    ifelse(adf$p.value  < 0.05, "reject -> stationary",      "fail-to-reject -> non-stationary"),
    ifelse(kpss$p.value < 0.05, "reject -> non-stationary",  "fail-to-reject -> stationary")
  ),
  stringsAsFactors = FALSE
)
print(out)
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("[01] OK -> %s\n", out_path))
