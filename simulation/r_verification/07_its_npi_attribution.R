#!/usr/bin/env Rscript
# 07_its_npi_attribution.R
# Interrupted Time Series (ITS) via segmented regression at NPI start (2020-03-02).
# Quantifies level shift + slope change attributable to COVID-19 NPI.
# Deps: segmented
# Usage:
#   Rscript 07_its_npi_attribution.R [ili.csv] [npi.csv] [out.csv]
# Inputs:
#   ili_series.csv  (week_start, ili_rate)
#   npi_window.csv  (event, iso_date) — rows event=npi_start, npi_end
# Output:
#   results/07_its_segmented.csv

suppressPackageStartupMessages({
  library(segmented)
})

args    <- commandArgs(trailingOnly = TRUE)
in_ili  <- if (length(args) >= 1) args[1] else "../results/post_E/ili_series.csv"
in_npi  <- if (length(args) >= 2) args[2] else "../results/post_E/npi_window.csv"
out_csv <- if (length(args) >= 3) args[3] else "results/07_its_segmented.csv"

dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_ili)) { cat(sprintf("[SKIP] %s\n", in_ili)); quit(status = 1) }

df <- read.csv(in_ili, stringsAsFactors = FALSE)
df$week_start <- as.Date(df$week_start)
df <- df[order(df$week_start), ]
df$t <- seq_len(nrow(df))
df$sin52 <- sin(2 * pi * df$t / 52)
df$cos52 <- cos(2 * pi * df$t / 52)

# NPI start index
if (file.exists(in_npi)) {
  npi <- read.csv(in_npi, stringsAsFactors = FALSE)
  ns  <- as.Date(npi$iso_date[npi$event == "npi_start"])
  ne  <- as.Date(npi$iso_date[npi$event == "npi_end"])
} else {
  ns <- as.Date("2020-03-02")
  ne <- as.Date("2022-12-26")
}
idx_start <- which.min(abs(df$week_start - ns))
idx_end   <- which.min(abs(df$week_start - ne))
cat(sprintf("[07] NPI window index: [%d, %d] (%s ~ %s)\n",
            idx_start, idx_end, as.character(ns), as.character(ne)))

# Segmented regression: baseline sin/cos + linear t + level shift + slope change at breakpoint
df$post_npi    <- as.integer(df$t >= idx_start)
df$time_post   <- pmax(df$t - idx_start, 0) * df$post_npi
df$post_recov  <- as.integer(df$t > idx_end)
df$time_recov  <- pmax(df$t - idx_end, 0) * df$post_recov

base_lm <- lm(ili_rate ~ t + sin52 + cos52, data = df)
its_lm  <- lm(ili_rate ~ t + sin52 + cos52 + post_npi + time_post + post_recov + time_recov, data = df)

co <- summary(its_lm)$coefficients
get <- function(name) {
  if (name %in% rownames(co)) co[name, ] else c(NA, NA, NA, NA)
}

out <- data.frame(
  term       = c("intercept_shift_NPI_start", "slope_change_during_NPI",
                 "intercept_shift_NPI_end",   "slope_change_post_NPI",
                 "R2_baseline",                "R2_ITS"),
  estimate   = c(get("post_npi")[1],  get("time_post")[1],
                 get("post_recov")[1], get("time_recov")[1],
                 summary(base_lm)$r.squared, summary(its_lm)$r.squared),
  std_error  = c(get("post_npi")[2],  get("time_post")[2],
                 get("post_recov")[2], get("time_recov")[2], NA, NA),
  p_value    = c(get("post_npi")[4],  get("time_post")[4],
                 get("post_recov")[4], get("time_recov")[4], NA, NA),
  stringsAsFactors = FALSE
)
print(out)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("[07] OK -> %s (dR2 = %+.4f)\n", out_csv,
            summary(its_lm)$r.squared - summary(base_lm)$r.squared))
