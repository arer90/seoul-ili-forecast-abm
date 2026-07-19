#!/usr/bin/env Rscript
# 08_tbats_ets_baseline.R
# forecast::tbats + ets univariate baselines. Reviewer 상식 체크.
# Deps: forecast
# Usage:
#   Rscript 08_tbats_ets_baseline.R [ili.csv] [out.csv]
# Input:  ili_series.csv (week_start, ili_rate)
# Output: results/08_tbats_ets.csv (model, train_n, test_n, R2, RMSE, MAE)

suppressPackageStartupMessages({
  library(forecast)
})

args    <- commandArgs(trailingOnly = TRUE)
in_path <- if (length(args) >= 1) args[1] else "../results/post_E/ili_series.csv"
out_csv <- if (length(args) >= 2) args[2] else "results/08_tbats_ets.csv"

dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) { cat(sprintf("[SKIP] %s\n", in_path)); quit(status = 1) }

df <- read.csv(in_path, stringsAsFactors = FALSE)
df$week_start <- as.Date(df$week_start)
df <- df[order(df$week_start), ]
y  <- as.numeric(df$ili_rate)
n  <- length(y)
cut <- floor(n * 0.85)
y_tr <- y[1:cut]; y_te <- y[(cut + 1):n]; h <- length(y_te)

ts_tr <- ts(y_tr, frequency = 52)

# TBATS
cat("[08] fitting TBATS ...\n")
fit_t <- tbats(ts_tr)
fc_t  <- forecast(fit_t, h = h)
yp_t  <- as.numeric(fc_t$mean)

# ETS
cat("[08] fitting ETS ...\n")
fit_e <- ets(ts_tr)
fc_e  <- forecast(fit_e, h = h)
yp_e  <- as.numeric(fc_e$mean)

met <- function(y, yp) {
  ss_res <- sum((y - yp)^2)
  ss_tot <- sum((y - mean(y))^2)
  c(R2 = 1 - ss_res / ss_tot,
    RMSE = sqrt(mean((y - yp)^2)),
    MAE  = mean(abs(y - yp)))
}
m_t <- met(y_te, yp_t)
m_e <- met(y_te, yp_e)
out <- data.frame(
  model   = c("TBATS", "ETS"),
  train_n = cut, test_n = h,
  R2      = c(m_t["R2"],   m_e["R2"]),
  RMSE    = c(m_t["RMSE"], m_e["RMSE"]),
  MAE     = c(m_t["MAE"],  m_e["MAE"]),
  stringsAsFactors = FALSE
)
print(out)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf("[08] OK -> %s\n", out_csv))
