#!/usr/bin/env Rscript
# 02_residual_diag.R
# Ljung-Box autocorrelation + ARCH heteroskedasticity test on each model's residuals.
# Deps: stats (base), FinTS
# Usage:
#   Rscript 02_residual_diag.R [input.csv] [output.csv]
# Input columns:  model, week_start, residual
# Output columns: model, lb_stat_lag10, lb_p_lag10, lb_stat_lag20, lb_p_lag20, arch_stat, arch_p

suppressPackageStartupMessages({
  library(FinTS)
})

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else "../results/post_E/model_residuals.csv"
out_path <- if (length(args) >= 2) args[2] else "results/02_residual_diag.csv"

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) {
  cat(sprintf("[SKIP] input not found: %s\n", in_path))
  quit(status = 1)
}

df <- read.csv(in_path, stringsAsFactors = FALSE)
models <- unique(df$model)
cat(sprintf("[02] %d models, %d residual rows\n", length(models), nrow(df)))

out_rows <- list()
for (m in models) {
  r <- as.numeric(df$residual[df$model == m])
  r <- r[is.finite(r)]
  if (length(r) < 30) next
  lb10 <- Box.test(r, lag = 10, type = "Ljung-Box")
  lb20 <- Box.test(r, lag = 20, type = "Ljung-Box")
  arch <- suppressWarnings(ArchTest(r, lags = 12))
  out_rows[[m]] <- data.frame(
    model         = m,
    n             = length(r),
    lb_stat_lag10 = unname(lb10$statistic),
    lb_p_lag10    = lb10$p.value,
    lb_stat_lag20 = unname(lb20$statistic),
    lb_p_lag20    = lb20$p.value,
    arch_stat     = unname(arch$statistic),
    arch_p        = arch$p.value,
    stringsAsFactors = FALSE
  )
}
out <- do.call(rbind, out_rows)
# p > 0.05 = 자기상관/이분산 없음 (good); p < 0.05 = 남아있음 (model misspecified)
out$autocorr_ok <- out$lb_p_lag10 > 0.05 & out$lb_p_lag20 > 0.05
out$homosk_ok   <- out$arch_p     > 0.05

print(out)
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("[02] OK -> %s\n", out_path))
