#!/usr/bin/env Rscript
# 05_nb_dispersion.R
# Negative Binomial vs Poisson — dispersion test (Cameron & Trivedi 1990, AER::dispersiontest).
# Justifies NegBinGLM choice over Poisson.
# Deps: MASS, AER
# Usage:
#   Rscript 05_nb_dispersion.R [input.csv] [output.csv]
# Input columns:  week_start, ili_rate  (optional: ili_count, patient_total)
# Output columns: test, statistic, p_value, alpha_hat, conclusion

suppressPackageStartupMessages({
  library(MASS)
  library(AER)
})

args     <- commandArgs(trailingOnly = TRUE)
in_path  <- if (length(args) >= 1) args[1] else "../results/post_E/ili_series.csv"
out_path <- if (length(args) >= 2) args[2] else "results/05_nb_dispersion.csv"

dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_path)) {
  cat(sprintf("[SKIP] input not found: %s\n", in_path))
  quit(status = 1)
}

df <- read.csv(in_path, stringsAsFactors = FALSE)

# ILI rate is per 1000 — reconstruct count if patient_total present,
# else treat rate * 100 as rounded pseudo-count (qualitatively equivalent).
if (all(c("ili_count", "patient_total") %in% names(df))) {
  df$y <- round(df$ili_count)
  df$offset_log <- log(pmax(df$patient_total, 1))
} else {
  df$y <- round(as.numeric(df$ili_rate) * 10)
  df$offset_log <- 0
}
df <- df[is.finite(df$y) & df$y >= 0, ]
df$t <- seq_len(nrow(df))
df$sin52 <- sin(2 * pi * df$t / 52)
df$cos52 <- cos(2 * pi * df$t / 52)

# Poisson fit
p.fit <- glm(y ~ sin52 + cos52 + t, data = df, family = poisson(), offset = df$offset_log)
# Cameron-Trivedi dispersion test: var(y) = mu + alpha * mu
ct <- dispersiontest(p.fit, trafo = 1)  # trafo=1 => alpha*mu (NB2)

# NB fit for alpha estimate
nb.fit <- tryCatch(
  glm.nb(y ~ sin52 + cos52 + t + offset(offset_log), data = df),
  error = function(e) NULL
)
alpha_hat <- if (!is.null(nb.fit)) 1 / nb.fit$theta else NA

out <- data.frame(
  test       = c("Cameron-Trivedi (NB2)", "NB glm.nb theta -> alpha"),
  statistic  = c(unname(ct$statistic), NA),
  p_value    = c(ct$p.value, NA),
  alpha_hat  = c(unname(ct$estimate), alpha_hat),
  conclusion = c(
    ifelse(ct$p.value < 0.05, "reject Poisson -> overdispersed (use NB)", "Poisson OK"),
    ifelse(!is.na(alpha_hat) && alpha_hat > 0.1, "NB alpha >> 0 confirms overdispersion", "alpha small")
  ),
  stringsAsFactors = FALSE
)
print(out)
write.csv(out, out_path, row.names = FALSE)
cat(sprintf("[05] OK -> %s\n", out_path))
