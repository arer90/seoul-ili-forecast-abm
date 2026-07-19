#!/usr/bin/env Rscript
# 06_rt_epiestim.R
# EpiEstim (Cori 2013) Rt estimation + SEIR-V2 rt_effective_trajectory overlay.
# Flu SI: mean 2.6d, sd 1.5d (Lessler 2009; Cowling 2009). Weekly aggregation:
# SI mean ~ 0.4 weeks, sd ~ 0.2 weeks.
# Deps: EpiEstim
# Usage:
#   Rscript 06_rt_epiestim.R [ili.csv] [seir_rt.csv] [out.csv] [out.pdf]
# Inputs:
#   ili_series.csv  (week_start, ili_rate)
#   rt_seir_v2.csv  (week_start, rt_eff)   (optional)
# Outputs:
#   results/06_rt_epiestim.csv (week_start, rt_cori_mean, rt_cori_q025, rt_cori_q975, rt_seir_v2)
#   results/06_rt_overlay.pdf

suppressPackageStartupMessages({
  library(EpiEstim)
})

args    <- commandArgs(trailingOnly = TRUE)
in_ili  <- if (length(args) >= 1) args[1] else "../results/post_E/ili_series.csv"
in_rt   <- if (length(args) >= 2) args[2] else "../results/post_E/rt_seir_v2.csv"
out_csv <- if (length(args) >= 3) args[3] else "results/06_rt_epiestim.csv"
out_pdf <- if (length(args) >= 4) args[4] else "results/06_rt_overlay.pdf"

dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)

if (!file.exists(in_ili)) {
  cat(sprintf("[SKIP] ILI not found: %s\n", in_ili))
  quit(status = 1)
}

df <- read.csv(in_ili, stringsAsFactors = FALSE)
df$week_start <- as.Date(df$week_start)
df <- df[order(df$week_start), ]
y  <- as.numeric(df$ili_rate)

# Upsample weekly -> daily (uniform split over 7 days) so flu SI (2.6d, 1.5d)
# can be used with EpiEstim which requires mean_si >= 1 time unit.
daily_incid <- rep(round(y * 100 / 7), each = 7)
daily_incid[!is.finite(daily_incid) | daily_incid < 0] <- 0
daily_dates <- seq(df$week_start[1], by = "day", length.out = length(daily_incid))

# SI in days — flu mean 2.6d, sd 1.5d (Lessler 2009, Cowling 2009)
cfg <- make_config(list(mean_si = 2.6, std_si = 1.5))
res <- estimate_R(incid = daily_incid, method = "parametric_si", config = cfg)

# Aggregate daily Rt back to weekly by keeping the last day of each week window
rt_daily <- data.frame(
  date         = daily_dates[res$R$t_end],
  rt_cori_mean = res$R$`Mean(R)`,
  rt_cori_q025 = res$R$`Quantile.0.025(R)`,
  rt_cori_q975 = res$R$`Quantile.0.975(R)`,
  stringsAsFactors = FALSE
)
# Align to weekly week_start (Monday of the week)
rt_daily$week_start <- as.Date(cut(rt_daily$date, breaks = "week", start.on.monday = TRUE))
rt <- aggregate(cbind(rt_cori_mean, rt_cori_q025, rt_cori_q975) ~ week_start,
                data = rt_daily, FUN = mean)

if (file.exists(in_rt)) {
  seir <- read.csv(in_rt, stringsAsFactors = FALSE)
  seir$week_start <- as.Date(seir$week_start)
  rt <- merge(rt, seir[, c("week_start", "rt_eff")], by = "week_start", all.x = TRUE)
  names(rt)[names(rt) == "rt_eff"] <- "rt_seir_v2"
} else {
  cat(sprintf("[06] SEIR-V2 Rt not found (%s) -> Cori only\n", in_rt))
  rt$rt_seir_v2 <- NA
}

write.csv(rt, out_csv, row.names = FALSE)
cat(sprintf("[06] CSV -> %s\n", out_csv))

pdf(out_pdf, width = 10, height = 4)
plot(rt$week_start, rt$rt_cori_mean, type = "l", col = "black", lwd = 2,
     ylim = range(c(rt$rt_cori_q025, rt$rt_cori_q975, rt$rt_seir_v2), na.rm = TRUE),
     xlab = "week", ylab = "Rt", main = "Rt: EpiEstim (Cori) vs SEIR-V2 β/γ·S/N")
polygon(c(rt$week_start, rev(rt$week_start)),
        c(rt$rt_cori_q025, rev(rt$rt_cori_q975)),
        col = rgb(0, 0, 0, 0.15), border = NA)
if (any(is.finite(rt$rt_seir_v2))) {
  lines(rt$week_start, rt$rt_seir_v2, col = "red", lwd = 2, lty = 2)
  legend("topright", c("EpiEstim (Cori 2013)", "SEIR-V2-Forced"),
         col = c("black", "red"), lty = c(1, 2), lwd = 2, bty = "n")
}
abline(h = 1.0, col = "gray", lty = 3)
dev.off()
cat(sprintf("[06] PDF -> %s\n", out_pdf))
