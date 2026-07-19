#!/usr/bin/env Rscript
# run_all.R — orchestrate all 8 verification scripts.
# Usage:
#   cd simulation/r_verification && Rscript run_all.R
#   Background (bash):  nohup Rscript run_all.R > r_verification.log 2>&1 &
#   Background (PS):    Start-Process Rscript run_all.R -NoNewWindow -RedirectStandardOutput r_verification.log

scripts <- c(
  "01_stationarity.R",
  "02_residual_diag.R",
  "03_dm_test_canonical.R",
  "04_wis_crps_pit.R",
  "05_nb_dispersion.R",
  "06_rt_epiestim.R",
  "07_its_npi_attribution.R",
  "08_tbats_ets_baseline.R"
)

status <- integer(length(scripts))
names(status) <- scripts
t0 <- Sys.time()

for (i in seq_along(scripts)) {
  s <- scripts[i]
  cat(sprintf("\n============================================================\n"))
  cat(sprintf("[run_all] (%d/%d) %s\n", i, length(scripts), s))
  cat(sprintf("============================================================\n"))
  rc <- tryCatch(
    {
      # Run in child R process so failure doesn't abort the whole sequence
      system2("Rscript", s)
    },
    error = function(e) { cat(sprintf("[ERR] %s\n", conditionMessage(e))); 99 }
  )
  status[i] <- rc
}

cat(sprintf("\n============================================================\n"))
cat(sprintf("[run_all] done in %.1f sec\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
cat(sprintf("============================================================\n"))
for (s in names(status)) {
  tag <- ifelse(status[[s]] == 0, "OK",
         ifelse(status[[s]] == 1, "SKIP (missing input)", sprintf("FAIL (rc=%d)", status[[s]])))
  cat(sprintf("  %-35s %s\n", s, tag))
}
n_ok   <- sum(status == 0)
n_skip <- sum(status == 1)
n_fail <- sum(status != 0 & status != 1)
cat(sprintf("\n  %d OK / %d SKIP / %d FAIL\n", n_ok, n_skip, n_fail))
quit(status = ifelse(n_fail > 0, 1, 0))
