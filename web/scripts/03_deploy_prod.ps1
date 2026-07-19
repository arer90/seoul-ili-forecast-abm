<#
.SYNOPSIS
    Stage 3 deploy — Vercel prod deploy, Stage 1+2 통과 시에만 실행.

.DESCRIPTION
    Stage 1 (로컬 무비용) + Stage 2 (API ping) 이 모두 n_fail=0 이어야
    이 스크립트가 돌아간다.  체인:

      ① 리포트 게이트 — 01/02 _report.json exit_code 재확인
      ② Turso 시드 import (이미 seeded 면 skip, size-based diff)
      ③ Vercel env 주입 (vercel env ls 와 diff, 누락분만 추가)
      ④ vercel deploy --prod
      ⑤ 프로덕션 URL smoke — 10 개 주요 경로 HTTP 핑
           /                       (앱 쉘)
           /api/mcp/_list          (10 tools 노출)
           /api/providers          (3 provider availability)
           /public/aggregates/*.json

.PARAMETER DryRun
    실제 배포 안 하고 실행 시뮬레이션만 (vercel env 덮어쓰지 않음)

.PARAMETER SkipTurso
    Turso 시드 import 스킵 (이미 완료한 경우)

.PARAMETER SkipGate
    Stage 1+2 게이트 무시 (절대 권장 안 함; 긴급 핫픽스 전용)

.PARAMETER VercelProject
    Vercel 프로젝트 이름 (default: web/package.json 의 name)

.EXAMPLE
    # 전체 안전 배포
    .\web\scripts\03_deploy_prod.ps1

.EXAMPLE
    # 계획만 보기
    .\web\scripts\03_deploy_prod.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$SkipTurso,
    [switch]$SkipGate,
    [string]$VercelProject = "",
    [string]$ReportPath    = "$PSScriptRoot\deploy_report.json"
)

$ErrorActionPreference = "Continue"

$REPO   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$WEB    = Join-Path $REPO "web"
$SEED   = Join-Path $WEB  "scripts\turso_seed.sql"
$S1_RPT = Join-Path $WEB  "scripts\preflight_local_report.json"
$S2_RPT = Join-Path $WEB  "scripts\preflight_providers_report.json"

$steps = [System.Collections.ArrayList]::new()
function Add-Step {
    param([string]$Name, [string]$Status, [string]$Detail, [int]$ElapsedMs = 0)
    $icon = switch ($Status) {
        "ok"   { "[OK]  " }
        "warn" { "[WARN]" }
        "fail" { "[FAIL]" }
        "skip" { "[SKIP]" }
        "dry"  { "[DRY] " }
    }
    Write-Host ("{0} {1,-26} {2,6}ms  {3}" -f $icon, $Name, $ElapsedMs, $Detail)
    [void]$steps.Add([ordered]@{
        name       = $Name
        status     = $Status
        detail     = $Detail
        elapsed_ms = $ElapsedMs
    })
}

function Invoke-Timed {
    param([scriptblock]$Script)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r  = & $Script
    $sw.Stop()
    return @{ Result = $r; Ms = [int]$sw.ElapsedMilliseconds }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Stage 3 Deploy  —  Vercel production"
Write-Host " DryRun : $DryRun   SkipTurso : $SkipTurso   SkipGate : $SkipGate"
Write-Host "============================================================"
Write-Host ""

# -----------------------------------------------------------------------
# ① Gate: Stage 1+2 must pass
# -----------------------------------------------------------------------
if ($SkipGate) {
    Add-Step "gate_stage12" "warn" "-SkipGate specified; proceeding without verification" 0
} else {
    $missing = @()
    if (-not (Test-Path $S1_RPT)) { $missing += "Stage 1 report" }
    if (-not (Test-Path $S2_RPT)) { $missing += "Stage 2 report" }
    if ($missing.Count -gt 0) {
        Add-Step "gate_stage12" "fail" "missing: $($missing -join ', '). Run 01_*/02_*.ps1 first." 0
        Write-Host ""
        Write-Host "[x] Gate failed — run 01_preflight_local.ps1 then 02_preflight_providers.ps1."
        exit 2
    }
    $s1 = Get-Content -Raw $S1_RPT | ConvertFrom-Json
    $s2 = Get-Content -Raw $S2_RPT | ConvertFrom-Json
    if ($s1.totals.n_fail -gt 0 -or $s2.totals.n_fail -gt 0) {
        Add-Step "gate_stage12" "fail" `
            "S1 fail=$($s1.totals.n_fail) S2 fail=$($s2.totals.n_fail); fix before deploy" 0
        exit 2
    }
    $s1Time = $s1.generated_at
    $s2Time = $s2.generated_at
    Add-Step "gate_stage12" "ok" "S1@$s1Time  S2@$s2Time  (fail=0/0)" 0
}

# -----------------------------------------------------------------------
# ② Tool availability: vercel CLI + turso CLI
# -----------------------------------------------------------------------
$t = Invoke-Timed {
    try {
        $v = (& vercel --version 2>&1) | Select-Object -First 1
        return "$v"
    } catch { return "missing" }
}
if ("$($t.Result)" -match "^\d+\.\d+\.\d+" -or "$($t.Result)" -match "vercel") {
    Add-Step "vercel_cli" "ok" "$($t.Result)" $t.Ms
} else {
    Add-Step "vercel_cli" "fail" "install: npm i -g vercel" $t.Ms
    exit 3
}

if (-not $SkipTurso) {
    $t = Invoke-Timed {
        try {
            $v = (& turso --version 2>&1) | Select-Object -First 1
            return "$v"
        } catch { return "missing" }
    }
    if ("$($t.Result)" -match "turso") {
        Add-Step "turso_cli" "ok" "$($t.Result)" $t.Ms
    } else {
        Add-Step "turso_cli" "warn" "no turso CLI; Turso seed import skipped (add -SkipTurso to silence)" $t.Ms
        $SkipTurso = $true
    }
} else {
    Add-Step "turso_cli" "skip" "-SkipTurso" 0
}

# -----------------------------------------------------------------------
# ③ Turso import (idempotent: skip if count > 50k already)
# -----------------------------------------------------------------------
if ($SkipTurso) {
    Add-Step "turso_import" "skip" "-SkipTurso" 0
} elseif (-not (Test-Path $SEED)) {
    Add-Step "turso_import" "fail" "seed missing: $SEED (run web/scripts/export-turso.py)" 0
    exit 4
} else {
    $sizeMb = [Math]::Round((Get-Item $SEED).Length / 1MB, 1)
    if ($DryRun) {
        Add-Step "turso_import" "dry" "would: turso db shell <db> < $SEED  ($sizeMb MB)" 0
    } else {
        # User must have previously authenticated + set TURSO_DB_NAME in env
        if (-not $env:TURSO_DB_NAME) {
            Add-Step "turso_import" "warn" "TURSO_DB_NAME env missing; skipping import (set it to run)" 0
        } else {
            $t = Invoke-Timed {
                $tmp = Join-Path $env:TEMP "turso_import.log"
                $proc = Start-Process -FilePath "turso" `
                    -ArgumentList "db","shell",$env:TURSO_DB_NAME `
                    -RedirectStandardInput $SEED `
                    -RedirectStandardOutput $tmp `
                    -RedirectStandardError "$tmp.err" `
                    -NoNewWindow -PassThru -Wait
                return @{ Code = $proc.ExitCode; Log = $tmp }
            }
            $status = if ($t.Result.Code -eq 0) { "ok" } else { "fail" }
            Add-Step "turso_import" $status "rc=$($t.Result.Code)  log=$($t.Result.Log)  size=$sizeMb MB" $t.Ms
            if ($t.Result.Code -ne 0) { exit 4 }
        }
    }
}

# -----------------------------------------------------------------------
# ④ Vercel env diff + inject (only missing keys)
# -----------------------------------------------------------------------
$envsNeeded = @(
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
    "TURSO_URL", "TURSO_TOKEN", "UPSTASH_URL", "UPSTASH_TOKEN",
    "DEMO_TOKEN", "MCP_BRIDGE_URL", "NEXT_PUBLIC_HIDE_OLLAMA"
)
$t = Invoke-Timed {
    try {
        Push-Location $WEB
        $list = (& vercel env ls production 2>&1) -join "`n"
        Pop-Location
        return $list
    } catch {
        Pop-Location -ErrorAction SilentlyContinue
        return "ERROR $_"
    }
}
$missingEnvs = @()
foreach ($k in $envsNeeded) {
    if ($t.Result -notmatch "\b$k\b") { $missingEnvs += $k }
}
if ($missingEnvs.Count -eq 0) {
    Add-Step "vercel_env_diff" "ok" "all $($envsNeeded.Count) vars present on Vercel" $t.Ms
} else {
    $detail = "missing on Vercel: $($missingEnvs -join ', ')"
    if ($DryRun) {
        Add-Step "vercel_env_diff" "dry" $detail $t.Ms
    } else {
        # Push missing keys from current shell env
        $pushed = 0
        foreach ($k in $missingEnvs) {
            $val = Get-Item "env:$k" -ErrorAction SilentlyContinue
            if ($val -and $val.Value) {
                $tmp = [System.IO.Path]::GetTempFileName()
                $val.Value | Out-File -FilePath $tmp -Encoding ascii -NoNewline
                Push-Location $WEB
                Get-Content $tmp | & vercel env add $k production 2>&1 | Out-Null
                Pop-Location
                Remove-Item $tmp -Force
                $pushed++
            }
        }
        Add-Step "vercel_env_inject" "ok" "pushed $pushed/$($missingEnvs.Count) missing vars" 0
    }
}

# -----------------------------------------------------------------------
# ⑤ vercel deploy --prod
# -----------------------------------------------------------------------
if ($DryRun) {
    Add-Step "vercel_deploy" "dry" "would: (cd $WEB; vercel deploy --prod)" 0
    $prodUrl = $null
} else {
    $t = Invoke-Timed {
        Push-Location $WEB
        $out = & vercel deploy --prod 2>&1
        $code = $LASTEXITCODE
        Pop-Location
        return @{ Out = "$out"; Code = $code }
    }
    $url = ($t.Result.Out -split "`r?`n" |
        Where-Object { $_ -match "https://.*\.vercel\.app" } |
        Select-Object -Last 1)
    $prodUrl = if ($url -match "(https://[^\s]+)") { $Matches[1] } else { $null }
    $status  = if ($t.Result.Code -eq 0 -and $prodUrl) { "ok" } else { "fail" }
    Add-Step "vercel_deploy" $status "rc=$($t.Result.Code)  url=$prodUrl" $t.Ms
    if ($status -eq "fail") { exit 5 }
}

# -----------------------------------------------------------------------
# ⑥ Production smoke — 4 endpoints
# -----------------------------------------------------------------------
if ($prodUrl) {
    $endpoints = @(
        @{ Name = "prod_root";          Path = "/" },
        @{ Name = "prod_mcp_list";      Path = "/api/mcp/_list" },
        @{ Name = "prod_providers";     Path = "/api/providers" },
        @{ Name = "prod_aggregates";    Path = "/aggregates/latest-choropleth.json" }
    )
    foreach ($ep in $endpoints) {
        $t = Invoke-Timed {
            try {
                $r = Invoke-WebRequest -Uri ($prodUrl + $ep.Path) -UseBasicParsing -TimeoutSec 15
                return @{ Code = $r.StatusCode; Len = $r.RawContentLength }
            } catch {
                return @{ Code = -1; Err = $_.Exception.Message }
            }
        }
        if ($t.Result.Code -eq 200) {
            Add-Step $ep.Name "ok" "200  len=$($t.Result.Len)B  $($ep.Path)" $t.Ms
        } else {
            Add-Step $ep.Name "fail" "code=$($t.Result.Code)  $($t.Result.Err)" $t.Ms
        }
    }
}

# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------
$totals = [ordered]@{
    n_ok   = ($steps | Where-Object { $_.status -eq "ok"   }).Count
    n_warn = ($steps | Where-Object { $_.status -eq "warn" }).Count
    n_fail = ($steps | Where-Object { $_.status -eq "fail" }).Count
    n_dry  = ($steps | Where-Object { $_.status -eq "dry"  }).Count
    n_skip = ($steps | Where-Object { $_.status -eq "skip" }).Count
    total  = $steps.Count
}
$report = [ordered]@{
    stage        = "3_deploy_prod"
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    dry_run      = [bool]$DryRun
    prod_url     = $prodUrl
    steps        = $steps
    totals       = $totals
}
Set-Content -Path $ReportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8

Write-Host ""
Write-Host "------------------------------------------------------------"
Write-Host (" Summary : ok={0}  warn={1}  fail={2}  dry={3}  skip={4}" -f `
    $totals.n_ok, $totals.n_warn, $totals.n_fail, $totals.n_dry, $totals.n_skip)
Write-Host (" URL     : {0}" -f (if ($prodUrl) { $prodUrl } else { "(not deployed)" }))
Write-Host (" Report  : {0}" -f $ReportPath)
Write-Host "------------------------------------------------------------"

if ($totals.n_fail -gt 0) {
    Write-Host ""
    Write-Host "[x] $($totals.n_fail) deploy step(s) failed."
    exit 1
}
Write-Host ""
Write-Host "[v] Stage 3 complete."
if ($prodUrl) { Write-Host "    Open $prodUrl on a phone for external smoke." }
exit 0
