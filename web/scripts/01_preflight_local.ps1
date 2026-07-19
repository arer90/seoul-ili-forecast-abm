<#
.SYNOPSIS
    Stage 1 preflight — API 키 없이 로컬에서 돈 0원으로 모든 것 검증.

.DESCRIPTION
    Frame D 배포 전 마지막 방어선.  API 호출 전에 "뭐가 깨졌는지" 전부
    알아내야 API 비용을 안 태운다.  다음을 모두 확인:

      ① Python venv + 필수 패키지
      ② epi_sim.db 71 tables / ~4.49M rows quick_check
      ③ MCP 10/10 tools smoke (simulation/scripts/smoke_stage6_mcp)
      ④ 정적 aggregates JSON (25 gus + 50 edges)
      ⑤ Turso seed SQL 크기 (>= 18 MB)
      ⑥ Node/npm + web/node_modules 존재
      ⑦ npm run build 통과 (선택 — 시간 오래 걸리면 -SkipBuild)
      ⑧ Ollama + qwen2.5 설치 여부 (없으면 WARN, FAIL 아님)
      ⑨ Ollama + MCP 10-tool E2E smoke (선택 — -SkipOllama 로 스킵)

    각 체크는 ok / warn / fail 중 하나.  fail 이 하나라도 있으면 exit 1.

.PARAMETER SkipBuild
    npm run build 건너뛰기 (이미 최근에 빌드한 경우)

.PARAMETER SkipOllama
    Ollama 관련 체크 모두 건너뛰기

.PARAMETER OllamaModel
    E2E 스모크에 쓸 모델 (default: qwen2.5:14b-instruct-q5_K_M)

.PARAMETER ReportPath
    JSON 리포트 출력 경로

.EXAMPLE
    # 풀 검증
    .\web\scripts\01_preflight_local.ps1

.EXAMPLE
    # 빠른 루프 (빌드 · Ollama 스킵)
    .\web\scripts\01_preflight_local.ps1 -SkipBuild -SkipOllama

.EXAMPLE
    # 다른 로컬 모델로 E2E
    .\web\scripts\01_preflight_local.ps1 -OllamaModel "exaone3.5:7.8b"
#>
[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipOllama,
    [string]$OllamaModel = "qwen2.5:14b-instruct-q5_K_M",
    [string]$ReportPath  = "$PSScriptRoot\preflight_local_report.json"
)

$ErrorActionPreference = "Continue"  # 한 체크 실패해도 다음 체크 진행

$REPO = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PY   = Join-Path $REPO ".venv\Scripts\python.exe"

$checks = [System.Collections.ArrayList]::new()

function Add-Check {
    param(
        [string]$Name,
        [string]$Status,   # ok | warn | fail
        [string]$Detail,
        [int]$ElapsedMs = 0
    )
    $icon = switch ($Status) {
        "ok"   { "[OK]  " }
        "warn" { "[WARN]" }
        "fail" { "[FAIL]" }
    }
    Write-Host ("{0} {1,-32} {2,6}ms  {3}" -f $icon, $Name, $ElapsedMs, $Detail)
    [void]$checks.Add([ordered]@{
        name       = $Name
        status     = $Status
        detail     = $Detail
        elapsed_ms = $ElapsedMs
    })
}

function Invoke-Timed {
    param([scriptblock]$Script)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $result = & $Script
    $sw.Stop()
    return @{ Result = $result; Ms = [int]$sw.ElapsedMilliseconds }
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Stage 1 Preflight  —  API-free local verification"
Write-Host " REPO  : $REPO"
Write-Host " Time  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"
Write-Host ""

# -----------------------------------------------------------------------
# ① Python venv
# -----------------------------------------------------------------------
$t = Invoke-Timed { Test-Path $PY }
if ($t.Result) {
    $ver = & $PY --version 2>&1
    Add-Check "python_venv" "ok" "$ver  @ .venv\Scripts\python.exe" $t.Ms
} else {
    Add-Check "python_venv" "fail" "missing: $PY" $t.Ms
}

# -----------------------------------------------------------------------
# ② DB quick_check — use a temp Python script so we don't fight
#    PowerShell here-string escape rules for quotes inside quotes.
# -----------------------------------------------------------------------
$t = Invoke-Timed {
    $pyScript = @'
from simulation.database import safe_connect, quick_check
from simulation.database.config import DB_PATH
# quick_check takes an optional path (default DB_PATH), not a Connection.
ok = quick_check()
con = safe_connect()
n_tables = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
n_rows = con.execute(
    "SELECT SUM(cnt) FROM ("
    " SELECT COUNT(*) AS cnt FROM weekly_disease"
    " UNION ALL SELECT COUNT(*) FROM who_flunet"
    " UNION ALL SELECT COUNT(*) FROM weather_historical)"
).fetchone()[0]
con.close()
print(f"{DB_PATH}|{ok}|{n_tables}|{n_rows}")
'@
    $tmpPy = [System.IO.Path]::GetTempFileName() + ".py"
    # Windows PowerShell 5.1's  Set-Content -Encoding UTF8  writes a BOM,
    # which Python parses as a literal `\ufeff` at the start of line 1 and
    # rejects with SyntaxError.  Force BOM-less UTF-8.
    [System.IO.File]::WriteAllText($tmpPy, $pyScript, (New-Object System.Text.UTF8Encoding $false))
    # PYTHONPATH so `from simulation.database import ...` resolves.
    # NB: Push-Location alone does NOT fix this — Windows native subprocesses
    # inherit [System.Environment]::CurrentDirectory, not PowerShell's
    # location provider.
    $prevPP = $env:PYTHONPATH
    $env:PYTHONPATH = $REPO
    try {
        $out = & $PY $tmpPy 2>&1
        return "$out"
    } finally {
        if ($null -eq $prevPP) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        else                   { $env:PYTHONPATH = $prevPP }
        Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
    }
}
if ($t.Result -match "\|ok\|(\d+)\|(\d+)") {
    Add-Check "db_quick_check" "ok" "tables=$($Matches[1])  key_rows=$($Matches[2])" $t.Ms
} else {
    $detail = ("$($t.Result)" -replace "`r?`n"," ")
    if ($detail.Length -gt 160) { $detail = $detail.Substring(0, 160) }
    Add-Check "db_quick_check" "fail" $detail $t.Ms
}

# -----------------------------------------------------------------------
# ③ MCP 10-tool smoke — parse the JSON, not stdout.
#    Success criterion: n_tools == 10 AND n_error == 0.  "ok" status
#    labels alone undercount because lead_time_analysis returns
#    "csv_proxy" and literature_rag returns "static_fallback" —
#    both are valid wired responses.
# -----------------------------------------------------------------------
$smokeJson = Join-Path $REPO "simulation\results\smoke_stage6_mcp.json"
$t = Invoke-Timed {
    # PYTHONPATH so `-m simulation...` finds the package — Push-Location
    # alone doesn't fix native-subprocess cwd on Windows.
    $prevPP = $env:PYTHONPATH
    $env:PYTHONPATH = $REPO
    try {
        $out  = & $PY -m simulation.scripts.smoke_stage6_mcp 2>&1
        $code = $LASTEXITCODE
        return @{ Out = "$out"; Code = $code }
    } finally {
        if ($null -eq $prevPP) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
        else                   { $env:PYTHONPATH = $prevPP }
    }
}
if (Test-Path $smokeJson) {
    $sj   = Get-Content -Raw -Encoding UTF8 $smokeJson | ConvertFrom-Json
    $nAll = $sj.totals.n_tools
    $nOk  = $sj.totals.n_ok
    $nErr = $sj.totals.n_error
    $nIns = $sj.totals.n_insufficient_data
    $nNa  = $sj.totals.n_not_available
    $nOther = $nAll - $nOk - $nErr - $nIns - $nNa        # e.g. csv_proxy, static_fallback
    $status = if ($nAll -ge 10 -and $nErr -eq 0) { "ok" }
              elseif ($nErr -eq 0) { "warn" } else { "fail" }
    Add-Check "mcp_10tool_smoke" $status `
        "$nAll tools  ok=$nOk  other=$nOther  insuf=$nIns  stub=$nNa  err=$nErr" $t.Ms
} else {
    Add-Check "mcp_10tool_smoke" "fail" "smoke json missing: $smokeJson" $t.Ms
}

# -----------------------------------------------------------------------
# ④ Static aggregates
# -----------------------------------------------------------------------
$aggDir = Join-Path $REPO "web\public\aggregates"
$t = Invoke-Timed {
    $chop = Get-ChildItem -Path $aggDir -Filter "*choropleth*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    $edges = Join-Path $aggDir "commuter-edges.json"
    if (-not $chop)                 { return "missing choropleth" }
    if (-not (Test-Path $edges))    { return "missing commuter-edges" }
    # -Encoding UTF8 is critical — Seoul gu names are Hangul; default cp949
    # read mangles them and ConvertFrom-Json chokes on the bad bytes.
    $cd = Get-Content -Raw -Encoding UTF8 -Path $chop.FullName | ConvertFrom-Json
    $ed = Get-Content -Raw -Encoding UTF8 -Path $edges         | ConvertFrom-Json
    $nGu = if ($cd.rows) { $cd.rows.Count } elseif ($cd.records) { $cd.records.Count } else { 0 }
    $nEdges = if ($ed -is [array]) { $ed.Count } elseif ($ed.edges) { $ed.edges.Count } else { 0 }
    return "gu=$nGu edges=$nEdges  file=$($chop.Name)"
}
if ($t.Result -match "gu=(\d+)\s+edges=(\d+)") {
    $nGu = [int]$Matches[1]; $nEdges = [int]$Matches[2]
    $status = if ($nGu -ge 25 -and $nEdges -ge 25) { "ok" }
              elseif ($nGu -ge 10) { "warn" } else { "fail" }
    Add-Check "static_aggregates" $status "$($t.Result)" $t.Ms
} else {
    Add-Check "static_aggregates" "fail" "$($t.Result)" $t.Ms
}

# -----------------------------------------------------------------------
# ⑤ Turso seed size
# -----------------------------------------------------------------------
$seed = Join-Path $REPO "web\scripts\turso_seed.sql"
$t = Invoke-Timed {
    if (Test-Path $seed) {
        $sizeMb = [Math]::Round((Get-Item $seed).Length / 1MB, 1)
        return "$sizeMb MB"
    } else { return "missing" }
}
if ("$($t.Result)" -match "([\d.]+)\s*MB") {
    $mb = [double]$Matches[1]
    $status = if ($mb -ge 18) { "ok" } elseif ($mb -ge 5) { "warn" } else { "fail" }
    Add-Check "turso_seed" $status "$($t.Result)  (expect ~19 MB)" $t.Ms
} else {
    Add-Check "turso_seed" "fail" "$($t.Result)" $t.Ms
}

# -----------------------------------------------------------------------
# ⑥ Node/npm
# -----------------------------------------------------------------------
$t = Invoke-Timed {
    try {
        $nodeV = (& node --version) 2>&1
        $npmV  = (& npm --version) 2>&1
        return "node $nodeV  npm $npmV"
    } catch { return "node/npm not on PATH" }
}
if ("$($t.Result)" -match "node\s+v(\d+)\.") {
    $major = [int]$Matches[1]
    $status = if ($major -ge 18) { "ok" } else { "warn" }
    Add-Check "node_npm" $status "$($t.Result)" $t.Ms
} else {
    Add-Check "node_npm" "fail" "$($t.Result)" $t.Ms
}

# -----------------------------------------------------------------------
# ⑦ web/node_modules + (optional) build
# -----------------------------------------------------------------------
$webDir = Join-Path $REPO "web"
$nm     = Join-Path $webDir "node_modules"
$t = Invoke-Timed { Test-Path $nm }
if ($t.Result) {
    Add-Check "web_node_modules" "ok" "$nm exists" $t.Ms
} else {
    Add-Check "web_node_modules" "warn" "run: npm --prefix $webDir ci --legacy-peer-deps" $t.Ms
}

if ($SkipBuild) {
    Add-Check "web_build" "warn" "-SkipBuild specified; skipped" 0
} else {
    $t = Invoke-Timed {
        $out = & npm --prefix $webDir run build 2>&1
        return @{ Out = "$out"; Code = $LASTEXITCODE }
    }
    $status = if ($t.Result.Code -eq 0) { "ok" } else { "fail" }
    $tail = ($t.Result.Out -split "`r?`n" | Select-Object -Last 2) -join " | "
    Add-Check "web_build" $status "rc=$($t.Result.Code)  $tail" $t.Ms
}

# -----------------------------------------------------------------------
# ⑧ Ollama availability
# -----------------------------------------------------------------------
if ($SkipOllama) {
    Add-Check "ollama_installed" "warn" "-SkipOllama specified; skipped" 0
    Add-Check "ollama_e2e_smoke" "warn" "-SkipOllama specified; skipped" 0
} else {
    $t = Invoke-Timed {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5
            $data = $resp.Content | ConvertFrom-Json
            $names = $data.models | ForEach-Object { $_.name }
            return @{ Up = $true; Names = $names }
        } catch { return @{ Up = $false; Error = $_.Exception.Message } }
    }
    if ($t.Result.Up) {
        $hasModel = $t.Result.Names | Where-Object { $_ -like ($OllamaModel.Split(':')[0] + ":*") }
        $status = if ($hasModel) { "ok" } else { "warn" }
        $detail = if ($hasModel) { "ollama up; $($t.Result.Names.Count) models; '$OllamaModel' available" }
                  else { "ollama up but '$OllamaModel' missing; run: ollama pull $OllamaModel" }
        Add-Check "ollama_installed" $status $detail $t.Ms
    } else {
        Add-Check "ollama_installed" "warn" "ollama not reachable at :11434 ($($t.Result.Error))" $t.Ms
    }

    # ⑨ Ollama + MCP E2E smoke — only if ollama up and model present
    if ($t.Result.Up -and $hasModel) {
        $t2 = Invoke-Timed {
            $prevPP = $env:PYTHONPATH
            $env:PYTHONPATH = $REPO
            try {
                $out = & $PY -m simulation.scripts.smoke_ollama_e2e `
                    --model $OllamaModel --max-hops 4 --timeout 240 2>&1
                return @{ Out = "$out"; Code = $LASTEXITCODE }
            } finally {
                if ($null -eq $prevPP) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
                else                   { $env:PYTHONPATH = $prevPP }
            }
        }
        $status = if ($t2.Result.Code -eq 0) { "ok" } else { "fail" }
        $tail = ($t2.Result.Out -split "`r?`n" | Select-Object -Last 8) -join " | "
        Add-Check "ollama_e2e_smoke" $status "rc=$($t2.Result.Code)  $($tail.Substring(0, [Math]::Min(160, $tail.Length)))" $t2.Ms
    } else {
        Add-Check "ollama_e2e_smoke" "warn" "skipped (ollama / model unavailable)" 0
    }
}

# -----------------------------------------------------------------------
# Write report
# -----------------------------------------------------------------------
# PowerShell 5.1 gotcha: when Where-Object returns 0 or 1 items, the result
# is $null or a scalar — .Count then returns $null or missing.  Wrapping
# with @(...) forces an array so .Count is always an int.
$totals = [ordered]@{
    n_ok   = @($checks | Where-Object { $_.status -eq "ok"   }).Count
    n_warn = @($checks | Where-Object { $_.status -eq "warn" }).Count
    n_fail = @($checks | Where-Object { $_.status -eq "fail" }).Count
    total  = $checks.Count
}

$report = [ordered]@{
    stage        = "1_preflight_local"
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    repo         = $REPO
    ollama_model = $OllamaModel
    checks       = $checks
    totals       = $totals
}
$reportJson = $report | ConvertTo-Json -Depth 8
# BOM-less UTF-8 for same reason as db_quick_check above — Stage 3 and any
# downstream JSON readers choke on the Windows-PS default BOM.
[System.IO.File]::WriteAllText($ReportPath, $reportJson, (New-Object System.Text.UTF8Encoding $false))

Write-Host ""
Write-Host "------------------------------------------------------------"
Write-Host (" Summary  : ok={0}  warn={1}  fail={2}  total={3}" -f `
    $totals.n_ok, $totals.n_warn, $totals.n_fail, $totals.total)
Write-Host (" Report   : {0}" -f $ReportPath)
Write-Host "------------------------------------------------------------"

if ($totals.n_fail -gt 0) {
    Write-Host ""
    Write-Host "[x] $($totals.n_fail) check(s) FAILED — fix before Stage 2 / Stage 3."
    exit 1
} else {
    Write-Host ""
    Write-Host "[v] Stage 1 passed.  Proceed to Stage 2 when you have API keys."
    exit 0
}
