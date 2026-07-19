<#
.SYNOPSIS
    Stage 2 preflight — 실제 API 키로 최소 비용(< $0.001) 핑 테스트.

.DESCRIPTION
    Stage 1 통과 후 돌리는 두 번째 방어선.  API 키가 실제로 유효한지,
    Turso / Upstash 가 접속되는지를 "가장 싼 모델 × max_tokens=8" 로만
    확인한다.  예상 비용:

        Claude Haiku 3.5    ~$0.00001  (입력 20 tok × $0.80/MTok)
        GPT-4o-mini         ~$0.00001
        Gemini 2.0 Flash    $0         (free tier)
        Turso SELECT 1      $0
        Upstash PING        $0         (free tier)
        ───────────────────────────────
        합계                < $0.00005

    각 provider 당 1 call 만.  재시도 없음.

.PARAMETER EnvFile
    .env 파일 경로 (default: repo root .env).  여기에 API_KEY 들이 있어야
    한다.  $env:ANTHROPIC_API_KEY 등이 이미 세팅돼 있으면 -NoEnvFile.

.PARAMETER NoEnvFile
    .env 를 읽지 않고 현재 shell env 만 사용

.PARAMETER ReportPath
    JSON 리포트 출력 경로

.EXAMPLE
    # repo/.env 로드 후 핑
    .\web\scripts\02_preflight_providers.ps1

.EXAMPLE
    # 이미 $env:ANTHROPIC_API_KEY 등 세팅돼 있는 경우
    .\web\scripts\02_preflight_providers.ps1 -NoEnvFile
#>
[CmdletBinding()]
param(
    [string]$EnvFile    = "",
    [switch]$NoEnvFile,
    [string]$ReportPath = "$PSScriptRoot\preflight_providers_report.json"
)

$ErrorActionPreference = "Continue"

$REPO = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $EnvFile) { $EnvFile = Join-Path $REPO ".env" }

$checks  = [System.Collections.ArrayList]::new()
$costEst = [System.Collections.ArrayList]::new()

function Add-Check {
    param([string]$Name, [string]$Status, [string]$Detail, [int]$ElapsedMs = 0,
           [double]$CostUsd = 0.0)
    $icon = switch ($Status) {
        "ok"   { "[OK]  " }
        "warn" { "[WARN]" }
        "fail" { "[FAIL]" }
    }
    $costStr = if ($CostUsd -gt 0) { ("  ~`$" + ("{0:F6}" -f $CostUsd)) } else { "" }
    Write-Host ("{0} {1,-28} {2,6}ms  {3}{4}" -f $icon, $Name, $ElapsedMs, $Detail, $costStr)
    [void]$checks.Add([ordered]@{
        name       = $Name
        status     = $Status
        detail     = $Detail
        elapsed_ms = $ElapsedMs
        cost_usd   = $CostUsd
    })
    if ($CostUsd -gt 0) { [void]$costEst.Add($CostUsd) }
}

function Invoke-Timed {
    param([scriptblock]$Script)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r  = & $Script
    $sw.Stop()
    return @{ Result = $r; Ms = [int]$sw.ElapsedMilliseconds }
}

# -----------------------------------------------------------------------
# Load .env
# -----------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================"
Write-Host " Stage 2 Preflight  —  Minimum-cost API probe"
Write-Host " Env file : $(if ($NoEnvFile) { '(shell only)' } else { $EnvFile })"
Write-Host "============================================================"
Write-Host ""

if (-not $NoEnvFile) {
    if (Test-Path $EnvFile) {
        Get-Content $EnvFile | ForEach-Object {
            if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*"?([^"#]*)"?\s*(#.*)?$') {
                $n = $Matches[1]; $v = $Matches[2].Trim()
                if ($v) { Set-Item -Path "env:$n" -Value $v }
            }
        }
        Write-Host "[i] loaded $EnvFile"
    } else {
        Write-Host "[!] $EnvFile not found — using shell env only"
    }
}

# -----------------------------------------------------------------------
# ① Key presence check (no network)
# -----------------------------------------------------------------------
$keys = @{
    ANTHROPIC_API_KEY = $env:ANTHROPIC_API_KEY
    OPENAI_API_KEY    = $env:OPENAI_API_KEY
    GOOGLE_API_KEY    = $env:GOOGLE_API_KEY
    TURSO_URL         = $env:TURSO_URL
    TURSO_TOKEN       = $env:TURSO_TOKEN
    UPSTASH_URL       = $env:UPSTASH_URL
    UPSTASH_TOKEN     = $env:UPSTASH_TOKEN
    DEMO_TOKEN        = $env:DEMO_TOKEN
}
$missing = @()
foreach ($k in $keys.Keys) {
    if ([string]::IsNullOrWhiteSpace($keys[$k])) {
        $missing += $k
    }
}
if ($missing.Count -eq 0) {
    Add-Check "env_keys_present" "ok" "8/8 env vars loaded" 0
} elseif ($missing.Count -le 3 -and ($missing | Where-Object { $_ -eq "DEMO_TOKEN" -or $_ -eq "UPSTASH_TOKEN" -or $_ -eq "TURSO_TOKEN" }).Count -eq $missing.Count) {
    Add-Check "env_keys_present" "warn" "missing: $($missing -join ', ')" 0
} else {
    Add-Check "env_keys_present" "fail" "missing: $($missing -join ', ')" 0
}

# -----------------------------------------------------------------------
# ② Anthropic Claude Haiku 3.5 ping
# -----------------------------------------------------------------------
if ($env:ANTHROPIC_API_KEY) {
    $t = Invoke-Timed {
        try {
            $body = @{
                model      = "claude-3-5-haiku-20241022"
                max_tokens = 8
                messages   = @(@{ role = "user"; content = "say hi" })
            } | ConvertTo-Json -Depth 5
            $resp = Invoke-WebRequest -Uri "https://api.anthropic.com/v1/messages" `
                -Method POST `
                -Headers @{
                    "x-api-key"         = $env:ANTHROPIC_API_KEY
                    "anthropic-version" = "2023-06-01"
                    "content-type"      = "application/json"
                } `
                -Body $body -UseBasicParsing -TimeoutSec 15
            $d = $resp.Content | ConvertFrom-Json
            $inTok  = $d.usage.input_tokens
            $outTok = $d.usage.output_tokens
            $cost   = ($inTok * 0.80 + $outTok * 4.00) / 1e6
            return @{ Ok = $true; In = $inTok; Out = $outTok; Cost = $cost; Text = $d.content[0].text }
        } catch {
            return @{ Ok = $false; Err = $_.Exception.Message }
        }
    }
    if ($t.Result.Ok) {
        Add-Check "anthropic_haiku_ping" "ok" `
            "in=$($t.Result.In)t out=$($t.Result.Out)t  reply='$($t.Result.Text.Trim())'" `
            $t.Ms $t.Result.Cost
    } else {
        Add-Check "anthropic_haiku_ping" "fail" "$($t.Result.Err)" $t.Ms
    }
} else {
    Add-Check "anthropic_haiku_ping" "warn" "ANTHROPIC_API_KEY missing; skipped" 0
}

# -----------------------------------------------------------------------
# ③ OpenAI GPT-4o-mini ping
# -----------------------------------------------------------------------
if ($env:OPENAI_API_KEY) {
    $t = Invoke-Timed {
        try {
            $body = @{
                model       = "gpt-4o-mini"
                max_tokens  = 8
                temperature = 0
                messages    = @(@{ role = "user"; content = "say hi" })
            } | ConvertTo-Json -Depth 5
            $resp = Invoke-WebRequest -Uri "https://api.openai.com/v1/chat/completions" `
                -Method POST `
                -Headers @{
                    "Authorization" = "Bearer $env:OPENAI_API_KEY"
                    "content-type"  = "application/json"
                } `
                -Body $body -UseBasicParsing -TimeoutSec 15
            $d = $resp.Content | ConvertFrom-Json
            $inTok  = $d.usage.prompt_tokens
            $outTok = $d.usage.completion_tokens
            # 4o-mini: $0.15/MTok in, $0.60/MTok out
            $cost   = ($inTok * 0.15 + $outTok * 0.60) / 1e6
            return @{ Ok = $true; In = $inTok; Out = $outTok; Cost = $cost; Text = $d.choices[0].message.content }
        } catch {
            return @{ Ok = $false; Err = $_.Exception.Message }
        }
    }
    if ($t.Result.Ok) {
        Add-Check "openai_4omini_ping" "ok" `
            "in=$($t.Result.In)t out=$($t.Result.Out)t  reply='$($t.Result.Text.Trim())'" `
            $t.Ms $t.Result.Cost
    } else {
        Add-Check "openai_4omini_ping" "fail" "$($t.Result.Err)" $t.Ms
    }
} else {
    Add-Check "openai_4omini_ping" "warn" "OPENAI_API_KEY missing; skipped" 0
}

# -----------------------------------------------------------------------
# ④ Google Gemini 2.0 Flash ping (free tier)
# -----------------------------------------------------------------------
if ($env:GOOGLE_API_KEY) {
    $t = Invoke-Timed {
        try {
            $url  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=$env:GOOGLE_API_KEY"
            $body = @{
                contents         = @(@{ parts = @(@{ text = "say hi" }) })
                generationConfig = @{ maxOutputTokens = 8; temperature = 0 }
            } | ConvertTo-Json -Depth 6
            $resp = Invoke-WebRequest -Uri $url -Method POST `
                -Headers @{ "content-type" = "application/json" } `
                -Body $body -UseBasicParsing -TimeoutSec 15
            $d = $resp.Content | ConvertFrom-Json
            $text = $d.candidates[0].content.parts[0].text
            $inTok  = $d.usageMetadata.promptTokenCount
            $outTok = $d.usageMetadata.candidatesTokenCount
            return @{ Ok = $true; In = $inTok; Out = $outTok; Cost = 0.0; Text = $text }
        } catch {
            return @{ Ok = $false; Err = $_.Exception.Message }
        }
    }
    if ($t.Result.Ok) {
        Add-Check "google_flash_ping" "ok" `
            "in=$($t.Result.In)t out=$($t.Result.Out)t  reply='$($t.Result.Text.Trim())'  (free tier)" `
            $t.Ms 0.0
    } else {
        Add-Check "google_flash_ping" "fail" "$($t.Result.Err)" $t.Ms
    }
} else {
    Add-Check "google_flash_ping" "warn" "GOOGLE_API_KEY missing; skipped" 0
}

# -----------------------------------------------------------------------
# ⑤ Turso SELECT 1
# -----------------------------------------------------------------------
if ($env:TURSO_URL -and $env:TURSO_TOKEN) {
    $t = Invoke-Timed {
        try {
            # Turso REST v1 (Hrana over HTTP)
            $url = $env:TURSO_URL.TrimEnd("/") + "/v2/pipeline"
            $body = @{
                requests = @(
                    @{ type = "execute"; stmt = @{ sql = "SELECT 1 AS ok" } }
                    @{ type = "close" }
                )
            } | ConvertTo-Json -Depth 6
            $resp = Invoke-WebRequest -Uri $url -Method POST `
                -Headers @{
                    "Authorization" = "Bearer $env:TURSO_TOKEN"
                    "content-type"  = "application/json"
                } `
                -Body $body -UseBasicParsing -TimeoutSec 15
            $d = $resp.Content | ConvertFrom-Json
            $val = $d.results[0].response.result.rows[0][0].value
            return @{ Ok = $true; Val = $val }
        } catch {
            return @{ Ok = $false; Err = $_.Exception.Message }
        }
    }
    if ($t.Result.Ok) {
        Add-Check "turso_select1" "ok" "SELECT 1 -> $($t.Result.Val)" $t.Ms 0.0
    } else {
        Add-Check "turso_select1" "fail" "$($t.Result.Err)" $t.Ms
    }
} else {
    Add-Check "turso_select1" "warn" "TURSO_URL/TURSO_TOKEN missing; skipped" 0
}

# -----------------------------------------------------------------------
# ⑥ Upstash Redis PING
# -----------------------------------------------------------------------
if ($env:UPSTASH_URL -and $env:UPSTASH_TOKEN) {
    $t = Invoke-Timed {
        try {
            $url = $env:UPSTASH_URL.TrimEnd("/") + "/ping"
            $resp = Invoke-WebRequest -Uri $url -Method GET `
                -Headers @{ "Authorization" = "Bearer $env:UPSTASH_TOKEN" } `
                -UseBasicParsing -TimeoutSec 10
            $d = $resp.Content | ConvertFrom-Json
            return @{ Ok = $true; Val = "$($d.result)" }
        } catch {
            return @{ Ok = $false; Err = $_.Exception.Message }
        }
    }
    if ($t.Result.Ok -and $t.Result.Val -eq "PONG") {
        Add-Check "upstash_ping" "ok" "-> PONG" $t.Ms 0.0
    } elseif ($t.Result.Ok) {
        Add-Check "upstash_ping" "warn" "unexpected reply: $($t.Result.Val)" $t.Ms 0.0
    } else {
        Add-Check "upstash_ping" "fail" "$($t.Result.Err)" $t.Ms
    }
} else {
    Add-Check "upstash_ping" "warn" "UPSTASH_URL/UPSTASH_TOKEN missing; skipped" 0
}

# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------
$totalCost = 0.0
foreach ($c in $costEst) { $totalCost += $c }
$totals = [ordered]@{
    n_ok        = ($checks | Where-Object { $_.status -eq "ok"   }).Count
    n_warn      = ($checks | Where-Object { $_.status -eq "warn" }).Count
    n_fail      = ($checks | Where-Object { $_.status -eq "fail" }).Count
    total       = $checks.Count
    cost_usd    = [Math]::Round($totalCost, 6)
}

$report = [ordered]@{
    stage        = "2_preflight_providers"
    generated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    repo         = $REPO
    checks       = $checks
    totals       = $totals
}
Set-Content -Path $ReportPath -Value ($report | ConvertTo-Json -Depth 8) -Encoding UTF8

Write-Host ""
Write-Host "------------------------------------------------------------"
Write-Host (" Summary : ok={0}  warn={1}  fail={2}  cost=`${3:F6}" -f `
    $totals.n_ok, $totals.n_warn, $totals.n_fail, $totals.cost_usd)
Write-Host (" Report  : {0}" -f $ReportPath)
Write-Host "------------------------------------------------------------"

if ($totals.n_fail -gt 0) {
    Write-Host ""
    Write-Host "[x] $($totals.n_fail) provider check(s) FAILED — fix keys/endpoints before Stage 3."
    exit 1
}
Write-Host ""
Write-Host "[v] Stage 2 passed.  Total probe cost = `$$($totals.cost_usd)"
Write-Host "    Proceed to Stage 3 (03_deploy_prod.ps1) when ready."
exit 0
