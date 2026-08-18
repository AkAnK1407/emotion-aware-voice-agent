# start-demo.ps1
# Starts (or restarts) the whole AdaptiveCX demo stack in one shot:
#   - the LiveKit agent worker (agent/agent.py dev)
#   - the frontend static file server (port 5500)
#   - two Cloudflare quick tunnels (one for each of the above)
# Safe to re-run any time something goes down -- it kills any previous
# instances of python.exe / cloudflared.exe first, then starts fresh ones,
# waits for the new tunnel URLs, and automatically patches frontend/app.js
# so the page always points at the current backend tunnel.
#
# NOTE: this stops ALL running python.exe processes on this machine, not
# just this project's. Fine on a dedicated demo laptop; don't run this if
# you have unrelated Python work open.

$ErrorActionPreference = "Stop"
$root        = "C:\Users\hp\Desktop\agent v1\adaptivecx-demo"
$cloudflared = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$venvPython  = "$root\.venv312\Scripts\python.exe"

# Voice CX server (Stage 1 + Stage 2, shadow mode) -- runs remotely on a
# Kaggle GPU notebook (voice-cx-server/kaggle_notebook.ipynb), tunneled via
# ngrok. Its URL is read from VOICE_CX_SERVER_URL in .env -- update .env
# (not this script) whenever the Kaggle notebook is restarted and issues a
# new ngrok URL.
$envPath = "$root\.env"
$voiceCxUrl = $null
if (Test-Path $envPath) {
    $m = Select-String -Path $envPath -Pattern '^VOICE_CX_SERVER_URL=(.+)$' | Select-Object -First 1
    if ($m) { $voiceCxUrl = $m.Matches[0].Groups[1].Value.Trim() }
}

Write-Host "Stopping any existing instances..." -ForegroundColor Yellow
Get-Process python      -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

if (-not $voiceCxUrl) {
    Write-Host "  VOICE_CX_SERVER_URL not set in .env -- voice-CX shadow panel will stay disabled." -ForegroundColor DarkYellow
} else {
    Write-Host "Checking remote voice-CX server ($voiceCxUrl)..." -ForegroundColor Cyan
    try {
        $health = Invoke-RestMethod -Uri "$voiceCxUrl/health" -TimeoutSec 10 -ErrorAction Stop
        if ($health.status -eq "ok" -and $health.model_loaded -eq $true) {
            Write-Host "  voice-CX server ready at $voiceCxUrl" -ForegroundColor Green
        } else {
            Write-Host "  voice-CX server reachable but still loading the model -- dashboard will work once it finishes." -ForegroundColor DarkYellow
        }
    } catch {
        Write-Host "  Could not reach $voiceCxUrl -- check the Kaggle notebook is running and .env has the current ngrok URL." -ForegroundColor Red
    }
}

$agentLog    = "$env:TEMP\agent_worker.log"
$frontendSrvLog = "$env:TEMP\frontend_server.log"

Write-Host "Starting agent worker..." -ForegroundColor Cyan
Start-Process -FilePath $venvPython -ArgumentList "agent.py","dev" `
    -WorkingDirectory "$root\agent" -WindowStyle Minimized `
    -RedirectStandardOutput $agentLog -RedirectStandardError "$agentLog.err"

Write-Host "Starting frontend server..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "-m","http.server","5500" `
    -WorkingDirectory "$root\frontend" -WindowStyle Minimized `
    -RedirectStandardOutput $frontendSrvLog -RedirectStandardError "$frontendSrvLog.err"

Start-Sleep -Seconds 3

Write-Host "Clearing any stale demo room state..." -ForegroundColor Cyan
& $venvPython "$root\clear_room.py" "adaptivecx-live-room"

Write-Host "Starting Cloudflare tunnels..." -ForegroundColor Cyan
$backendLog  = "$env:TEMP\cf_backend.log"
$frontendLog = "$env:TEMP\cf_frontend.log"
$authLog     = "$env:TEMP\cf_auth.log"
Remove-Item $backendLog, $frontendLog, $authLog -ErrorAction SilentlyContinue

Start-Process -FilePath $cloudflared -WindowStyle Hidden -ArgumentList `
    "tunnel","--url","http://localhost:8765","--logfile",$backendLog
Start-Process -FilePath $cloudflared -WindowStyle Hidden -ArgumentList `
    "tunnel","--url","http://localhost:5500","--logfile",$frontendLog
# Login/signup/history API (auth_server.py) -- separate service, separate
# tunnel, since dashboard_bridge's server can't accept anything but a
# bodyless GET (see auth_server.py's module docstring for why).
Start-Process -FilePath $cloudflared -WindowStyle Hidden -ArgumentList `
    "tunnel","--url","http://localhost:8766","--logfile",$authLog

Write-Host "Waiting for tunnel URLs..." -ForegroundColor Cyan
$backendUrl = $null
$frontendUrl = $null
$authUrl = $null
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (-not $backendUrl -and (Test-Path $backendLog)) {
        $m = Select-String -Path $backendLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $backendUrl = $m.Matches[0].Value }
    }
    if (-not $frontendUrl -and (Test-Path $frontendLog)) {
        $m = Select-String -Path $frontendLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $frontendUrl = $m.Matches[0].Value }
    }
    if (-not $authUrl -and (Test-Path $authLog)) {
        $m = Select-String -Path $authLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $authUrl = $m.Matches[0].Value }
    }
    if ($backendUrl -and $frontendUrl -and $authUrl) { break }
}

if (-not $backendUrl -or -not $frontendUrl -or -not $authUrl) {
    Write-Host "Could not detect tunnel URLs after 30s -- check $backendLog / $frontendLog / $authLog manually." -ForegroundColor Red
    exit 1
}

Write-Host "Patching frontend/app.js with the new backend + auth hosts..." -ForegroundColor Cyan
$backendHostOnly = $backendUrl -replace "https://", ""
$authHostOnly = $authUrl -replace "https://", ""
$appJsPath = "$root\frontend\app.js"
# Read/write via .NET with an explicit no-BOM UTF8 encoding -- PowerShell's
# own Get-Content/Set-Content -Encoding utf8 guesses the source encoding and
# will silently mangle the file's existing em-dashes/emoji otherwise.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$content = [System.IO.File]::ReadAllText($appJsPath, $utf8NoBom)
$content = $content -replace 'const BACKEND_HOST = "[^"]*";', "const BACKEND_HOST = `"$backendHostOnly`";"
$content = $content -replace 'const AUTH_HOST = "[^"]*";', "const AUTH_HOST = `"$authHostOnly`";"
[System.IO.File]::WriteAllText($appJsPath, $content, $utf8NoBom)

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " DEMO READY -- share this link:" -ForegroundColor Green
Write-Host " $frontendUrl" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "(backend: $backendUrl)" -ForegroundColor DarkGray
Write-Host "(auth:    $authUrl)" -ForegroundColor DarkGray
