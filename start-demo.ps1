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

Write-Host "Stopping any existing instances..." -ForegroundColor Yellow
Get-Process python      -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

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
Remove-Item $backendLog, $frontendLog -ErrorAction SilentlyContinue

Start-Process -FilePath $cloudflared -WindowStyle Hidden -ArgumentList `
    "tunnel","--url","http://localhost:8765","--logfile",$backendLog
Start-Process -FilePath $cloudflared -WindowStyle Hidden -ArgumentList `
    "tunnel","--url","http://localhost:5500","--logfile",$frontendLog

Write-Host "Waiting for tunnel URLs..." -ForegroundColor Cyan
$backendUrl = $null
$frontendUrl = $null
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
    if ($backendUrl -and $frontendUrl) { break }
}

if (-not $backendUrl -or -not $frontendUrl) {
    Write-Host "Could not detect tunnel URLs after 30s -- check $backendLog / $frontendLog manually." -ForegroundColor Red
    exit 1
}

Write-Host "Patching frontend/app.js with the new backend host..." -ForegroundColor Cyan
$backendHostOnly = $backendUrl -replace "https://", ""
$appJsPath = "$root\frontend\app.js"
# Read/write via .NET with an explicit no-BOM UTF8 encoding -- PowerShell's
# own Get-Content/Set-Content -Encoding utf8 guesses the source encoding and
# will silently mangle the file's existing em-dashes/emoji otherwise.
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$content = [System.IO.File]::ReadAllText($appJsPath, $utf8NoBom)
$content = $content -replace 'const BACKEND_HOST = "[^"]*";', "const BACKEND_HOST = `"$backendHostOnly`";"
[System.IO.File]::WriteAllText($appJsPath, $content, $utf8NoBom)

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " DEMO READY -- share this link:" -ForegroundColor Green
Write-Host " $frontendUrl" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "(backend: $backendUrl)" -ForegroundColor DarkGray
