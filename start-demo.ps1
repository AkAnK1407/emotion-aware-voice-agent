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

# Voice CX server (Stage 1 + Stage 2, shadow mode) -- EC2 instance running
# voice-cx-server/. NOTE: if this instance is ever stopped and restarted
# (not just rebooted), it gets a new public hostname/IP unless it has an
# Elastic IP attached -- update this if the SSH tunnel below stops working.
$voiceCxHost = "ec2-3-80-80-136.compute-1.amazonaws.com"
$voiceCxKey  = "$root\keypair3.pem"

Write-Host "Stopping any existing instances..." -ForegroundColor Yellow
Get-Process python      -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*8000:localhost:8000*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host "Starting voice-CX SSH tunnel (shadow mode, non-critical)..." -ForegroundColor Cyan
# Fire-and-forget Start-Process doesn't confirm the tunnel actually stayed up
# -- on a resource-tight machine it can spawn and die within a second or two
# (observed under memory pressure while other processes are also starting).
# So verify with a health check and retry a couple of times before giving up.
$voiceCxUp = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        Start-Process -FilePath "ssh" -WindowStyle Hidden -ArgumentList @(
            "-i", $voiceCxKey, "-N", "-L", "8000:localhost:8000",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
            "ubuntu@$voiceCxHost"
        )
    } catch {
        Write-Host "  attempt ${attempt}: Start-Process threw ($_)" -ForegroundColor DarkYellow
    }
    Start-Sleep -Seconds 3
    try {
        $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
        if ($health.status -eq "ok") { $voiceCxUp = $true; break }
    } catch {
        Write-Host "  attempt ${attempt}: tunnel not up yet, retrying..." -ForegroundColor DarkYellow
    }
}
if ($voiceCxUp) {
    Write-Host "  voice-CX tunnel confirmed up." -ForegroundColor Green
} else {
    Write-Host "  voice-CX tunnel failed after 3 attempts -- dashboard panel will just stay empty, rest of the demo is unaffected." -ForegroundColor DarkYellow
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
