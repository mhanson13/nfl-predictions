# Registers the "NFL Pipeline Dispatcher" scheduled task (runs every 15 minutes).
#
#   pwsh -File tools\install_dispatcher_task.ps1            # install / update
#   pwsh -File tools\install_dispatcher_task.ps1 -DispatcherArgs "--no-site-deploy"  # trial: no site deploy
#   pwsh -File tools\install_dispatcher_task.ps1 -Uninstall # remove
#
# The task runs as the current user while logged on (so git/gh/npm credentials work),
# hidden (conhost --headless), wakes the PC for each trigger, never overlaps itself,
# and appends output to logs\dispatcher\dispatcher.log. Most triggers exit within
# seconds; only a due game-day slot starts the pipeline (see tools\dispatcher.py).
param(
    [string]$Python = "$env:USERPROFILE\anaconda3\envs\nflgpu\python.exe",
    [string]$Repo = (Resolve-Path "$PSScriptRoot\..").Path,
    [int]$EveryMinutes = 15,
    [string]$DispatcherArgs = "",   # e.g. "--no-site-deploy" for a trial weekend
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"
$TaskName = "NFL Pipeline Dispatcher"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed '$TaskName'."
    return
}
if (-not (Test-Path $Python)) { throw "Python not found: $Python" }

New-Item -ItemType Directory -Force -Path "$Repo\logs\dispatcher" | Out-Null
$cmd = "cd /d `"$Repo`" && `"$Python`" -u -m tools.dispatcher $DispatcherArgs >> logs\dispatcher\dispatcher.log 2>&1"
$action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless cmd.exe /c $cmd" -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Runs tools.dispatcher every $EveryMinutes min (nfl-predictions)." -Force | Out-Null
Write-Host "Installed '$TaskName': every $EveryMinutes min, repo $Repo, python $Python, args '$DispatcherArgs'"
Write-Host "Check: Get-ScheduledTaskInfo -TaskName '$TaskName'   Log: $Repo\logs\dispatcher\dispatcher.log"
