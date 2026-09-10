param([Parameter(Mandatory=$true)][string]$Suite, [Parameter(Mandatory=$true)][string]$Label)
$ErrorActionPreference = 'Stop'
$taskProject = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskSuite = (Resolve-Path -LiteralPath $Suite).Path
if (-not $taskSuite.StartsWith((Join-Path $taskProject 'benchmarks') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Suite must resolve inside project benchmarks.' }
if ($Label -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid batch label.' }
$taskFolder = Join-Path $taskSuite ('batches\' + $Label)
if (-not (Test-Path -LiteralPath $taskFolder)) { throw 'Unknown batch.' }
$taskSnapshot = @(Get-CimInstance Win32_Process)
$taskPattern = '--label\s+' + [regex]::Escape($Label) + '(\s|$)'
$taskRoots = @($taskSnapshot | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m\s+smartcar\.batch\s+run' -and $_.CommandLine -match $taskPattern -and $_.CommandLine.Contains((Split-Path $taskSuite -Leaf)) })
if (-not $taskRoots.Count) { throw 'No matching active batch process.' }
$taskPids = [System.Collections.Generic.HashSet[uint32]]::new()
foreach ($taskRoot in $taskRoots) { [void]$taskPids.Add($taskRoot.ProcessId) }
do {
    $taskChanged = $false
    foreach ($taskProcess in $taskSnapshot) {
        if ($taskPids.Contains($taskProcess.ParentProcessId) -and -not $taskPids.Contains($taskProcess.ProcessId)) {
            if ($taskProcess.Name -ne 'python.exe') { throw 'Unexpected non-Python child; inspect before stopping.' }
            [void]$taskPids.Add($taskProcess.ProcessId); $taskChanged = $true
        }
    }
} while ($taskChanged)
$taskReason = 'Superseded implementation stopped after a diagnosed failure or runtime bottleneck; a separate rerun records the replacement implementation. Existing geometry and validation reports are preserved.'
# Stop queue supervisors first, then their verified descendants.
foreach ($taskRoot in $taskRoots) { Stop-Process -Id $taskRoot.ProcessId -Force -ErrorAction SilentlyContinue }
foreach ($taskProcessId in $taskPids) { Stop-Process -Id $taskProcessId -Force -ErrorAction SilentlyContinue }
$taskRemaining = @(Get-Process -Id @($taskPids) -ErrorAction SilentlyContinue)
if ($taskRemaining.Count) { throw 'Some owned batch processes remain.' }
$taskEvent = @{ label=$Label; timestamp=(Get-Date).ToString('o'); reason=$taskReason; stopped_process_ids=@($taskPids); remaining_owned_processes=0 }
$taskEvent | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskFolder 'cancellation.json') -Encoding utf8
$taskPrefix = (Split-Path $taskSuite -Leaf) + '-' + $Label + '-m'
foreach ($taskRun in (Get-ChildItem -LiteralPath (Join-Path $taskProject 'runs') -Directory | Where-Object { $_.Name.StartsWith($taskPrefix) })) {
    $taskStatusPath = Join-Path $taskRun.FullName 'run_status.json'
    if (-not (Test-Path -LiteralPath $taskStatusPath)) { continue }
    $taskState = Get-Content -LiteralPath $taskStatusPath -Raw | ConvertFrom-Json
    if ($taskState.status -in @('RUNNING','ERROR')) {
        $taskPrevious = $taskState
        @{ status='CANCELLED'; reason=$taskReason; previous_status=$taskPrevious; release_ready=$false; timestamp=$taskEvent.timestamp } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskStatusPath -Encoding utf8
    }
}
$taskEvent | ConvertTo-Json -Depth 5
