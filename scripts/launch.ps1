param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('setup', 'run', 'test', 'batch')]
    [string] $Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs = @()
)

$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
$uvVersion = '0.11.14'
$uvArchiveHash = '52ba5d19409aaa688a8a1a6ec8dfb6a4817230d20186e75f4006105c3e39a846'

function Find-ProjectUv {
    # Use only the tested uv version. A different installed version is left alone.
    $installed = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($installed) {
        $description = & $installed.Source --version
        if ($LASTEXITCODE -eq 0 -and $description -match "^uv $([regex]::Escape($uvVersion))( |$)") {
            return $installed.Source
        }
    }
    $toolDirectory = Join-Path $projectDirectory ".tools/uv-$uvVersion"
    $localUv = Join-Path $toolDirectory 'uv.exe'
    if (Test-Path -LiteralPath $localUv) { return $localUv }
    New-Item -ItemType Directory -Path $toolDirectory -Force | Out-Null
    $archive = Join-Path $toolDirectory 'uv.zip'
    $url = "https://github.com/astral-sh/uv/releases/download/$uvVersion/uv-x86_64-pc-windows-msvc.zip"
    Write-Host "Downloading verified uv $uvVersion into .tools ..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $url -OutFile $archive -UseBasicParsing
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $uvArchiveHash) {
        throw 'uv download checksum mismatch; the archive has not been executed.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $toolDirectory -Force
    if (!(Test-Path -LiteralPath $localUv)) { throw 'Verified uv archive did not contain uv.exe.' }
    return $localUv
}

try {
    if (![Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
        throw 'The supplied dependency lock supports Windows x64. Other platforms are not verified.'
    }
    $uvExecutable = Find-ProjectUv
    # A caller's active environment or another checkout must not receive packages.
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectDirectory '.venv'
    $env:SMARTCAR_PROJECT_ROOT = $projectDirectory
    $env:PYTHONUTF8 = '1'
    $env:MPLBACKEND = 'Agg'
    Push-Location -LiteralPath $projectDirectory
    try {
        & $uvExecutable sync --locked --python 3.12
        if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed. Check network access to GitHub and PyPI, then retry.' }
    } finally { Pop-Location }
    $pythonExecutable = Join-Path $projectDirectory '.venv/Scripts/python.exe'
    switch ($Command) {
        'setup' { & $pythonExecutable (Join-Path $projectDirectory 'tools/check_install.py') @CommandArgs }
        'run' { & $pythonExecutable -m smartcar.pipeline @CommandArgs }
        'batch' { & $pythonExecutable -m smartcar.batch @CommandArgs }
        'test' {
            Push-Location -LiteralPath $projectDirectory
            try { & $pythonExecutable -m pytest @CommandArgs } finally { Pop-Location }
        }
    }
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
