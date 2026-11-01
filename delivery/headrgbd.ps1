param(
    [Alias('h')]
    [switch]$Help,
    [switch]$Version,
    [string]$Runtime = '',
    [Parameter(Position = 0)]
    [string]$Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Arguments = @()
)

$ErrorActionPreference = 'Stop'
$Project = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Runtime)) {
    $Runtime = if ($env:WITHB_RUNTIME_ROOT) { $env:WITHB_RUNTIME_ROOT } else { Join-Path $Project 'runtime' }
}
if (-not [IO.Path]::IsPathRooted($Runtime)) { $Runtime = Join-Path $Project $Runtime }
$Python = Join-Path $Runtime 'env\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime Python not found: $Python. See README.md section 'Install'."
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $Project
$env:HEADRGBD_RUNTIME_DIR = (Resolve-Path -LiteralPath $Runtime).Path
$env:WITHB_RUNTIME_ROOT = $env:HEADRGBD_RUNTIME_DIR
if ($Version) { $Command = '--version' }
elseif ($Help -and [string]::IsNullOrWhiteSpace($Command)) { $Command = '--help' }
elseif ([string]::IsNullOrWhiteSpace($Command)) { $Command = '--help' }
& $Python -m head_rgbd $Command @Arguments
exit $LASTEXITCODE
