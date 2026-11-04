param(
    [Alias('h')]
    [switch]$Help,
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
    throw "Runtime Python not found: $Python. See docs/TURN_GUI.md."
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $Project
$env:HEADRGBD_RUNTIME_DIR = (Resolve-Path -LiteralPath $Runtime).Path
$env:WITHB_RUNTIME_ROOT = $env:HEADRGBD_RUNTIME_DIR
if ($Help) {
    if ([string]::IsNullOrWhiteSpace($Command)) { $Command = '--help' }
    else { $Arguments = @('--help') + $Arguments }
}
if ([string]::IsNullOrWhiteSpace($Command)) {
    $GuiPython = Join-Path $Runtime 'env\pythonw.exe'
    if (Test-Path -LiteralPath $GuiPython -PathType Leaf) {
        Start-Process -FilePath $GuiPython -ArgumentList @('-m', 'head_rgbd.experiment.turn_app', 'gui') -WorkingDirectory $Project -WindowStyle Normal
        exit 0
    }
    $Command = 'gui'
}
& $Python -m head_rgbd.experiment.turn_app $Command @Arguments
exit $LASTEXITCODE
