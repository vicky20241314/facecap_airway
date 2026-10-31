param(
    [string]$Runtime = 'D:\proiect\withB-runtime',
    [Alias('h')]
    [switch]$Help,
    [switch]$Version,
    [Parameter(Position = 0)]
    [string]$Command,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Arguments = @()
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
if ($Version) {
    $Command = '--version'
}
elseif ($Help) {
    if ([string]::IsNullOrWhiteSpace($Command)) {
        $Command = '--help'
    }
    else {
        $Arguments = @($Arguments) + @('--help')
    }
}
elseif ([string]::IsNullOrWhiteSpace($Command)) {
    $Command = '--help'
}
$Python = Join-Path $Runtime 'env\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Runtime Python not found: $Python"
}
$env:HEADRGBD_RUNTIME_DIR = (Resolve-Path -LiteralPath $Runtime).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m head_rgbd --runtime $env:HEADRGBD_RUNTIME_DIR $Command @Arguments
    $Code = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $Code
