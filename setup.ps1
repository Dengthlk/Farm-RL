# Only used by a source-code download. The Windows release already includes runtime/.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtime = Join-Path $PSScriptRoot 'runtime'
$python = Join-Path $runtime 'python.exe'
$packages = Join-Path $runtime 'Lib/site-packages'
$tmp = Join-Path $env:TEMP 'FarmRL-download'
New-Item -ItemType Directory -Force -Path $tmp,$packages | Out-Null
Write-Host 'Preparing Farm RL for Windows x64 (first run only)...'
if (-not (Test-Path -LiteralPath $python)) {
    $zip = Join-Path $tmp 'python-embed.zip'
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.13.9/python-3.13.9-embed-amd64.zip' -OutFile $zip -UseBasicParsing
    Expand-Archive -LiteralPath $zip -DestinationPath $runtime -Force
}
@('python313.zip','.','Lib/site-packages','../game','import site') | Set-Content -Encoding ASCII -LiteralPath (Join-Path $runtime 'python313._pth')
$wheel = (Invoke-RestMethod 'https://pypi.org/pypi/pip/25.3/json').urls | Where-Object filename -like '*.whl' | Select-Object -First 1
$wheelZip = Join-Path $tmp 'pip.zip'
Invoke-WebRequest $wheel.url -OutFile $wheelZip -UseBasicParsing
Expand-Archive -LiteralPath $wheelZip -DestinationPath $packages -Force
& $python -m pip install --disable-pip-version-check --no-compile --target $packages 'numpy==2.5.4' 'https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp313-cp313-win_amd64.whl'
if ($LASTEXITCODE -ne 0) { throw 'Runtime download failed. Run Play.cmd again to retry.' }
& $python -B -c 'import torch, numpy'
if ($LASTEXITCODE -ne 0) { throw 'Unable to start the CPU runtime. Use the self-contained Windows release.' }
Set-Content -LiteralPath (Join-Path $runtime 'ready.txt') -Encoding ASCII -Value 'Windows x64 CPU'
