# CodecDelta paketleme: PyInstaller ile tek klasorlu, pencereli Windows yapisi.
#
# Kullanim (depo kokunden ya da herhangi bir yerden):
#   powershell -ExecutionPolicy Bypass -File tools\build.ps1          # yapi + duman testi + zip
#   powershell -ExecutionPolicy Bypass -File tools\build.ps1 -NoZip
#
# Cikti: dist\CodecDelta\CodecDelta.exe ve dist\CodecDelta-<surum>-win64.zip
#
# ffmpeg PAKETE GIRMEZ (~290 MB acik): uygulama ilk acilista bulur (winget,
# PATH, %LOCALAPPDATA%\CodecDelta\bin, exe'nin yanindaki bin\). Yoksa
# tools\setup-ffmpeg.ps1 indirir.
#
# Yapi sonunda exe bir "duman testi" ile calistirilir (--smoke-test): pencere
# kurulur, surum ve bulunan ffmpeg JSON'a yazilir, uygulama hemen kapanir.
# Test gecmezse betik hata verir; bozuk bir yapi zip'lenmez.
#
# -Python <yol>: .venv yerine baska bir yorumlayici (CI'da sistem Python'u).
param([switch]$NoZip, [string]$Python = '')

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Py = if ($Python) { $Python } else { Join-Path $Root '.venv\Scripts\python.exe' }
if (-not $Python -and -not (Test-Path -LiteralPath $Py)) {
    throw "Sanal ortam yok: $Py  (uv venv --python 3.14 .venv; uv pip install -e .[dev]) ya da -Python verin"
}
Push-Location $Root
try {
    $version = (& $Py -c "import app; print(app.__version__)").Trim()
    Write-Host "CodecDelta $version paketleniyor..."
    foreach ($dir in 'build', 'dist') {
        if (Test-Path -LiteralPath $dir) { Remove-Item -LiteralPath $dir -Recurse -Force }
    }

    $templates = Join-Path $Root 'app\report\templates'
    $assets = Join-Path $Root 'app\ui\assets'
    & $Py -m PyInstaller --noconfirm --clean --windowed `
        --name CodecDelta `
        --icon (Join-Path $assets 'codecdelta.ico') `
        --add-data "$assets;app\ui\assets" `
        --distpath (Join-Path $Root 'dist') `
        --workpath (Join-Path $Root 'build') `
        --specpath (Join-Path $Root 'build') `
        --add-data "$templates;app\report\templates" `
        (Join-Path $Root 'tools\entry.py')
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller basarisiz (cikis kodu $LASTEXITCODE)" }

    $out = Join-Path $Root 'dist\CodecDelta'
    $exe = Join-Path $out 'CodecDelta.exe'
    Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination $out
    Copy-Item -LiteralPath (Join-Path $Root 'README.md') -Destination $out
    Copy-Item -LiteralPath (Join-Path $Root 'tools\setup-ffmpeg.ps1') -Destination $out
    Copy-Item -LiteralPath (Join-Path $Root 'tools\install.ps1') -Destination $out

    # ---- duman testi ----
    $smoke = Join-Path $env:TEMP ("codecdelta-smoke-" + [guid]::NewGuid().ToString('N') + '.json')
    $env:QT_QPA_PLATFORM = 'offscreen'
    try {
        $proc = Start-Process -FilePath $exe -ArgumentList @('--smoke-test', "`"$smoke`"") -Wait -PassThru
    } finally {
        Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $smoke)) { throw "Duman testi rapor yazmadi (cikis kodu $($proc.ExitCode))" }
    $report = Get-Content -LiteralPath $smoke -Raw -Encoding UTF8 | ConvertFrom-Json
    Remove-Item -LiteralPath $smoke -Force
    if ($report.version -ne $version) { throw "Duman testi surumu $($report.version), beklenen $version" }
    if ($report.report -ne $true) { throw "Rapor sablonu yuklenemedi: $($report.report)" }
    if (@($report.icon).Count -eq 0) { throw 'Pencere ikonu yuklenemedi (app\ui\assets\codecdelta.ico)' }
    if ($null -eq $report.audio) {
        Write-Warning '  duman testi: ses cikis cihazi yok, ABX sesi denenemedi'
    } elseif (-not $report.audio.opened -or $report.audio.error -ne 'NoError') {
        throw "Ses cikisi kurulamadi ($($report.audio.device)): $($report.audio.error)"
    } else {
        Write-Host "  ses cikisi: $($report.audio.device)"
    }
    if ($report.ffmpeg) {
        Write-Host "  duman testi: surum $($report.version), $($report.tabs) sekme, ffmpeg: $($report.ffmpeg)"
    } else {
        # ffmpeg'i olmayan bir yapi makinesinde exe yine de calisti; uyari yeter.
        Write-Warning "  duman testi: exe calisti ama ffmpeg bulunamadi ($($report.error))"
    }

    if (-not $NoZip) {
        $zip = Join-Path $Root "dist\CodecDelta-$version-win64.zip"
        Compress-Archive -Path (Join-Path $out '*') -DestinationPath $zip -Force
        $mb = [math]::Round((Get-Item -LiteralPath $zip).Length / 1MB, 1)
        Write-Host "Zip: $zip ($mb MB)"
    }
    Write-Host "Tamam: $exe"
} finally {
    Pop-Location
}
