# CodecDelta kurulumu (yonetici yetkisi gerekmez).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1             # kur / guncelle
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Desktop    # + masaustu kisayolu
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall  # kaldir
#
# Kaynak: bu betigin yanindaki CodecDelta.exe'li klasor (zip'ten cikarilmis
# hal) ya da depodaki dist\CodecDelta. Hedef: %LOCALAPPDATA%\Programs\CodecDelta.
# Baslat menusune kisayol eklenir. Ayarlar ve tarama onbellegi
# (%APPDATA%\CodecDelta) kaldirmada SILINMEZ; -Purge ile silinir.
param([switch]$Desktop, [switch]$Uninstall, [switch]$Purge)

$ErrorActionPreference = 'Stop'
$Target = Join-Path $env:LOCALAPPDATA 'Programs\CodecDelta'
$StartMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'CodecDelta.lnk'
$DesktopLink = Join-Path ([Environment]::GetFolderPath('Desktop')) 'CodecDelta.lnk'

function Stop-IfRunning {
    $running = Get-Process -Name 'CodecDelta' -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($Target, [StringComparison]::OrdinalIgnoreCase) }
    if ($running) { throw 'CodecDelta calisiyor; kapatip tekrar deneyin.' }
}

if ($Uninstall) {
    Stop-IfRunning
    foreach ($link in $StartMenu, $DesktopLink) {
        if (Test-Path -LiteralPath $link) { Remove-Item -LiteralPath $link -Force }
    }
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    if ($Purge) {
        $data = Join-Path $env:APPDATA 'CodecDelta'
        if (Test-Path -LiteralPath $data) { Remove-Item -LiteralPath $data -Recurse -Force }
    }
    Write-Host 'CodecDelta kaldirildi.'
    exit 0
}

$candidates = @($PSScriptRoot, (Join-Path (Split-Path -Parent $PSScriptRoot) 'dist\CodecDelta'))
$Source = $candidates | Where-Object { Test-Path -LiteralPath (Join-Path $_ 'CodecDelta.exe') } | Select-Object -First 1
if (-not $Source) { throw 'CodecDelta.exe bulunamadi: once tools\build.ps1 calistirin ya da zip''i cikarin.' }
if ([IO.Path]::GetFullPath($Source).TrimEnd('\') -ieq [IO.Path]::GetFullPath($Target).TrimEnd('\')) {
    throw 'Kaynak ile hedef ayni klasor.'
}

Stop-IfRunning
if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Path (Join-Path $Source '*') -Destination $Target -Recurse -Force
# Kaldirma icin betik kurulu klasorde de bulunsun (depodan kurulumda yoktur).
if (-not (Test-Path -LiteralPath (Join-Path $Target 'install.ps1'))) {
    Copy-Item -LiteralPath $PSCommandPath -Destination $Target -Force
}

$exe = Join-Path $Target 'CodecDelta.exe'
$shell = New-Object -ComObject WScript.Shell
$links = @($StartMenu)
if ($Desktop) { $links += $DesktopLink }
foreach ($path in $links) {
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $exe
    $link.WorkingDirectory = $Target
    $link.Description = 'CodecDelta - measure and hear what a lossy encoder did to your audio'
    $link.Save()
}
Write-Host "Kuruldu: $exe"
Write-Host "Baslat menusu: $StartMenu"
if ($Desktop) { Write-Host "Masaustu: $DesktopLink" }
Write-Host 'ffmpeg bulunamazsa: powershell -ExecutionPolicy Bypass -File' (Join-Path $Target 'setup-ffmpeg.ps1')
