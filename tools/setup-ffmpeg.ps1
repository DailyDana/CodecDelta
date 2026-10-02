# CodecDelta icin ffmpeg kurulumu (yonetici yetkisi gerekmez).
#
#   powershell -ExecutionPolicy Bypass -File setup-ffmpeg.ps1           # indir + dogrula + kur
#   powershell -ExecutionPolicy Bypass -File setup-ffmpeg.ps1 -Force    # varsa da yeniden indir
#   powershell -ExecutionPolicy Bypass -File setup-ffmpeg.ps1 -DryRun   # yalniz indirme adresini coz
#
# BtbN'in resmi statik GPL yapimini indirir; yalniz bin\ffmpeg.exe ve
# bin\ffprobe.exe cikarilir ve %LOCALAPPDATA%\CodecDelta\bin'e (CodecDelta'nin
# arama yollarindan biri) kopyalanir. Kurmadan once dogrulanir: calisiyor mu,
# aresample/astats filtreleri ve libsoxr var mi (karsilastirma zinciri soxr
# kullanir). ffmpeg GPL'dir ve CodecDelta paketine girmez.
#
# Adres cozumleme ve indirme yardimcilari Aniflow'un setup.ps1'inden
# (github.com/DailyDana/Aniflow) alindi.
# Cikis kodu: 0 = kurulu, 1 = kurulamadi.
param([switch]$Force, [switch]$DryRun)

$ErrorActionPreference = 'Stop'
# PS 5.1'de ilerleme cubugu buyuk indirmeleri kat kat yavaslatir
$ProgressPreference = 'SilentlyContinue'
# TLS 1.2'yi mevcut protokollere EKLE. Deger 0 (SystemDefault) ise isletim sistemi
# zaten TLS 1.2/1.3 secer; oraya Tls12 yazmak TLS 1.3'u kapatirdi, dokunma.
$sp = [Net.ServicePointManager]::SecurityProtocol
if ([int]$sp -ne 0) {
    [Net.ServicePointManager]::SecurityProtocol = $sp -bor [Net.SecurityProtocolType]::Tls12
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Bin = Join-Path $env:LOCALAPPDATA 'CodecDelta\bin'
$ff = Join-Path $Bin 'ffmpeg.exe'
$fp = Join-Path $Bin 'ffprobe.exe'

# Once .part dosyasina indirir; bos degilse ve (verildiyse) SHA256 tutarsa asil ada tasir.
# Ag hatasi ya da hash uyusmazliginda 3 kez dener.
function Get-Download([string]$Url, [string]$Out, [string]$Sha256 = '') {
    $part  = $Out + '.part'
    $tries = 3
    for ($i = 1; $i -le $tries; $i++) {
        try {
            Write-Host "Indiriliyor: $Url"
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
            if (-not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -le 0) {
                throw 'indirilen dosya bos'
            }
            if ($Sha256) {
                $h = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash
                if ($h -ne $Sha256) { throw "SHA256 uyusmuyor: beklenen $Sha256, gelen $h" }
                Write-Host '  SHA256 OK'
            }
            Move-Item -LiteralPath $part -Destination $Out -Force
            return
        } catch {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
            if ($i -ge $tries) { throw }
            Write-Warning "Deneme $i/$tries basarisiz: $($_.Exception.Message) - tekrar deneniyor..."
            Start-Sleep -Seconds (3 * $i)
        }
    }
}

# Zip'ten yalniz $Keep (param: goreli yol, '\' ayracli) $true donen dosyalari cikarir.
# Hedef disina tasan girdileri (zip-slip) reddeder. Cikarilan dosya sayisini dondurur.
function Expand-ZipFiltered([string]$Zip, [string]$Dest, [scriptblock]$Keep) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $destFull = [IO.Path]::GetFullPath($Dest).TrimEnd('\')
    $za = [IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $n = 0
        foreach ($e in $za.Entries) {
            if (-not $e.Name) { continue }   # klasor girdisi
            $rel = $e.FullName -replace '/', '\'
            if (-not (& $Keep $rel)) { continue }
            $target = [IO.Path]::GetFullPath((Join-Path $destFull $rel))
            if (-not $target.StartsWith($destFull + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "Zip girdisi hedef klasorun disina isaret ediyor: $rel"
            }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($e, $target, $true)
            $n++
        }
        return $n
    } finally {
        $za.Dispose()
    }
}

# Harici programi calistirir; ciktiyi ve cikis kodunu dondurur (stderr hata sayilmaz).
function Invoke-Native([string]$Exe, [string[]]$ArgList) {
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $Exe @ArgList 2>&1 | Out-String -Width 4096
        return @{ Code = $LASTEXITCODE; Out = $out }
    } catch {
        return @{ Code = -1; Out = $_.Exception.Message }
    } finally {
        $ErrorActionPreference = $eap
    }
}

# CodecDelta'nin ihtiyac duydugu yetenekler bu ffmpeg'de var mi? Eksikleri dondurur.
function Get-MissingCaps([string]$Exe) {
    $missing = New-Object System.Collections.Generic.List[string]
    $ver = Invoke-Native $Exe @('-hide_banner', '-version')
    if ($ver.Code -ne 0) { $missing.Add("calismiyor (cikis kodu $($ver.Code))"); return $missing }
    if ($ver.Out -notmatch '--enable-libsoxr') { $missing.Add('libsoxr') }
    $flt = Invoke-Native $Exe @('-hide_banner', '-filters')
    foreach ($name in 'aresample', 'astats') {
        if ($flt.Out -notmatch "(?m)^\s*\S+\s+$name\s") { $missing.Add($name) }
    }
    return $missing
}

# BtbN ffmpeg adresini (ve varsa SHA256 ozetini) cozer. ffmpeg her gun yeniden
# derlendigi icin pinlenemez; GitHub API "digest" alani varsa onunla dogrulanir.
#  1) API: son surumler icinde sabit adli "ffmpeg-master-latest-win64-gpl.zip" ya da
#     tarihli autobuild adi "ffmpeg-N-<sayi>-g<hash>-win64-gpl.zip" (+ digest)
#  2) API yoksa (hiz limiti vb.) sabit adresler; ikisi de "latest" etiketine gider:
#       releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
#       releases/latest/download/ffmpeg-master-latest-win64-gpl.zip
#  3) O da olmazsa surum sayfasindaki en yeni autobuild-* etiketinin asset listesi (HTML)
function Resolve-FfmpegUrl {
    $repo    = 'https://github.com/BtbN/FFmpeg-Builds'
    $fixName = 'ffmpeg-master-latest-win64-gpl.zip'
    $rxName  = '^(ffmpeg-master-latest-win64-gpl|ffmpeg-N-\d+-g[0-9a-f]+-win64-gpl)\.zip$'
    try {
        $rels = Invoke-RestMethod 'https://api.github.com/repos/BtbN/FFmpeg-Builds/releases?per_page=5' -UseBasicParsing
        foreach ($rel in @($rels)) {
            $a = @($rel.assets | Where-Object { $_.name -match $rxName }) | Select-Object -First 1
            if ($a) {
                $sha = ''
                if ($a.digest -and "$($a.digest)" -match '^sha256:([0-9a-fA-F]{64})$') { $sha = $Matches[1] }
                else { Write-Warning 'API ffmpeg icin SHA256 ozeti vermedi; indirme hash dogrulamasiz yapilacak.' }
                return @{ Url = $a.browser_download_url; Sha256 = $sha }
            }
        }
    } catch {
        Write-Host "  (GitHub API yanit vermedi: $($_.Exception.Message))"
    }
    Write-Warning 'ffmpeg icin SHA256 ozeti alinamadi; indirme hash dogrulamasiz yapilacak.'
    foreach ($u in @("$repo/releases/download/latest/$fixName", "$repo/releases/latest/download/$fixName")) {
        try {
            $null = Invoke-WebRequest -Uri $u -Method Head -UseBasicParsing
            return @{ Url = $u; Sha256 = '' }
        } catch {
            Write-Host "  (yanit yok: $u)"
        }
    }
    try {
        $page = (Invoke-WebRequest "$repo/releases" -UseBasicParsing).Content
        if ($page -match 'expanded_assets/(autobuild-[^"/?#]+)') {
            $html = (Invoke-WebRequest "$repo/releases/expanded_assets/$($Matches[1])" -UseBasicParsing).Content
            if ($html -match 'href="([^"]*/ffmpeg-N-[^"/]*-win64-gpl\.zip)"') {
                $u = $Matches[1]
                if ($u -notmatch '^https?:') { $u = 'https://github.com' + $u }
                return @{ Url = $u; Sha256 = '' }
            }
        }
    } catch {
        Write-Host "  (surum sayfasi okunamadi: $($_.Exception.Message))"
    }
    throw ('BtbN ffmpeg indirme adresi cozulemedi. Elle kurulum: https://github.com/BtbN/FFmpeg-Builds/releases ' +
           "adresinden `"win64-gpl.zip`" dosyasini indirip icindeki ffmpeg.exe ve ffprobe.exe'yi $Bin klasorune kopyalayin.")
}

if ($DryRun) {
    $src = Resolve-FfmpegUrl
    Write-Host "Adres : $($src.Url)"
    Write-Host ("SHA256: " + $(if ($src.Sha256) { $src.Sha256 } else { '(yok)' }))
    Write-Host "Hedef : $Bin"
    exit 0
}

if (-not $Force -and (Test-Path -LiteralPath $ff) -and (Test-Path -LiteralPath $fp)) {
    $missing = Get-MissingCaps $ff
    if ($missing.Count -eq 0) { Write-Host "ffmpeg zaten kurulu: $Bin"; exit 0 }
    Write-Host "Kurulu ffmpeg eksik: $($missing -join ', ') - yeniden indiriliyor."
}

# her calismada temiz, benzersiz gecici klasor: eski kalintilar yanlislikla secilmesin
$Tmp = Join-Path $env:TEMP ('CodecDelta-setup-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
try {
    $src = Resolve-FfmpegUrl
    $zip = Join-Path $Tmp 'ffmpeg.zip'
    Get-Download $src.Url $zip $src.Sha256
    # yalniz bin\ffmpeg.exe + bin\ffprobe.exe; once gecici klasore, dogrulanirsa hedefe
    $stage = Join-Path $Tmp 'ffmpeg'
    $null = Expand-ZipFiltered $zip $stage { param($p) $p -match '(^|\\)bin\\(ffmpeg|ffprobe)\.exe$' }
    $exe = @(Get-ChildItem -LiteralPath $stage -Recurse -Filter 'ffmpeg.exe')
    $prb = @(Get-ChildItem -LiteralPath $stage -Recurse -Filter 'ffprobe.exe')
    if ($exe.Count -ne 1 -or $prb.Count -ne 1) {
        throw "zip icinde birer ffmpeg.exe/ffprobe.exe bekleniyordu, bulunan: $($exe.Count)/$($prb.Count)"
    }
    $missing = Get-MissingCaps $exe[0].FullName
    if ($missing.Count -gt 0) { throw "indirilen ffmpeg uygun degil, eksik: $($missing -join ', ')" }
    $probe = Invoke-Native $prb[0].FullName @('-hide_banner', '-version')
    if ($probe.Code -ne 0) { throw "indirilen ffprobe calismadi (cikis kodu $($probe.Code))" }

    New-Item -ItemType Directory -Force -Path $Bin | Out-Null
    Copy-Item -LiteralPath $exe[0].FullName -Destination $ff -Force
    Copy-Item -LiteralPath $prb[0].FullName -Destination $fp -Force
    Write-Host "ffmpeg kuruldu: $Bin"
    Write-Host 'CodecDelta bir sonraki acilista bu kopyayi bulur (Ayarlar''da baska bir klasor secili degilse).'
    exit 0
} catch {
    Write-Warning "ffmpeg kurulamadi: $($_.Exception.Message)"
    exit 1
} finally {
    Remove-Item -LiteralPath $Tmp -Recurse -Force -ErrorAction SilentlyContinue
}
