@echo off
rem CodecDelta'yi baslatir.
rem
rem Kullanim:
rem   - Cift tikla: bos pencere acilir.
rem   - Bir ya da iki ses dosyasini bu dosyanin UZERINE surukle: ilki referans,
rem     ikincisi test olarak yuklenir.
rem   - Komut satirindan: CodecDelta.bat referans.flac test.opus
rem
rem Konsol penceresi acik kalmasin diye pythonw ile baslatilir. Uygulama
rem acilmiyorsa hatayi gormek icin:  CodecDelta.bat --console

setlocal
set "ROOT=%~dp0"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Sanal ortam bulunamadi: "%ROOT%.venv"
    echo.
    echo Once proje klasorunde su komutlari calistirin:
    echo   uv venv --python 3.14 .venv
    echo   uv pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

cd /d "%ROOT%"

if /i "%~1"=="--console" goto console

start "" "%PYW%" -m app %*
exit /b 0

:console
rem Parantezli bir blok icinde %%1, blok AYRISTIRILIRKEN genisletilir; shift
rem henuz calismamis olur ve "--console" dosya adi diye uygulamaya gecerdi.
rem Bu yuzden etiket + goto.
shift
"%PY%" -m app %1 %2
set "CODE=%ERRORLEVEL%"
echo.
echo Uygulama kapandi, cikis kodu %CODE%.
pause
exit /b %CODE%
