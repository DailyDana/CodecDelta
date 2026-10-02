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

if /i "%~1"=="--console" goto console

rem Yollar `cd` ONCESI mutlaga cevrilir: sonra proje klasorune gore cozulup
rem bulunamiyorlardi (denetim D24).
call :absolute "%~1" "%~2"
cd /d "%ROOT%"
start "" "%PYW%" -m app "%A1%" "%A2%"
exit /b 0

:console
rem Parantezli bir blok icinde %%1, blok AYRISTIRILIRKEN genisletilir; shift
rem henuz calismamis olur ve "--console" dosya adi diye uygulamaya gecerdi.
rem Bu yuzden etiket + goto.
shift
call :absolute "%~1" "%~2"
cd /d "%ROOT%"
"%PY%" -m app "%A1%" "%A2%"
set "CODE=%ERRORLEVEL%"
echo.
echo Uygulama kapandi, cikis kodu %CODE%.
pause
exit /b %CODE%

:absolute
rem Bos arguman bos kalir; uygulama bos argumanlari yok sayar.
set "A1="
set "A2="
if not "%~1"=="" set "A1=%~f1"
if not "%~2"=="" set "A2=%~f2"
exit /b 0
