@echo off
rem VoiceGym launcher (Windows, PowerShell or cmd):  .\voicegym doctor | setup | benchmark | sft | dpo
rem Installs a pinned uv into .voicegym\bin the first time, then runs the CLI with a uv-managed Python 3.12.
setlocal
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "UV_VERSION=0.12.16"
set "VG_HOME=%ROOT%\.voicegym"
set "UV=%VG_HOME%\bin\uv.exe"

if not exist "%UV%" (
  echo voicegym: installing uv %UV_VERSION% into .voicegym\bin ^(one time^) 1>&2
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR='%VG_HOME%\bin'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex"
  if not exist "%UV%" (
    echo voicegym: could not install uv 1>&2
    exit /b 1
  )
)

set "UV_PYTHON_INSTALL_DIR=%VG_HOME%\python"
set "UV_CACHE_DIR=%VG_HOME%\cache\uv"
set "PYTHONPATH=%ROOT%"
set "VOICEGYM_UV=%UV%"
"%UV%" run --quiet --no-project --python 3.12 --python-preference only-managed --with pyyaml==6.0.2 python -m voicegym_cli %*
exit /b %errorlevel%
