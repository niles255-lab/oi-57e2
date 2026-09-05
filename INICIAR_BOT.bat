@echo off
REM ============================================================
REM INICIAR_BOT.bat - Atalho para iniciar o BlockBlaster Bot
REM ============================================================
REM Este arquivo deve estar na raiz do projeto (mesma pasta do bot.py)
REM Funciona independentemente da pasta atual do terminal.
REM ============================================================

REM Entra na pasta onde este .bat está localizado
cd /d "%~dp0"

echo ============================================================
echo BlockBlaster Bot  |  Redmi Note 8 / Blocks of Bitcoin
echo ============================================================
echo Pasta do projeto: %~dp0
echo Iniciando bot.py ...
echo.

REM Executa o bot (mantém janela aberta ao finalizar)
python bot.py

echo.
echo ============================================================
echo Bot finalizado. Pressione qualquer tecla para fechar.
echo ============================================================
pause >nul